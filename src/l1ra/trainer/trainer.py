import logging
from peft.mapping import PEFT_TYPE_TO_TUNER_MAPPING
import peft
from trl import SFTTrainer
from l1ra.tuner.model import L1RAModel
from transformers import Trainer
from torch import nn


logger = logging.getLogger(__name__)


PEFT_TYPE_TO_TUNER_MAPPING.update(
    {"L1RA": L1RAModel}
)


class L1RASFTTrainer(SFTTrainer):
    """
    Class definition of the L1RA Trainer based on the Supervised Finetuning Trainer (SFT Trainer).
    This class is a wrapper around the `trl.SFTTranier` class and inherits all of its attributes and methods.
    The trainer takes care of properly initializing the PeftModel in case a user passes a `L1RAConfig` object.

    Args:
        model (Union[`transformers.PreTrainedModel`, `nn.Module`, `str`]):
            The model to train, can be a `PreTrainedModel`, a `torch.nn.Module` or a string with the model name to
            load from cache or download. The model can be also converted to a `PeftModel` if a `PeftConfig` object is
            passed to the `peft_config` argument.
        args (Optional[`SFTConfig`]):
            The arguments to tweak for training. Will default to a basic instance of [`SFTConfig`] with the `output_dir`
            set to a directory named *tmp_trainer* in the current directory if not provided.
        data_collator (Optional[`transformers.DataCollator`]):
            The data collator to use for training.
        train_dataset (Optional[`datasets.Dataset`]):
            The dataset to use for training. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        eval_dataset (Optional[Union[`datasets.Dataset`, Dict[`str`, `datasets.Dataset`]]]):
            The dataset to use for evaluation. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        tokenizer (Optional[`transformers.PreTrainedTokenizer`]):
            The tokenizer to use for training. If not specified, the tokenizer associated to the model will be used.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        compute_metrics (`Callable[[transformers.EvalPrediction], Dict]`, *optional* defaults to None):
            The function used to compute metrics during evaluation. It should return a dictionary mapping metric names to metric values.
            If not specified, only the loss will be computed during evaluation.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Optional[PeftConfig]`):
            The L1RAConfig object to use to initialize the L1RAModel.
        formatting_func (`Optional[Callable]`):
            The formatting function to be used for creating the `ConstantLengthDataset`.
    """

    def __init__(self, *args, **kwargs):
        self.real_step = 0
        super().__init__(*args, **kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.num_training_steps = num_training_steps
        return super().create_optimizer_and_scheduler(num_training_steps)

    def create_optimizer(self):
        c_vectors = []
        AB_parameters = []
        other_parameters = []

        for name, param in self.model.named_parameters():
            if "lora_c" in name:
                c_vectors.append(param)
            elif param.requires_grad:
                AB_parameters.append(param)
            else:
                other_parameters.append(param)

        optimizer_grouped_parameters = [
            {
                "params": c_vectors,
                "weight_decay": 0.0,
                "lr": list(self.model.peft_config.values())[0].eta_c
            },
            {
                "params": AB_parameters,
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate
            },
            {
                "params": other_parameters,
                "weight_decay": 0.0,
            },
        ]

        # The following code is partially copied from
        # github.com/huggingface/transformers/blob/main/src/transformers/trainer.py

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)

        # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
        # e.g. for GaLore optimizer.
        if "params" in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop("params")

        # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
        # e.g. for LOMO optimizer.
        if "model" in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop("model")

        # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
        # to avoid arguments conflicts.
        if "optimizer_dict" in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if optimizer_cls.__name__ == "Adam8bit":
            import bitsandbytes

            manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

            skipped = 0
            for module in self.model.modules():
                if isinstance(module, nn.Embedding):
                    skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                    logger.info(f"skipped {module}: {skipped/2**20}M params")
                    manager.register_module_override(module, "weight", {"optim_bits": 32})
                    logger.debug(f"bitsandbytes: will optimize {module} in fp32")
            logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def restart_optimizer(self):
        learning_rates = []
        for p in self.optimizer.param_groups:
            learning_rates.append(p["lr"])

        self.optimizer = self.create_optimizer()
        for p, lr in zip(self.optimizer.param_groups, learning_rates):
            p["lr"] = lr

        self.lr_scheduler.optimizer = self.optimizer

    def training_step(self, model, inputs):
        num_training_steps = self.num_training_steps * self.args.gradient_accumulation_steps
        updated = model.update_ranks(self.real_step, num_training_steps)
        if updated:
            self.restart_optimizer()
        self.real_step += 1
        return super().training_step(model, inputs)


# The functiuons overrides are the same as the L1RASFTTrainer
class L1RATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.real_step = 0
        super().__init__(*args, **kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.num_training_steps = num_training_steps
        return super().create_optimizer_and_scheduler(num_training_steps)

    def create_optimizer(self):
        c_vectors = []
        AB_parameters = []
        other_parameters = []

        for name, param in self.model.named_parameters():
            if "lora_c" in name:
                c_vectors.append(param)
            elif param.requires_grad:
                AB_parameters.append(param)
            else:
                other_parameters.append(param)

        optimizer_grouped_parameters = [
            {
                "params": c_vectors,
                "weight_decay": 0.0,
                "lr": list(self.model.peft_config.values())[0].eta_c
            },
            {
                "params": AB_parameters,
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate
            },
            {
                "params": other_parameters,
                "weight_decay": 0.0,
            },
        ]

        # The following code is partially copied from
        # github.com/huggingface/transformers/blob/main/src/transformers/trainer.py

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)

        # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
        # e.g. for GaLore optimizer.
        if "params" in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop("params")

        # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
        # e.g. for LOMO optimizer.
        if "model" in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop("model")

        # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
        # to avoid arguments conflicts.
        if "optimizer_dict" in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if optimizer_cls.__name__ == "Adam8bit":
            import bitsandbytes

            manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

            skipped = 0
            for module in self.model.modules():
                if isinstance(module, nn.Embedding):
                    skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                    logger.info(f"skipped {module}: {skipped/2**20}M params")
                    manager.register_module_override(module, "weight", {"optim_bits": 32})
                    logger.debug(f"bitsandbytes: will optimize {module} in fp32")
            logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def restart_optimizer(self):
        learning_rates = []
        for p in self.optimizer.param_groups:
            learning_rates.append(p["lr"])

        self.optimizer = self.create_optimizer()
        for p, lr in zip(self.optimizer.param_groups, learning_rates):
            p["lr"] = lr

        self.lr_scheduler.optimizer = self.optimizer

    def training_step(self, model, inputs):
        num_training_steps = self.num_training_steps * self.args.gradient_accumulation_steps
        updated = model.update_ranks(self.real_step, num_training_steps)
        if updated:
            self.restart_optimizer()
        self.real_step += 1
        return super().training_step(model, inputs)