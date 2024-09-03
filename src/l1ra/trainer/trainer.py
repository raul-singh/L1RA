import importlib.metadata
import logging
from enum import Enum
from typing import Any, Optional, Tuple

from packaging import version
from peft.mapping import PEFT_TYPE_TO_TUNER_MAPPING
from torch import nn
from transformers import PreTrainedModel, Trainer, TrainingArguments
from transformers.utils import is_bitsandbytes_available
from trl import SFTTrainer

from l1ra.tuner.model import L1RAModel

logger = logging.getLogger(__name__)


PEFT_TYPE_TO_TUNER_MAPPING.update(
    {"L1RA": L1RAModel}
)

class AdamEOptimizers(Enum):
    ADAME = "adame"
    ADAME_BNB = "adame_bnb"
    ADAME_8BIT = "adame_8bit"
    ADAME_BNB_8BIT = "adame_bnb_8bit"
    PAGED_ADAME = "paged_adame_32bit"
    PAGED_ADAME_8BIT = "paged_adame_8bit"


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

        # TODO Hardcoded default name, must change
        lasso_coef = self.model.peft_config["default"].l1ra_lambda

        optimizer_grouped_parameters = [
            {
                "params": c_vectors,
                "weight_decay": 0.0,
                "lr": list(self.model.peft_config.values())[0].eta_c,
                "lasso": lasso_coef,
            },
            {
                "params": AB_parameters,
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate,
                "lasso": 0.0,
            },
            {
                "params": other_parameters,
                "weight_decay": 0.0,
                "lasso": 0.0
            },
        ]

        # The following code is partially copied from
        # github.com/huggingface/transformers/blob/main/src/transformers/trainer.py

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)

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

    def create_scheduler(self, *args, **kwargs):
        scheduler = super().create_scheduler(*args, **kwargs)
        scheduler.lr_lambdas[0] = lambda _: 1
        return scheduler

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

    @staticmethod
    def get_optimizer_cls_and_kwargs(
        args: TrainingArguments, model: Optional[PreTrainedModel] = None
    ) -> Tuple[Any, Any]:
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`transformers.training_args.TrainingArguments`):
                The training arguments for the training session.

        """

        if args.optim not in [o.value for o in AdamEOptimizers]:
            logger.warning(
                "You are using the L1RA trainer wihtout using the AdamE optimizer. "
                "If you are training a L1RA model, you should use the AdamE optimizer or "
                "one of its variants, otherwise L1RA won't work."
                )
            return Trainer.get_optimizer_cls_and_kwargs(args, model)

        # parse args.optim_args
        optim_args = {}
        if args.optim_args:
            for mapping in args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[key] = value

        optimizer_kwargs = {"lr": args.learning_rate}

        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        if args.optim in [AdamEOptimizers.ADAME, AdamEOptimizers.ADAME_BNB]:
            from bitsandbytes import AdamE

            optimizer_cls = AdamE
            optimizer_kwargs.update(adam_kwargs)

        elif args.optim in [
            AdamEOptimizers.ADAME_BNB_8BIT,
            AdamEOptimizers.ADAME_8BIT,
            AdamEOptimizers.PAGED_ADAME,
            AdamEOptimizers.PAGED_ADAME_8BIT,
        ]:
            try:
                from bitsandbytes.optim import AdamE

                is_paged = False
                optim_bits = 32
                optimizer_cls = None
                additional_optim_kwargs = adam_kwargs
                if "paged" in args.optim:
                    is_paged = True
                if "8bit" in args.optim:
                    optim_bits = 8
                if "adam" in args.optim:
                    optimizer_cls = AdamE

                bnb_kwargs = {"optim_bits": optim_bits}
                if "rmsprop" not in args.optim:
                    bnb_kwargs["is_paged"] = is_paged

                optimizer_kwargs.update(additional_optim_kwargs)
                optimizer_kwargs.update(bnb_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate bnb optimizer but `bitsandbytes` is not installed!")
            if is_bitsandbytes_available() and version.parse(
                importlib.metadata.version("bitsandbytes")
            ) < version.parse("0.41.1"):
                logger.warning(
                    "You are using 8-bit optimizers with a version of `bitsandbytes` < 0.41.1. "
                    "It is recommended to update your version as a major bug has been fixed in 8-bit optimizers."
                )
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
        return optimizer_cls, optimizer_kwargs


# The functions overrides are the same as the L1RASFTTrainer
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

        # TODO Hardcoded default name, must change
        lasso_coef = self.model.peft_config["default"].l1ra_lambda

        optimizer_grouped_parameters = [
            {
                "params": c_vectors,
                "weight_decay": 0.0,
                "lr": list(self.model.peft_config.values())[0].eta_c,
                "lasso": lasso_coef,
            },
            {
                "params": AB_parameters,
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate,
                "lasso": 0.0,
            },
            {
                "params": other_parameters,
                "weight_decay": 0.0,
                "lasso": 0.0
            },
        ]

        # The following code is partially copied from
        # github.com/huggingface/transformers/blob/main/src/transformers/trainer.py

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)

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

    def create_scheduler(self, *args, **kwargs):
        scheduler = super().create_scheduler(*args, **kwargs)
        scheduler.lr_lambdas[0] = lambda _: 1
        return scheduler

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

    @staticmethod
    def get_optimizer_cls_and_kwargs(
        args: TrainingArguments, model: Optional[PreTrainedModel] = None
    ) -> Tuple[Any, Any]:
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`transformers.training_args.TrainingArguments`):
                The training arguments for the training session.

        """

        if args.optim not in [o.value for o in AdamEOptimizers]:
            logger.warning(
                "You are using the L1RA trainer wihtout using the AdamE optimizer. "
                "If you are training a L1RA model, you should use the AdamE optimizer or "
                "one of its variants, otherwise L1RA won't work."
                )
            return Trainer.get_optimizer_cls_and_kwargs(args, model)

        # parse args.optim_args
        optim_args = {}
        if args.optim_args:
            for mapping in args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[key] = value

        optimizer_kwargs = {"lr": args.learning_rate}

        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        if args.optim in [AdamEOptimizers.ADAME, AdamEOptimizers.ADAME_BNB]:
            from bitsandbytes import AdamE

            optimizer_cls = AdamE
            optimizer_kwargs.update(adam_kwargs)

        elif args.optim in [
            AdamEOptimizers.ADAME_BNB_8BIT,
            AdamEOptimizers.ADAME_8BIT,
            AdamEOptimizers.PAGED_ADAME,
            AdamEOptimizers.PAGED_ADAME_8BIT,
        ]:
            try:
                from bitsandbytes.optim import AdamE

                is_paged = False
                optim_bits = 32
                optimizer_cls = None
                additional_optim_kwargs = adam_kwargs
                if "paged" in args.optim:
                    is_paged = True
                if "8bit" in args.optim:
                    optim_bits = 8
                if "adam" in args.optim:
                    optimizer_cls = AdamE

                bnb_kwargs = {"optim_bits": optim_bits}
                if "rmsprop" not in args.optim:
                    bnb_kwargs["is_paged"] = is_paged

                optimizer_kwargs.update(additional_optim_kwargs)
                optimizer_kwargs.update(bnb_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate bnb optimizer but `bitsandbytes` is not installed!")
            if is_bitsandbytes_available() and version.parse(
                importlib.metadata.version("bitsandbytes")
            ) < version.parse("0.41.1"):
                logger.warning(
                    "You are using 8-bit optimizers with a version of `bitsandbytes` < 0.41.1. "
                    "It is recommended to update your version as a major bug has been fixed in 8-bit optimizers."
                )
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
        return optimizer_cls, optimizer_kwargs
