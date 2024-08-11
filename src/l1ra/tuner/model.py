import warnings
from itertools import chain
import re

import torch
from transformers.pytorch_utils import Conv1D

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.lora import LoraConfig, LoraModel
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
    _freeze_adapter,
    _get_submodules,
    get_auto_gptq_quant_linear,
    get_quantization_config,
)

from .gptq import L1RAQuantLinear
from .layer import L1RALayer, L1RALinear


class L1RAModel(LoraModel):

    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

        self.exclude_pruned = self.peft_config[adapter_name].exclude_pruned

        traininable_mode_counter = 0
        for config in self.peft_config.values():
            if not config.inference_mode:
                traininable_mode_counter += 1

        if traininable_mode_counter > 1:
            raise ValueError(
                "L1RAModel supports only 1 trainable adapter. "
                "When using multiple adapters, set inference_mode to True for all adapters except the one you want to train."
            )

        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)
        else:
            self.trainable_adapter_name = adapter_name

        self.rank_evolution = []
        self.num_training_steps = 0

    def _check_new_adapter_config(self, config: LoraConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        super()._check_new_adapter_config(config)

        traininable_mode_counter = 0
        for config_ in self.peft_config.values():
            if not config_.inference_mode:
                traininable_mode_counter += 1

        if traininable_mode_counter > 1:
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 trainable adapter. "
                "When using multiple adapters, set inference_mode to True for all adapters except the one "
                "you want to train."
            )

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }
        if (
            kwargs["loaded_in_8bit"] or kwargs["loaded_in_4bit"]
        ) and not is_bnb_available():
            raise ImportError(
                "To use L1RA with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

        quantization_config = get_quantization_config(self.model, method="gptq")
        if quantization_config is not None:
            kwargs["gptq_quantization_config"] = quantization_config

        # If it is not a L1RA, create a new module, else update it with new adapters
        if not isinstance(target, L1RALayer):
            new_module = self._create_new_module(
                lora_config, adapter_name, target, **kwargs
            )
            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(
                adapter_name,
                lora_config.r,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # avoid eager bnb import
        if is_bnb_available():
            import bitsandbytes as bnb

            from .bnb import L1RALinear8bitLt
        if is_bnb_4bit_available():
            from .bnb import L1RALinear4bit

        gptq_quantization_config = kwargs.get("gptq_quantization_config", None)
        AutoGPTQQuantLinear = get_auto_gptq_quant_linear(gptq_quantization_config)

        loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
            kwargs.update(
                {
                    "has_fp16_weights": target_base_layer.state.has_fp16_weights,
                    "memory_efficient_backward": target_base_layer.state.memory_efficient_backward,
                    "threshold": target_base_layer.state.threshold,
                    "index": target_base_layer.index,
                }
            )
            new_module = L1RALinear8bitLt(target, adapter_name, **kwargs)
        elif (
            loaded_in_4bit
            and is_bnb_4bit_available()
            and isinstance(target_base_layer, bnb.nn.Linear4bit)
        ):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target_base_layer.compute_dtype,
                    "compress_statistics": target_base_layer.weight.compress_statistics,
                    "quant_type": target_base_layer.weight.quant_type,
                }
            )
            new_module = L1RALinear4bit(target, adapter_name, **fourbit_kwargs)
        elif AutoGPTQQuantLinear is not None and isinstance(
            target, AutoGPTQQuantLinear
        ):
            new_module = L1RAQuantLinear(target, adapter_name, **kwargs)
        else:
            if isinstance(target_base_layer, torch.nn.Linear):
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target_base_layer, Conv1D):
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = L1RALinear(target, adapter_name, **kwargs)

        return new_module

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if (
                model_config["model_type"]
                not in TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING
            ):
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = (
                TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING[
                    model_config["model_type"]
                ]
            )
        return peft_config

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, evaluation=False, *args, **kwargs):
        outputs = self.model.forward(*args, **kwargs)

        if (
            (getattr(outputs, "loss", None) is not None)
            and isinstance(outputs.loss, torch.Tensor)
            and not evaluation
        ):
            # Calculate the orthogonal regularization
            sparse_reg_weight = self.peft_config[
                self.trainable_adapter_name
            ].l1ra_lambda

            if sparse_reg_weight < 0:
                raise ValueError(
                    "sparse_reg_weight should be greater or equal than 0. "
                )
            """
            # L1 regularization computation
            regu_loss = 0
            num_param = 0
            for n, p in self.model.named_parameters():
                if "lora_c" in n and self.trainable_adapter_name in n:
                    num_param += 1
                    regu_loss += torch.norm(p, p=1)
            if num_param > 0:
                regu_loss = regu_loss / num_param
            else:
                regu_loss = 0

            outputs.loss += sparse_reg_weight * regu_loss
            """
        return outputs


    def normalize_c(self):
        with torch.no_grad():
            A_matrix = None
            for n, p in self.model.named_parameters():
                if "lora_A" in n and self.trainable_adapter_name in n:
                    A_matrix = p

                if "lora_c" in n and self.trainable_adapter_name in n:

                    if A_matrix == None:
                        raise RuntimeError("Matrix A not found before vector c.")

                    scale_factor = p.max()
                    p.div_(scale_factor)
                    A_matrix.mul_(scale_factor)

                    A_matrix = None

    def update_ranks(self, global_step, num_training_steps):

        #self.normalize_c()

        if (
            global_step
            % int(self.peft_config[self.trainable_adapter_name].rank_update_ratio * num_training_steps)
            != 0 or
            global_step == 0
        ):
            return False

        t = self.peft_config[self.trainable_adapter_name].prune_threshold
        block_names = []
        for n, m in self.model.named_modules():
            if "lora" in n and not "lora_dropout" in n:
                name = ".".join(n.split(".")[:-1])
                if name not in block_names:
                    block_names.append(name)

        min_values = []
        pruned_idxs = []
        spare_ranks = 0
        pruned = 0
        for i, n in enumerate(block_names):
            if type(self.active_adapter) == list:
                active_adapter = self.active_adapter[0]
            else:
                active_adapter = self.active_adapter

            block = self.get_module_by_attribute(self.model, n)
            mask = block.lora_c[active_adapter] < t
            ranks_to_prune = mask.nonzero(as_tuple=True)[0]
            if block.lora_c[active_adapter].numel() > 0:
                min_values.append(block.lora_c[active_adapter].min())

            if ranks_to_prune.numel() > 0:
                pruned_idxs.append(i)

                if ranks_to_prune.numel() == block.lora_c[active_adapter].numel():
                    ranks_to_prune = ranks_to_prune[:-1]

                spare_ranks += ranks_to_prune.numel()

            for r in ranks_to_prune.flip(dims=(0,)):
                with torch.no_grad():
                    pruned += 1
                    block.lora_c[active_adapter] = torch.nn.parameter.Parameter(
                        data=self.remove_elem(block.lora_c[active_adapter], r),
                        requires_grad=True,
                    )
                    block.lora_A[active_adapter] = torch.nn.parameter.Parameter(
                        data=self.remove_col(block.lora_A[active_adapter], r),
                        requires_grad=True,
                    )
                    block.lora_B[active_adapter] = torch.nn.parameter.Parameter(
                        data=self.remove_row(block.lora_B[active_adapter], r),
                        requires_grad=True,
                    )

        if self.peft_config[self.trainable_adapter_name].reassign:
            s = 0
            for n, p in self.model.named_parameters():
                if "lora_c" in n:
                    s += p.numel()

            min_values = torch.stack(min_values)
            to_expand = min_values.argsort(descending=True)

            if self.exclude_pruned:
                for p in pruned_idxs:
                    to_expand = to_expand[to_expand != p]

            if len(to_expand) > 0:
                with torch.no_grad():
                    i = 0
                    for r in range(spare_ranks):
                        if i >= len(to_expand):
                            i = 0

                        idx = to_expand[i]
                        block = self.get_module_by_attribute(
                            self.model, block_names[idx]
                        )
                        block.lora_c[
                            active_adapter
                        ] = torch.nn.parameter.Parameter(
                            data=self.expand_c(block.lora_c[active_adapter]),
                            requires_grad=True,
                        )
                        block.lora_A[
                            active_adapter
                        ] = torch.nn.parameter.Parameter(
                            data=self.expand_A(block.lora_A[active_adapter]),
                            requires_grad=True,
                        )
                        block.lora_B[
                            active_adapter
                        ] = torch.nn.parameter.Parameter(
                            data=self.expand_B(block.lora_B[active_adapter]),
                            requires_grad=True,
                        )

                        i += 1

        self.rank_distribution = []
        for n, p in self.named_parameters():
            if "lora_c" in n:
                self.rank_distribution.append(p.numel())

        self.rank_evolution.append(self.rank_distribution)

        rank_pattern = {n: r for n, r in zip(block_names, self.rank_distribution)}
        self.peft_config[active_adapter].rank_pattern = rank_pattern

        torch.cuda.empty_cache()
        self.reassigned_ranks = spare_ranks

        print(f"Reassigned: {spare_ranks}")
        if spare_ranks == 0:
            return False

        return True

    def remove_elem(self, tensor, i):
        return torch.cat([tensor[0:i], tensor[i + 1 :]])

    def remove_row(self, tensor, i):
        return torch.cat([tensor[0:i, :], tensor[i + 1 :, :]], dim=0)

    def remove_col(self, tensor, i):
        return torch.cat([tensor[:, 0:i], tensor[:, i + 1 :]], dim=1)

    def expand_c(self, c):
        element = torch.ones((1,)).cuda()
        return torch.cat([c, element])

    def expand_A(self, A):
        if type(self.active_adapter) == list:
            active_adapter = self.active_adapter[0]
        else:
            active_adapter = self.active_adapter
        col_size = A.size()[0]
        vector = (
            torch.randn(col_size, 1).cuda()
            * 1
            / self.peft_config[active_adapter].r
        )
        return torch.cat([A, vector], dim=1)

    def expand_B(self, B):
        row_size = B.size()[1]
        vector = torch.zeros(1, row_size).cuda()
        return torch.cat([B, vector])

    def get_module_by_attribute(self, model, attr):
        attributes = attr.split(".")
        prev = model
        for a in attributes:
            if a.isdigit():
                prev = prev[int(a)]
            else:
                prev = getattr(prev, a)
        return prev