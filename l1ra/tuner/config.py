from dataclasses import dataclass, field
from typing import Optional

from peft.tuners.lora import LoraConfig
from peft.utils import PeftType


@dataclass
class L1RAConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.AdaLora`].

    Args:
        target_r (`int`): The target average rank of incremental matrix.
        init_r (`int`): The initial rank for each incremental matrix.
        tinit (`int`): The steps of initial fine-tuning warmup.
        tfinal (`int`): The step of final fine-tuning.
        deltaT (`int`): The time internval between two budget allocations.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        orth_reg_weight (`float`): The coefficient of orthogonal regularization.
        total_step (`int`): The total training steps that should be specified before training.
        rank_pattern (`list`): The allocated rank for each weight matrix by RankAllocator.
    """

    sparse_reg_weight: float = field(default=1e-3, metadata={"help": "The sparse l1 regularization coefficient."})
    prune_threshold: float = field(default=1e-10, metadata={"help": "Threshold under which ranks are pruned."})
    rank_update_steps: int = field(default=50, metadata={"help": "How many training steps between each rank update"})
    reassign: bool = field(default=True, metadata={"help": "Whether to reassign pruned ranks"})

    def __post_init__(self):
        self.peft_type = PeftType.LORA
