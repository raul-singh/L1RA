from dataclasses import dataclass, field
from aenum import extend_enum

from peft.tuners.lora import LoraConfig
from peft.utils import PeftType
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING

from l1ra.tuner.model import L1RAModel

PEFT_TYPE_TO_MODEL_MAPPING.update(
    {"L1RA": L1RAModel}
)
extend_enum(PeftType, "L1RA", "L1RA")


@dataclass
class L1RAConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`L1RAModel`].

    Args:
        r (`int`):
            Lora attention dimension (the "rank").
        lora_alpha (`int`):
            The alpha parameter for Lora scaling.
        l1ra_lambda (`float`):
            The sparse l1 regularization coefficient.
        eta_c (`float`):
            The decoupled learning rate for the gate vectors
        rank_update_ratio (`float`):
            Ratio of training steps between each rank update.
        prune_threshold (`float`):
            Threshold under which ranks are pruned.
        reassign (`bool`):
            Whether to reassign pruned ranks.
        exclude_pruned (`bool`):
            Whether to exclude pruned adapters from rank reassignment.

    """

    l1ra_lambda: float = field(default=1e-3, metadata={"help": "The sparse l1 regularization coefficient."})
    rank_update_ratio: int = field(default=0.1, metadata={"help": "Ratio of training steps between each rank update."})
    prune_threshold: float = field(default=1e-10, metadata={"help": "Threshold under which ranks are pruned."})
    reassign: bool = field(default=True, metadata={"help": "Whether to reassign pruned ranks."})
    exclude_pruned: bool = field(default=True, metadata={"help": "Whether to exclude pruned adapters from rank reassignment."})

    def __post_init__(self):
        self.peft_type = PeftType.L1RA
