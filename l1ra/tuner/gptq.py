import torch

from .layer import L1RALayer


class L1RAQuantLinear(torch.nn.Module, L1RALayer):
    def __init__(
        self,
        base_layer,
        adapter_name,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        L1RALayer.__init__(self, base_layer)

        # self.base_layer and self.quant_linear_module are the same; we need the former for consistency and the latter
        # for backwards compatibility
        self.quant_linear_module = base_layer
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.quant_linear_module(x)

        if self.disable_adapters:
            return result

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            lora_c = self.lora_c[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            r = self.r[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                if x.dtype != torch.float32:
                    x = x.float()

            output = (dropout(x) @ (lora_A * lora_c).T @ lora_B.T) * scaling / r

            if requires_conversion:
                output = output.to(expected_dtype)
            result += output
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "l1ra." + rep
