"""
Generic LoRA (Low-Rank Adaptation, Hu et al. 2021) wrapper -- module-agnostic,
works on any nn.Module built from nn.Linear layers. Used here to fine-tune
the value head under self-play with a capacity-constrained update instead of
a full-parameter one, which empirically fine-tunes more reliably on a small,
self-generated data distribution without regressing.
"""
import torch
from torch import nn


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear, freezing its original weight/bias and
    adding a trainable low-rank delta:
        output = frozen_linear(x) + scaling * (x @ A^T) @ B^T
    `B` is initialized to ZERO, so the wrapped layer's output is byte-
    identical to the original before any training happens."""

    def __init__(self, linear, rank=4, alpha=1.0):
        super().__init__()
        assert isinstance(linear, nn.Linear), f"LoRALinear only wraps nn.Linear, got {type(linear)}"
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        device = linear.weight.device
        dtype = linear.weight.dtype
        self.lora_A = nn.Parameter(torch.randn(rank, linear.in_features, device=device, dtype=dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank, device=device, dtype=dtype))

    def forward(self, x):
        base = self.linear(x)
        delta = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base + self.scaling * delta

    def lora_parameters(self):
        return [self.lora_A, self.lora_B]

    def merge_into_base(self):
        """Fold the current LoRA delta into the frozen base weight and
        reset B to zero -- used to "commit" a trained adapter back into a
        plain nn.Linear-equivalent state so the saved checkpoint is an
        ordinary state_dict, loadable with zero LoRA-awareness downstream."""
        with torch.no_grad():
            delta_w = self.scaling * (self.lora_B @ self.lora_A)
            self.linear.weight.add_(delta_w)
            self.lora_B.zero_()


def apply_lora(module, rank=4, alpha=1.0):
    """Recursively replace every nn.Linear submodule of `module` with a
    LoRALinear wrapper. Returns the new LoRA parameters for the caller's
    optimizer. Modifies `module` in place."""
    lora_params = []
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            wrapped = LoRALinear(child, rank=rank, alpha=alpha)
            setattr(module, name, wrapped)
            lora_params.extend(wrapped.lora_parameters())
        else:
            lora_params.extend(apply_lora(child, rank=rank, alpha=alpha))
    return lora_params
