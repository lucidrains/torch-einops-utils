from __future__ import annotations

from copy import deepcopy

from torch.nn import Module

# straight through estimator

def straight_through(src, tgt):
    # forward -> tgt, gradient -> src
    return src + (tgt - src).detach()

def zero_with_grad(tgt):
    # forward -> 0, gradient -> tgt
    return tgt - tgt.detach()

# frozen subnetwork
# deepcopy once so the live module can keep training independently
# params are frozen, but gradients still flow to inputs

class Frozen(Module):
    def __init__(self, module):
        super().__init__()
        self.module = deepcopy(module)
        self.module.requires_grad_(False)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

# functional form
# frozen_head = frozen_grad(head)
# out = frozen_head(x)

def frozen_grad(module):
    return Frozen(module)
