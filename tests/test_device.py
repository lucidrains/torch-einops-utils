from __future__ import annotations

import torch
from torch import nn


def test_module_device() -> None:
    from torch_einops_utils.device import module_device  # noqa: PLC0415

    assert module_device(nn.Linear(3, 3)) == torch.device("cpu")
    assert module_device(nn.Identity()) is None


def test_move_input_to_device() -> None:
    from torch_einops_utils.device import (  # noqa: PLC0415
        move_inputs_to_device
    )

    def fn(t):
        return t

    decorated = move_inputs_to_device(torch.device("cpu"))(fn)
    decorated(torch.randn(3))
