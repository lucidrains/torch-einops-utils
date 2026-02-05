import torch
from torch import nn

def test_module_device():
    from torch_einops_utils.device import module_device

    assert module_device(nn.Linear(3, 3)) == torch.device('cpu')
    assert module_device(nn.Identity()) is None
