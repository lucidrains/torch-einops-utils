import torch
from torch import nn
from torch_einops_utils.nn import Sequential, Lambda, Identity

def test_sequential():
    # Test that it filters out None
    seq = Sequential(
        nn.Linear(10, 10),
        None,
        nn.ReLU()
    )
    assert all(isinstance(module, nn.Module) for module in seq)

    # Test forward pass
    x = torch.randn(2, 10)
    out = seq(x)
    assert out.shape == (2, 10)

def test_lambda():
    fn = lambda x: x * 2
    lam = Lambda(fn)
    x = torch.tensor([1., 2., 3.])
    assert torch.allclose(lam(x), torch.tensor([2., 4., 6.]))

def test_identity():
    ident = Identity()
    x = torch.tensor([1., 2., 3.])
    assert torch.allclose(ident(x), x)
