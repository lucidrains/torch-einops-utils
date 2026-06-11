import torch
from torch import nn
from torch_einops_utils.nn import Sequential, Lambda, Identity, Residual

def test_sequential():
    # Test that it filters out None
    seq = Sequential(
        nn.Linear(10, 10),
        None,
        nn.ReLU()
    )
    assert len(seq) == 2

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

def test_residual():
    fn = lambda x: x * 2
    res = Residual(fn)
    x = torch.tensor([1., 2., 3.])
    assert torch.allclose(res(x), torch.tensor([3., 6., 9.]))

    def fn_tuple(x):
        return x * 2, x * 3, dict(a = x * 4)

    res_tuple = Residual(fn_tuple)
    out1, out2, out3 = res_tuple(x)
    assert torch.allclose(out1, torch.tensor([3., 6., 9.]))
    assert torch.allclose(out2, torch.tensor([3., 6., 9.]))
    assert torch.allclose(out3['a'], torch.tensor([4., 8., 12.]))
