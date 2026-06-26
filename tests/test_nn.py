import torch
from torch import nn
from torch_einops_utils.nn import Sequential, Lambda, Identity, Residual, count_parameters

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

def test_count_parameters():
    model = nn.Linear(10, 10)
    assert count_parameters(model) == 110

    # Test requires_grad filter
    model.bias.requires_grad_(False)
    assert count_parameters(model) == 110
    assert count_parameters(model, requires_grad=True) == 100
    assert count_parameters(model, requires_grad=False) == 10

    # Test as a decorator
    @count_parameters
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
            self.linear.bias.requires_grad_(False)

    my_model = MyModel()
    assert my_model.num_parameters == 110

    # Test as a decorator with kwargs
    @count_parameters(requires_grad=True)
    class MyModelTrainable(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
            self.linear.bias.requires_grad_(False)

    my_model_trainable = MyModelTrainable()
    assert my_model_trainable.num_parameters == 100
