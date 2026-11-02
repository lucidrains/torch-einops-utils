import pytest
import torch
from torch import nn

from torch_einops_utils import (
    dict_add,
    grad_scope,
    named_grads,
    named_grads_from_loss,
    orthogonal_component,
    set_named_grads,
)


class TwoHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(3, 4)
        self.head_a = nn.Linear(4, 2)
        self.head_b = nn.Linear(4, 2)

    def forward(self, x):
        hidden = self.trunk(x)
        return self.head_a(hidden), self.head_b(hidden)


# reconcile two task grads

def reconcile(g1, g2):
    shared = [name for name in g1 if name in g2]

    if not shared:
        return dict_add(g1, g2)

    a = torch.cat([g1[name].flatten() for name in shared])
    b = torch.cat([g2[name].flatten() for name in shared])
    a0, b0 = a, b

    if a0 @ b0 < 0:
        a = orthogonal_component(a0, b0)
        b = orthogonal_component(b0, a0)

    merged = dict(g1)
    merged.update(g2)

    offset = 0

    for name in shared:
        size = g1[name].numel()
        merged[name] = ((a[offset:offset + size] + b[offset:offset + size]) / 2).view_as(g1[name])
        offset += size

    return merged


def test_named_grads_roundtrip():
    torch.manual_seed(0)

    model = nn.Linear(3, 2)
    x = torch.randn(4, 3)

    model(x).sum().backward()

    grads = named_grads(model)

    assert set(grads) == {'weight', 'bias'}

    for name, param in model.named_parameters():
        assert torch.equal(grads[name], param.grad)

    model.zero_grad()

    assert named_grads(model) == {}
    assert all(grad is None for grad in named_grads(model, remove_none = False).values())

    set_named_grads(model, grads)

    for name, param in model.named_parameters():
        assert torch.equal(param.grad, grads[name])

def test_named_grads_from_loss_skips_unused_and_frozen():
    torch.manual_seed(0)

    model = TwoHead()
    x = torch.randn(5, 3)

    out_a, _ = model(x)
    grads = named_grads_from_loss(out_a.sum(), model)

    assert set(grads) == {'trunk.weight', 'trunk.bias', 'head_a.weight', 'head_a.bias'}
    assert 'head_b.weight' not in grads

    model.trunk.requires_grad_(False)

    out_a, _ = model(x)
    grads = named_grads_from_loss(out_a.sum(), model)

    assert 'trunk.weight' not in grads
    assert 'head_a.weight' in grads

def test_named_grads_from_loss_create_graph():
    torch.manual_seed(0)

    model = nn.Linear(2, 1)
    x = torch.randn(3, 2)

    grads = named_grads_from_loss(model(x).square().sum(), model, create_graph = True)

    assert all(grad.requires_grad for grad in grads.values())

def test_set_named_grads():
    model = nn.Linear(2, 1)
    grads = {'weight': torch.ones_like(model.weight)}

    set_named_grads(model, grads)

    assert torch.equal(model.weight.grad, grads['weight'])
    assert model.bias.grad is None

def test_reconcile_two_grads():
    g1 = {'shared': torch.tensor([1., 0.]), 'a': torch.tensor([1.])}
    g2 = {'shared': torch.tensor([-1., 1.]), 'b': torch.tensor([2.])}

    merged = reconcile(g1, g2)

    assert torch.allclose(merged['shared'], torch.tensor([0.25, 0.75]))
    assert torch.allclose(merged['a'], torch.tensor([1.]))
    assert torch.allclose(merged['b'], torch.tensor([2.]))

    # inputs are left alone
    assert torch.equal(g1['shared'], torch.tensor([1., 0.]))

def test_reconcile_e2e():
    torch.manual_seed(0)

    model = TwoHead()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
    x = torch.randn(5, 3)

    out_a, out_b = model(x)

    grads_a = named_grads_from_loss(out_a.sum(), model, retain_graph = True)
    grads_b = named_grads_from_loss(out_b.sum(), model)

    merged = reconcile(grads_a, grads_b)
    initial = {name: param.detach().clone() for name, param in model.named_parameters()}

    optimizer.zero_grad()
    set_named_grads(model, merged)
    optimizer.step()

    for name, param in model.named_parameters():
        expected = initial[name] - 0.1 * merged[name]
        assert torch.allclose(param, expected, atol = 1e-6)

    for name in ('head_a.weight', 'head_a.bias'):
        assert torch.allclose(merged[name], grads_a[name])

    for name in ('head_b.weight', 'head_b.bias'):
        assert torch.allclose(merged[name], grads_b[name])

def test_grad_scope_disable():
    torch.manual_seed(0)

    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
    x = torch.randn(3, 2)

    with grad_scope(model, disable = ['0.weight']):
        model(x).sum().backward()

        assert model[0].weight.grad is None
        assert model[0].bias.grad is not None
        assert model[2].weight.grad is not None

    assert model[0].weight.requires_grad

def test_grad_scope_enable():
    torch.manual_seed(0)

    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
    x = torch.randn(3, 2)

    model.requires_grad_(False)

    with grad_scope(model, enable = ['0.weight']):
        model(x).sum().backward()

        assert model[0].weight.grad is not None
        assert model[0].bias.grad is None

    assert not model[0].weight.requires_grad

def test_grad_scope_hard_set_and_reset():
    model = nn.Linear(2, 1)

    with grad_scope(model, hard_set = {'weight': False}):
        assert not model.weight.requires_grad
        assert model.bias.requires_grad

    assert model.weight.requires_grad
    assert model.bias.requires_grad

def test_grad_scope_strict_false():
    model = nn.Linear(2, 1)

    with grad_scope(model, disable = ['nope'], strict = False):
        pass

def test_grad_scope_nested():
    model = nn.Linear(2, 1)

    with grad_scope(model, disable = ['weight']):
        assert not model.weight.requires_grad

        with grad_scope(model, enable = ['weight']):
            assert model.weight.requires_grad

        assert not model.weight.requires_grad

    assert model.weight.requires_grad

def test_grad_scope_unknown_name():
    model = nn.Linear(2, 1)

    with pytest.raises(KeyError), grad_scope(model, disable = ['nope']):
        pass
