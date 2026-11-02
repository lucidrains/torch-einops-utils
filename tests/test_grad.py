import torch
from torch import nn

from torch_einops_utils import (
    Frozen,
    frozen_grad,
    mask_grad,
    per_sample_grad,
    straight_through,
    zero_with_grad,
)


def mask_transform(keep):
    def transform(per_sample_grads, ctx):
        return {
            name: (grads * keep.reshape(-1, *((1,) * (grads.ndim - 1)))).sum(dim = 0)
            for name, grads in per_sample_grads.items()
        }

    return transform

def test_grad_utils_end_to_end():
    torch.manual_seed(0)

    encoder = nn.Linear(4, 4)
    head = nn.Linear(4, 2)
    frozen_head = frozen_grad(head)

    assert isinstance(frozen_head, Frozen)

    x = torch.randn(3, 4, requires_grad = True)
    features = encoder(x)

    # same head used two ways in one graph
    trained = head(features.detach())     # normal path -> head gets grads
    frozen = frozen_head(features)        # frozen path -> encoder gets grads, head does not

    # straight through -> forward is rounded, backward is identity
    rounded = straight_through(features, features.round())

    # forward is zero, but gradients still flow back through `rounded`
    zero = zero_with_grad(rounded)

    assert torch.equal(zero, torch.zeros_like(zero))
    assert torch.allclose(rounded, features.round())

    (trained.sum() + frozen.sum() + zero.sum()).backward()

    # head grads come only from the trainable path
    ref_weight = head.weight.detach().clone().requires_grad_()
    ref_bias = head.bias.detach().clone().requires_grad_()
    ref_features = features.detach().clone().requires_grad_()
    (ref_features @ ref_weight.T + ref_bias).sum().backward()

    assert torch.allclose(head.weight.grad, ref_weight.grad)
    assert torch.allclose(head.bias.grad, ref_bias.grad)

    # encoder and input still train through the frozen path
    assert encoder.weight.grad is not None
    assert encoder.bias.grad is not None
    assert x.grad is not None

    # flags restored, so the head remains trainable elsewhere
    assert all(p.requires_grad for p in head.parameters())

    # the frozen module is an independent snapshot
    assert frozen_head.module is not head
    assert frozen_head.module.weight is not head.weight
    assert torch.equal(frozen_head.module.weight, head.weight)
    assert frozen_head.module.weight.grad is None
    assert frozen_head.module.bias.grad is None

    # training the live head leaves the snapshot untouched
    snapshot = frozen_head.module.weight.detach().clone()

    optimizer = torch.optim.SGD([*encoder.parameters(), *head.parameters()], lr = 0.1)
    optimizer.step()

    assert not torch.equal(head.weight, snapshot)
    assert torch.equal(frozen_head.module.weight, snapshot)

def test_per_sample_grad_default_matches_autograd():
    torch.manual_seed(0)

    layer = nn.Linear(4, 2)
    x = torch.randn(3, 4, requires_grad = True)

    wrapped = per_sample_grad(layer)
    wrapped(x).sum().backward()

    ref_layer = nn.Linear(4, 2)
    ref_layer.load_state_dict(layer.state_dict())

    ref_x = x.detach().clone().requires_grad_()
    ref_layer(ref_x).sum().backward()

    assert torch.allclose(layer.weight.grad, ref_layer.weight.grad)
    assert torch.allclose(layer.bias.grad, ref_layer.bias.grad)
    assert torch.allclose(x.grad, ref_x.grad)

def test_per_sample_grad_transform_context():
    torch.manual_seed(0)

    layer = nn.Linear(4, 2)
    x = torch.randn(3, 4, requires_grad = True)
    seen = {}

    def transform(per_sample_grads, ctx):
        seen['names'] = tuple(per_sample_grads)
        seen['shapes'] = {name: tuple(grads.shape) for name, grads in per_sample_grads.items()}
        seen['params'] = ctx.params
        seen['inputs'] = ctx.inputs
        seen['kwargs'] = ctx.kwargs
        seen['outputs'] = ctx.outputs
        seen['batch_size'] = ctx.batch_size
        seen['batch_dim'] = ctx.batch_dim
        seen['module'] = ctx.module

        return {name: grads.sum(dim = 0) for name, grads in per_sample_grads.items()}

    wrapped = per_sample_grad(layer, transform)
    out = wrapped(x)
    out.sum().backward()

    assert seen['names'] == ('weight', 'bias')
    assert seen['shapes'] == {'weight': (3, 2, 4), 'bias': (3, 2)}
    assert seen['params']['weight'] is layer.weight
    assert seen['params']['bias'] is layer.bias
    assert seen['inputs'][0] is x
    assert seen['kwargs'] == {}
    assert seen['outputs'] is out
    assert seen['batch_size'] == 3
    assert seen['batch_dim'] == 0
    assert seen['module'] is layer

def test_per_sample_grad_arbitrary_transform():
    torch.manual_seed(0)

    layer = nn.Linear(4, 2)
    x = torch.randn(5, 4)
    max_norm = 0.1

    def clip_aggregate(per_sample_grads, ctx):
        grads = {name: per_sample_grads[name].sum(dim = 0) for name in per_sample_grads}
        norm = torch.sqrt(sum(grad.square().sum() for grad in grads.values()))
        scale = (max_norm / norm.clamp(min = 1e-8)).clamp(max = 1.)
        return {name: grad * scale for name, grad in grads.items()}

    wrapped = per_sample_grad(layer, clip_aggregate)
    wrapped(x).sum().backward()

    ref_layer = nn.Linear(4, 2)
    ref_layer.load_state_dict(layer.state_dict())
    ref_layer(x).sum().backward()

    norm = torch.sqrt(sum(param.grad.square().sum() for param in ref_layer.parameters()))
    scale = (max_norm / norm.clamp(min = 1e-8)).clamp(max = 1.)

    assert torch.allclose(layer.weight.grad, ref_layer.weight.grad * scale)
    assert torch.allclose(layer.bias.grad, ref_layer.bias.grad * scale)

def test_per_sample_grad_partial_return_defaults_to_sum():
    torch.manual_seed(0)

    layer = nn.Linear(4, 2)
    x = torch.randn(3, 4)

    def scale_weight_only(per_sample_grads, ctx):
        return {'weight': per_sample_grads['weight'].sum(dim = 0) * 0.5}

    wrapped = per_sample_grad(layer, scale_weight_only)
    wrapped(x).sum().backward()

    ref_layer = nn.Linear(4, 2)
    ref_layer.load_state_dict(layer.state_dict())
    ref_layer(x).sum().backward()

    assert torch.allclose(layer.weight.grad, ref_layer.weight.grad * 0.5)
    assert torch.allclose(layer.bias.grad, ref_layer.bias.grad)

def test_per_sample_grad_masks_samples():
    torch.manual_seed(0)

    layer = nn.Linear(4, 2)
    x = torch.randn(4, 4, requires_grad = True)
    keep = torch.tensor([True, False, True, False])

    wrapped = per_sample_grad(layer, mask_transform(keep))
    wrapped(x).sum().backward()

    ref_layer = nn.Linear(4, 2)
    ref_layer.load_state_dict(layer.state_dict())

    ref_x = x.detach()[keep].clone().requires_grad_()
    ref_layer(ref_x).sum().backward()

    assert torch.allclose(layer.weight.grad, ref_layer.weight.grad)
    assert torch.allclose(layer.bias.grad, ref_layer.bias.grad)

    # input gradients are left alone
    full_layer = nn.Linear(4, 2)
    full_layer.load_state_dict(layer.state_dict())

    full_x = x.detach().clone().requires_grad_()
    full_layer(full_x).sum().backward()

    assert torch.allclose(x.grad, full_x.grad)

def test_per_sample_grad_multiple_outputs():
    torch.manual_seed(0)

    class MultiOut(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(2, 2))

        def forward(self, x):
            return x @ self.weight, x * 2

    module = MultiOut()
    x = torch.randn(4, 2, requires_grad = True)
    keep = torch.tensor([True, False, True, False])

    wrapped = per_sample_grad(module, mask_transform(keep))
    out1, out2 = wrapped(x)

    (out1.sum() + out2.sum()).backward()

    ref = MultiOut()
    ref.load_state_dict(module.state_dict())
    ref(x.detach()[keep])[0].sum().backward()

    assert torch.allclose(module.weight.grad, ref.weight.grad)

    full = MultiOut()
    full.load_state_dict(module.state_dict())
    full_x = x.detach().clone().requires_grad_()
    a, b = full(full_x)
    (a.sum() + b.sum()).backward()

    assert torch.allclose(x.grad, full_x.grad)

def test_per_sample_grad_last_dim():
    class Scale(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.))

        def forward(self, x):
            return x * self.weight

    module = Scale()
    x = torch.randn(2, 3, 4, requires_grad = True)
    keep = torch.tensor([True, True, False, True])

    wrapped = per_sample_grad(module, mask_transform(keep), dim = -1)
    wrapped(x).sum().backward()

    assert torch.allclose(module.weight.grad, x.detach()[..., keep].sum())
    assert torch.allclose(x.grad, torch.full_like(x, 2.))

def test_per_sample_grad_extra_inputs():
    class AddScale(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.))

        def forward(self, x, scale):
            return x * self.weight + scale

    module = AddScale()
    x = torch.randn(3, 2, requires_grad = True)
    scale = torch.tensor(2., requires_grad = True)
    keep = torch.tensor([True, False, True])

    wrapped = per_sample_grad(module, mask_transform(keep))
    wrapped(x, scale).sum().backward()

    assert torch.allclose(module.weight.grad, x.detach()[keep].sum())
    assert torch.allclose(x.grad, torch.ones_like(x))
    assert scale.grad is None

def test_per_sample_grad_frozen_params():
    torch.manual_seed(0)

    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    model[0].requires_grad_(False)

    seen = {}

    def transform(per_sample_grads, ctx):
        seen['names'] = tuple(per_sample_grads)
        return {name: grads.sum(dim = 0) for name, grads in per_sample_grads.items()}

    wrapped = per_sample_grad(model, transform)
    wrapped(torch.randn(3, 4)).sum().backward()

    assert seen['names'] == ('1.weight', '1.bias')
    assert model[0].weight.grad is None
    assert model[1].weight.grad is not None

def test_per_sample_grad_transform_none_uses_sum():
    torch.manual_seed(0)

    layer = nn.Linear(4, 2)
    x = torch.randn(3, 4)

    wrapped = per_sample_grad(layer, lambda per_sample_grads, ctx: None)
    wrapped(x).sum().backward()

    ref_layer = nn.Linear(4, 2)
    ref_layer.load_state_dict(layer.state_dict())
    ref_layer(x).sum().backward()

    assert torch.allclose(layer.weight.grad, ref_layer.weight.grad)
    assert torch.allclose(layer.bias.grad, ref_layer.bias.grad)

def test_mask_grad():
    torch.manual_seed(0)

    t = torch.randn(3, 4, requires_grad = True)
    mask = torch.tensor([True, False, True])[:, None]

    out = mask_grad(t, mask)

    assert torch.allclose(out, t)

    out.sum().backward()

    assert torch.allclose(t.grad, mask.to(t.dtype).expand_as(t))

    t = torch.randn(2, 3, requires_grad = True)
    mask = torch.tensor([1., 0.5, 0.])

    out = mask_grad(t, mask)

    assert torch.allclose(out, t)

    out.sum().backward()

    assert torch.allclose(t.grad, mask.expand_as(t))

    t = torch.randn(2, 3, requires_grad = True)
    mask_grad(t, False).sum().backward()

    assert torch.equal(t.grad, torch.zeros_like(t))

def test_mask_grad_controls_parameter_updates():
    param = nn.Parameter(torch.zeros(4))
    mask = torch.tensor([True, False, True, False])

    mask_grad(param, mask).sum().backward()

    optimizer = torch.optim.SGD([param], lr = 0.1)
    optimizer.step()

    expected = -0.1 * mask.to(param.dtype)
    assert torch.allclose(param, expected)
