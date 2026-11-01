import torch
from torch import nn

from torch_einops_utils import (
    Frozen,
    frozen_grad,
    straight_through,
    zero_with_grad
)

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
