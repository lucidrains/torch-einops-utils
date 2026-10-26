import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical, kl_divergence

from torch_einops_utils import (
    bernoulli_kl,
    binary_cross_entropy_with_logits,
    categorical_kl,
    cross_entropy,
    log_softmax,
    soft_target_binary_cross_entropy,
    soft_target_cross_entropy,
    soft_target_cross_entropy_from_logits,
)

param = pytest.mark.parametrize


def test_hard_cross_entropy_survives_precision_cliff():
    z = torch.tensor([[9., -9., -9.]], requires_grad = True)
    loss = cross_entropy(z, torch.tensor([0]))
    grad, = torch.autograd.grad(loss, z)

    expected = torch.tensor([-3.04599723e-08, 1.52299790e-08, 1.52299790e-08])

    assert torch.allclose(loss, expected[0].neg(), rtol = 1e-5)
    assert torch.allclose(grad, expected.unsqueeze(0), rtol = 1e-5)
    assert grad.sum().abs() < 1e-12
    assert torch.allclose(log_softmax(z.detach())[0, 0], expected[0], rtol = 1e-5)

def test_cross_entropy_requires_batched_logits():
    with pytest.raises(AssertionError):
        cross_entropy(torch.tensor([9., -9., -9.]), torch.tensor(0))

@param('reduction', ['none', 'sum', 'mean'])
def test_cross_entropy_matches_torch(reduction):
    logits, target = torch.randn(4, 8, 5, requires_grad = True), torch.randint(0, 5, (4, 8))
    target[:, 0] = -100

    loss = cross_entropy(logits, target, class_dim = -1, reduction = reduction)
    ref = F.cross_entropy(logits.movedim(-1, 1), target, reduction = reduction)

    assert torch.allclose(loss, ref)
    assert torch.allclose(
        torch.autograd.grad(loss.sum(), logits)[0],
        torch.autograd.grad(ref.sum(), logits)[0]
    )

@param('reduction', ['none', 'sum', 'mean'])
def test_cross_entropy_all_ignored(reduction):
    logits = torch.randn(3, 4, 5, requires_grad = True)
    target = torch.full((3, 4), -100)

    loss = cross_entropy(logits, target, class_dim = -1, reduction = reduction)
    ref = F.cross_entropy(logits.movedim(-1, 1), target, reduction = reduction)

    assert torch.isnan(loss) if reduction == 'mean' else torch.equal(loss, ref)
    assert torch.equal(torch.autograd.grad(loss.sum(), logits)[0], torch.zeros_like(logits))

@param('dtype, logit, expected_loss, rtol', [
    (torch.float32, 20., 2.0611536922e-09, 1e-5),
    (torch.bfloat16, 9., 1.2339e-04, 1e-2),
])
def test_binary_cross_entropy_survives_precision_cliff(dtype, logit, expected_loss, rtol):
    logits = torch.tensor([logit, -logit], dtype = dtype, requires_grad = True)
    target = torch.tensor([1., 0.])

    loss = binary_cross_entropy_with_logits(logits, target, reduction = 'none')
    grad, = torch.autograd.grad(loss.sum(), logits)

    expected_loss = torch.tensor([expected_loss, expected_loss])
    expected_grad = expected_loss * torch.tensor([-1., 1.])

    assert torch.allclose(loss.float(), expected_loss, rtol = rtol)
    assert torch.allclose(grad.float(), expected_grad, rtol = rtol)

    standard_grad, = torch.autograd.grad(
        F.binary_cross_entropy_with_logits(logits, target, reduction = 'none').sum(),
        logits
    )

    assert not torch.allclose(standard_grad.float(), expected_grad, rtol = rtol, atol = 1e-12)

@param('reduction', ['none', 'sum', 'mean'])
def test_binary_cross_entropy_matches_torch(reduction):
    torch.manual_seed(0)

    logits = torch.randn(4, 8, requires_grad = True)
    target = torch.rand(4, 8)

    loss = binary_cross_entropy_with_logits(logits, target, reduction = reduction)
    ref = F.binary_cross_entropy_with_logits(logits, target, reduction = reduction)

    assert torch.allclose(loss, ref)
    assert torch.allclose(
        torch.autograd.grad(loss.sum(), logits)[0],
        torch.autograd.grad(ref.sum(), logits)[0],
        atol = 1e-7
    )

@param('z, target', [
    ([20., 20.], [1., 0.]),
    ([-20., -20.], [1., 0.]),
    ([20., -20.], [1. - 1e-7, 1e-7]),
])
def test_binary_cross_entropy_survives_saturation(z, target):
    logits, target = map(torch.tensor, (z, target))
    logits.requires_grad_()

    loss = binary_cross_entropy_with_logits(logits, target, reduction = 'none')
    grad, = torch.autograd.grad(loss.sum(), logits)

    logits64 = logits.detach().double().requires_grad_()
    loss64 = binary_cross_entropy_with_logits(logits64, target.double(), reduction = 'none')
    grad64, = torch.autograd.grad(loss64.sum(), logits64)

    assert torch.allclose(loss.double(), loss64, rtol = 1e-5)
    assert torch.allclose(grad.double(), grad64, rtol = 1e-5)

def test_binary_cross_entropy_handles_inf_logits():
    logits = torch.tensor([float('inf'), float('-inf')])
    target = torch.tensor([1., 0.])

    loss = binary_cross_entropy_with_logits(logits, target, reduction = 'none')

    assert torch.equal(loss, torch.zeros(2))

def test_soft_target_binary_cross_entropy_vanishes_at_equality():
    z_t = torch.tensor([2., 1., 0.])
    z_s = z_t.clone().requires_grad_()

    loss = soft_target_binary_cross_entropy(z_s, z_t)
    grad, = torch.autograd.grad(loss.sum(), z_s)

    p = z_t.sigmoid()

    assert torch.allclose(loss, binary_cross_entropy_with_logits(z_s, p, reduction = 'none'))
    assert torch.allclose(loss, -(p * p.log() + (1. - p) * (1. - p).log()))
    assert torch.equal(grad, torch.zeros_like(grad))

def test_soft_target_binary_cross_entropy_matches_fp64_near_equality():
    z_t = torch.tensor([2., 1., 0.])
    z_s = (z_t + torch.tensor([-1e-7, 1e-7, 0.])).requires_grad_()

    loss = soft_target_binary_cross_entropy(z_s, z_t)
    grad, = torch.autograd.grad(loss.sum(), z_s)

    z_s64, z_t64 = z_s.detach().double().requires_grad_(), z_t.double()
    loss64 = soft_target_binary_cross_entropy(z_s64, z_t64)
    grad64, = torch.autograd.grad(loss64.sum(), z_s64)

    assert torch.allclose(loss.double(), loss64, atol = 1e-12)
    assert torch.allclose(grad.double(), grad64, rtol = 1e-5, atol = 1e-12)

def test_soft_target_cross_entropy_from_logits_vanishes_at_equality():
    z_t = torch.tensor([2., 1., 0.])
    z_s = z_t.clone().requires_grad_()

    loss = soft_target_cross_entropy_from_logits(z_s, z_t)
    grad, = torch.autograd.grad(loss, z_s)

    p = z_t.softmax(-1)

    assert torch.allclose(loss, -(p * p.log()).sum())
    assert torch.equal(grad, torch.zeros_like(grad))

def test_soft_target_cross_entropy_from_logits_matches_fp64_near_equality():
    z_t = torch.tensor([2., 1., 0.])
    z_s = (z_t + torch.tensor([-1e-7, 1e-7, 0.])).requires_grad_()

    loss = soft_target_cross_entropy_from_logits(z_s, z_t)
    grad, = torch.autograd.grad(loss, z_s)

    z_s64 = z_s.detach().double().requires_grad_()
    p64 = z_t.double().softmax(-1)
    grad64, = torch.autograd.grad(-(p64 * z_s64.log_softmax(-1)).sum(), z_s64)

    expected = torch.tensor([-4.5955e-8, 4.1442e-8, 4.5131e-9])

    assert torch.allclose(grad.double(), grad64, rtol = 1e-5, atol = 1e-12)
    assert torch.allclose(grad, expected, atol = 1e-12)

@param('reduction', ['none', 'sum', 'mean'])
def test_soft_target_cross_entropy_matches_logits(reduction):
    torch.manual_seed(0)

    target_logits = torch.randn(4, 8, 5)
    target_probs = target_logits.softmax(dim = -1)
    logits = torch.randn(4, 8, 5, requires_grad = True)

    loss = soft_target_cross_entropy(logits, target_probs, reduction = reduction)
    ref = soft_target_cross_entropy_from_logits(logits, target_logits, reduction = reduction)

    assert torch.allclose(loss, ref)

@param('reduction', ['none', 'sum', 'mean'])
def test_soft_target_cross_entropy_broadcasts_heads_and_masks(reduction):
    torch.manual_seed(0)

    logits = torch.randn(2, 4, 3, 5, requires_grad = True)
    target_probs = torch.randn(2, 3, 5).softmax(dim = -1)
    mask = torch.tensor([[True, True, True], [True, False, False]])

    loss = soft_target_cross_entropy(logits, target_probs, mask = mask, reduction = reduction)

    ref = soft_target_cross_entropy_from_logits(
        logits,
        target_probs.log()[:, None].expand_as(logits),
        mask = mask,
        reduction = reduction
    )

    assert torch.allclose(loss, ref)
    assert torch.isfinite(loss).all()

    if reduction != 'none':
        loss.backward()
        assert torch.isfinite(logits.grad).all()

def test_soft_target_cross_entropy_handles_empty_targets():
    logits = torch.randn(1, 2, 5, requires_grad = True)
    target_probs = torch.zeros(1, 2, 5)
    target_probs[0, 0, 0] = 1.

    loss = soft_target_cross_entropy(logits, target_probs, reduction = 'mean')

    assert torch.isfinite(loss)

    loss.backward()
    assert torch.isfinite(logits.grad).all()

def test_categorical_kl_matches_fp64_near_equality():
    z_p = torch.tensor([2., 1., 0.], requires_grad = True)
    z_q = (z_p.detach() + torch.tensor([-1e-7, 1e-7, 0.])).requires_grad_()

    kl = categorical_kl(z_p, z_q)
    grads = torch.autograd.grad(kl, (z_p, z_q))

    z_p64, z_q64 = z_p.detach().double().requires_grad_(), z_q.detach().double().requires_grad_()
    p64, q64 = z_p64.softmax(-1), z_q64.softmax(-1)
    kl64 = (p64 * (p64.log() - q64.log())).sum()
    grads64 = torch.autograd.grad(kl64, (z_p64, z_q64))

    assert torch.allclose(kl.double(), kl64, atol = 1e-12)
    assert all(torch.allclose(a.double(), b, rtol = 1e-5, atol = 1e-12) for a, b in zip(grads, grads64))

def test_categorical_kl_skips_frozen_input_gradients():
    z_p = torch.randn(4, 5, requires_grad = True)
    z_q = torch.randn(4, 5)

    assert torch.autograd.grad(categorical_kl(z_p, z_q).sum(), z_p)[0].shape == z_p.shape

    z_q.requires_grad_()
    assert torch.autograd.grad(categorical_kl(z_p.detach(), z_q).sum(), z_q)[0].shape == z_q.shape

def test_categorical_kl_matches_torch_and_is_stable_near_equality():
    torch.manual_seed(0)

    z_p = torch.randn(8, 5)
    z_q = torch.randn(8, 5)

    p, q = Categorical(logits = z_p), Categorical(logits = z_q)
    assert torch.allclose(categorical_kl(z_p, z_q), kl_divergence(p, q), atol = 1e-6)

    z_s = z_p.clone().requires_grad_()
    kl = categorical_kl(z_p, z_s)
    grad, = torch.autograd.grad(kl.sum(), z_s)

    assert torch.equal(kl, torch.zeros_like(kl))
    assert torch.equal(grad, torch.zeros_like(grad))

def test_bernoulli_kl_is_stable_near_equality():
    z_p = torch.tensor([0.2, -0.3])
    z_s = (z_p + torch.tensor([1e-7, -1e-7])).requires_grad_()

    p, q = Bernoulli(logits = z_p), Bernoulli(logits = z_s)
    standard_grad, = torch.autograd.grad(kl_divergence(p, q).sum(), z_s)
    fixed_grad, = torch.autograd.grad(bernoulli_kl(z_p, z_s).sum(), z_s)

    z_s64 = z_s.detach().double().requires_grad_()
    ref, = torch.autograd.grad(kl_divergence(Bernoulli(logits = z_p.double()), Bernoulli(logits = z_s64)).sum(), z_s64)

    assert torch.allclose(fixed_grad.double(), ref, rtol = 1e-5, atol = 1e-12)
    assert not torch.allclose(standard_grad.double(), ref, rtol = 1e-2, atol = 1e-12)

    z_p = z_p.bfloat16()
    z_s = (z_p + torch.tensor([1e-7, -1e-7], dtype = torch.bfloat16)).requires_grad_()
    standard_grad, = torch.autograd.grad(kl_divergence(Bernoulli(logits = z_p), Bernoulli(logits = z_s)).sum(), z_s)
    fixed_grad, = torch.autograd.grad(bernoulli_kl(z_p, z_s).sum(), z_s)

    assert torch.equal(z_s, z_p)
    assert standard_grad.abs().sum() > 0
    assert torch.equal(fixed_grad, torch.zeros_like(fixed_grad))

def test_reverse_kl_gradient_matches_fp64():
    torch.manual_seed(0)

    z_t = torch.randn(8, 5)
    z_s = torch.randn(8, 5, requires_grad = True)

    kl = categorical_kl(z_s, z_t)
    grad, = torch.autograd.grad(kl.sum(), z_s)

    z_s64, z_t64 = z_s.detach().double().requires_grad_(), z_t.double()
    q64, p64 = z_s64.softmax(-1), z_t64.softmax(-1)
    grad64, = torch.autograd.grad((q64 * (q64.log() - p64.log())).sum(), z_s64)

    assert torch.allclose(grad.double(), grad64, rtol = 1e-5, atol = 1e-6)

def test_categorical_kl_handles_mixed_large_and_tiny_logit_differences():
    z_p = torch.tensor([[0., 0., -37.7]])
    d = torch.tensor([[0., 0., 20.]])
    d = d - (z_p.softmax(-1) * d).sum(dim = -1, keepdim = True)
    z_q = z_p + d

    z_p64, z_q64 = z_p.double(), z_q.double()
    p64, q64 = z_p64.softmax(-1), z_q64.softmax(-1)

    assert torch.allclose(categorical_kl(z_p, z_q), (p64 * (p64.log() - q64.log())).sum(-1).float(), rtol = 1e-5)

@param('z_p, z_q', [
    ([9., -9., -9.], [9., -9., -9.]),
    ([100., 0.], [-100., 0.]),
    ([1000., 0.], [-1000., 0.]),
    ([-100., 0.], [100., 0.]),
    ([-1000., 0.], [1000., 0.]),
    ([[2., 1., 0.], [-1000., 0., 0.]], [[2. + 1e-7, 1. - 1e-7, 0.], [1000., 0., 0.]]),
])
def test_kl_and_soft_target_cross_entropy_survive_saturation(z_p, z_q):
    z_p, z_q = map(torch.tensor, (z_p, z_q))
    z_q.requires_grad_()

    kl = categorical_kl(z_p, z_q)
    ce = soft_target_cross_entropy_from_logits(z_q, z_p)

    kl_grad, = torch.autograd.grad(kl.sum(), z_q)
    ce_grad, = torch.autograd.grad(ce.sum(), z_q)

    log_p, log_q = z_p.double().log_softmax(-1), z_q.detach().double().log_softmax(-1)
    p, q = log_p.exp(), log_q.exp()

    assert torch.allclose(kl.double(), (p * (log_p - log_q)).sum(-1))
    assert torch.allclose(ce.double(), -(p * log_q).sum(-1))
    assert torch.allclose(kl_grad, (q - p).float())
    assert torch.allclose(ce_grad, (q - p).float())
