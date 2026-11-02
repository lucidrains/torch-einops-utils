import pytest
import torch
from torch.nn.functional import normalize

from torch_einops_utils import (
    components,
    orthogonal_component,
    parallel_component,
    slerp,
)

param = pytest.mark.parametrize

def test_vector_components():
    v = torch.tensor([3., 4.])
    u = torch.tensor([1., 0.])

    parallel = parallel_component(v, u)
    orthogonal = orthogonal_component(v, u)

    assert torch.allclose(parallel, torch.tensor([3., 0.]))
    assert torch.allclose(orthogonal, torch.tensor([0., 4.]))

    parallel, orthogonal = components(v, u)

    assert torch.allclose(parallel + orthogonal, v)
    assert torch.allclose((parallel * orthogonal).sum(), torch.tensor(0.))

def test_vector_components_parallel():
    v = torch.tensor([2., 4.])
    u = torch.tensor([1., 2.])

    parallel, orthogonal = components(v, u)

    assert torch.allclose(parallel, v)
    assert torch.allclose(orthogonal, torch.zeros_like(v))

def test_vector_components_orthogonal():
    v = torch.tensor([0., 1.])
    u = torch.tensor([1., 0.])

    parallel, orthogonal = components(v, u)

    assert torch.allclose(parallel, torch.zeros_like(v))
    assert torch.allclose(orthogonal, v)

def test_vector_components_batch():
    torch.manual_seed(0)

    v = torch.randn(4, 5, 3)
    u = torch.randn(4, 5, 3)

    parallel, orthogonal = components(v, u)

    assert torch.allclose(parallel + orthogonal, v)
    assert torch.allclose((parallel * orthogonal).sum(dim = -1), torch.zeros(4, 5), atol = 1e-6)

def test_vector_components_dim():
    torch.manual_seed(0)

    v = torch.randn(3, 4)
    u = torch.randn(3)

    parallel, orthogonal = components(v, u, dim = 0)

    assert parallel.shape == v.shape
    assert torch.allclose(parallel + orthogonal, v)
    assert torch.allclose((parallel * orthogonal).sum(dim = 0), torch.zeros(4), atol = 1e-6)

def test_vector_components_zero_u():
    v = torch.tensor([1., 2., 3.])

    parallel, orthogonal = components(v, torch.zeros(3))

    assert torch.equal(parallel, torch.zeros_like(v))
    assert torch.equal(orthogonal, v)

@param('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_vector_components_dtypes(dtype):
    torch.manual_seed(0)

    v = torch.randn(5, dtype = dtype)
    u = torch.randn(5, dtype = dtype)

    parallel, orthogonal = components(v, u)

    assert parallel.dtype == dtype
    assert orthogonal.dtype == dtype
    assert torch.isfinite(parallel).all()
    assert torch.isfinite(orthogonal).all()

def test_vector_components_gradients():
    torch.manual_seed(0)

    v = torch.randn(3, 4, requires_grad = True)
    u = torch.randn(3, 4, requires_grad = True)

    parallel, orthogonal = components(v, u)

    (parallel.square().sum() + orthogonal.square().sum()).backward()

    assert torch.isfinite(v.grad).all()
    assert torch.isfinite(u.grad).all()

def test_slerp():
    v0 = torch.tensor([1., 0.])
    v1 = torch.tensor([0., 1.])

    out = slerp(v0, v1, 0.5)
    expected = torch.tensor([1., 1.]) / 2 ** 0.5

    assert torch.allclose(out, expected, atol = 1e-6)

@param('t', [0., 1.])
def test_slerp_endpoints(t):
    v0 = torch.tensor([2., 0., 0.])
    v1 = torch.tensor([0., 3., 0.])

    out = slerp(v0, v1, t)
    expected = normalize(v0 if t == 0. else v1, dim = -1)

    assert torch.allclose(out, expected, atol = 1e-6)

def test_slerp_parallel():
    v0 = torch.randn(5)

    out = slerp(v0, v0, 0.5)
    assert torch.allclose(out, normalize(v0, dim = -1), atol = 1e-6)

@param('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_slerp_parallel_dtypes(dtype):
    v = torch.randn(5, dtype = dtype)

    out = slerp(v, v, 0.5)

    assert out.dtype == dtype
    assert torch.isfinite(out).all()

def test_slerp_skip_normalize():
    v0, v1 = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    t = torch.rand(2, 3)

    out = slerp(v0, v1, t)
    expected = slerp(normalize(v0, dim = -1), normalize(v1, dim = -1), t, normalize = False)

    assert torch.allclose(out, expected, atol = 1e-6)

@param('shape, t_shape', [
    ((3, 4), (3,)),
    ((2, 3, 4), ()),
    ((2, 3, 4), (1,)),
    ((2, 3, 4), (3,)),
    ((2, 3, 4), (2, 3)),
    ((2, 3, 4), (2, 1)),
    ((2, 3, 4), (2, 3, 1)),
    ((2, 3, 2, 4), (2, 3, 2))
])
def test_slerp_leading_dims_and_time(shape, t_shape):
    v0, v1 = torch.randn(shape), torch.randn(shape)
    t = 0.3 if not t_shape else torch.rand(t_shape)

    out = slerp(v0, v1, t)

    t = torch.full(shape[:-1], 0.3) if not t_shape else t
    t = t[..., 0] if t.ndim > len(shape) - 1 else t
    t = t.expand(shape[:-1])

    expected = torch.stack([
        slerp(a, b, ti)
        for a, b, ti in zip(v0.reshape(-1, shape[-1]), v1.reshape(-1, shape[-1]), t.reshape(-1))
    ]).reshape(shape)

    assert out.shape == shape
    assert torch.allclose(out, expected, atol = 1e-6)

@param('t_shape', [(3,), (2, 3)])
def test_slerp_single_vector_expands_time(t_shape):
    v0, v1 = torch.randn(4), torch.randn(4)
    t = torch.rand(t_shape)

    out = slerp(v0, v1, t)
    expected = torch.stack([
        slerp(v0, v1, ti) for ti in t.reshape(-1)
    ]).reshape(*t_shape, 4)

    assert out.shape == (*t_shape, 4)
    assert torch.allclose(out, expected, atol = 1e-6)

def test_slerp_gradients():
    v0 = torch.randn(3, 4, requires_grad = True)
    v1 = torch.randn(3, 4, requires_grad = True)
    t = torch.rand(3, requires_grad = True)

    slerp(v0, v1, t).sum().backward()

    assert torch.isfinite(v0.grad).all()
    assert torch.isfinite(v1.grad).all()
    assert torch.isfinite(t.grad).all()
