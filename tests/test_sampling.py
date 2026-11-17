import pytest
import torch

from torch_einops_utils import uniform, uniform_like

param = pytest.mark.parametrize

def test_uniform_single_bound_is_symmetric():
    samples = uniform((10000,), 0.5)

    assert samples.shape == (10000,)
    assert samples.min() >= -0.5
    assert samples.max() <= 0.5
    assert samples.mean().abs() < 0.05

def test_uniform_single_negative_bound_is_symmetric():
    samples = uniform((10000,), -0.5)

    assert samples.min() >= -0.5
    assert samples.max() <= 0.5

def test_uniform_both_bounds():
    samples = uniform((10000,), low = 0., high = 1.)

    assert samples.min() >= 0.
    assert samples.max() <= 1.
    assert 0.45 < samples.mean() < 0.55

def test_uniform_default():
    samples = uniform((10000,))

    assert samples.min() >= -1.
    assert samples.max() <= 1.

def test_uniform_keyword_high():
    samples = uniform((10000,), high = 0.25)

    assert samples.min() >= -0.25
    assert samples.max() <= 0.25

def test_uniform_per_dim_scale():
    scale = torch.tensor([0.1, 0.2, 0.3])
    samples = uniform((10000, 3), scale)

    assert samples.shape == (10000, 3)
    assert (samples.abs() <= scale).all()

def test_uniform_like():
    t = torch.randn(3, 7)
    samples = uniform_like(t, 0.25)

    assert samples.shape == t.shape
    assert samples.dtype == t.dtype
    assert samples.device == t.device
    assert samples.min() >= -0.25
    assert samples.max() <= 0.25

def test_uniform_like_preserves_dtype():
    t = torch.randn(3, 7, dtype = torch.float64)
    samples = uniform_like(t, 0.1)

    assert samples.dtype == torch.float64

def test_uniform_generator():
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)

    assert torch.equal(uniform((10,), 1., generator = g1), uniform((10,), 1., generator = g2))
