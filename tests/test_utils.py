import torch
from torch import tensor

from torch_einops_utils.torch_einops_utils import (
    exists,
    maybe,
    shape_with_replace,
    pad_ndim,
    pad_left_ndim,
    pad_right_ndim,
    pad_right_ndim_to,
    pad_left_ndim_to,
    align_dims_left,
    pad_at_dim,
    pad_left_at_dim,
    pad_right_at_dim,
    pad_left_at_dim_to,
    pad_right_at_dim_to,
    pad_sequence,
    lens_to_mask,
    and_masks,
    or_masks,
    tree_flatten_with_inverse,
    tree_map_tensor,
    pack_with_inverse,
    masked_mean,
    slice_at_dim,
    slice_left_at_dim,
    slice_right_at_dim,
    safe_stack,
    safe_cat
)

def test_exist():
    assert not exists(None)

def test_maybe():
    assert maybe(None)(1) == 1
    assert not exists(maybe(lambda t: t + 1)(None))

def test_pad_ndim():
    t = torch.randn(3)
    t = pad_ndim(t, (1, 2))
    assert t.shape == (1, 3, 1, 1)

    t = torch.randn(3)
    t = pad_right_ndim_to(t, 3)
    assert t.shape == (3, 1, 1)

    t = torch.randn(3, 4, 5)
    t = pad_right_ndim_to(t, 3)
    assert t.shape == (3, 4, 5)

    t = torch.randn(3)
    t = pad_left_ndim_to(t, 3)
    assert t.shape == (1, 1, 3)

def test_align_ndim_left():
    t = torch.randn(3)
    u = torch.randn(3, 5, 2)
    v = torch.randn(3, 5)

    t, u, v = align_dims_left((t, u, v))
    assert t.shape == (3, 1, 1)
    assert u.shape == (3, 5, 2)
    assert v.shape == (3, 5, 1)

def test_pad_at_dim():
    t = torch.randn(3, 6, 1)
    padded = pad_at_dim(t, (0, 1), dim = 1)

    assert padded.shape == (3, 7, 1)
    assert torch.allclose(padded, pad_right_at_dim(t, 1, dim = 1))
    assert not torch.allclose(padded, pad_left_at_dim(t, 1, dim = 1))

    t = torch.randn(3, 6, 1)
    padded = pad_right_at_dim_to(t, 7, dim = 1)
    assert padded.shape == (3, 7, 1)

    padded = pad_left_at_dim_to(t, 7, dim = 1)
    assert padded.shape == (3, 7, 1)

    padded = pad_right_at_dim_to(t, 6, dim = 1)
    assert padded.shape == (3, 6, 1)

def test_tree_flatten_with_inverse():
    tree = (1, (2, 3), 4)
    (first, *rest), inverse = tree_flatten_with_inverse(tree)

    out = inverse((first + 1, *rest))
    assert out == (2, (2, 3), 4)

def test_tree_map_tensor():
    tree = (1, tensor(2), 3)
    tree = tree_map_tensor(lambda t: t + 1, tree)
    assert tree[0] == 1
    assert tree[-1] == 3
    assert (tree[1] == 3).all()

def test_pack_with_inverse():
    t = torch.randn(3, 12, 2, 2)
    t, inverse = pack_with_inverse(t, 'b * d')

    assert t.shape == (3, 24, 2)
    t = inverse(t)
    assert t.shape == (3, 12, 2, 2)

    u = torch.randn(3, 4, 2)
    t, inverse = pack_with_inverse([t, u], 'b * d')
    assert t.shape == (3, 28, 2)

    t = t.sum(dim = -1)
    t, u = inverse(t, 'b *')
    assert t.shape == (3, 12, 2)
    assert u.shape == (3, 4)

def test_better_pad_sequence():

    x = torch.randn(2, 4, 5)
    y = torch.randn(2, 3, 5)
    z = torch.randn(2, 1, 5)

    packed, lens = pad_sequence([x, y, z], dim = 1, return_lens = True)
    assert packed.shape == (3, 2, 4, 5)
    assert lens.tolist() == [4, 3, 1]

    mask = lens_to_mask(lens)
    assert torch.allclose(mask.sum(dim = -1), lens)

def test_and_masks():
    assert not exists(and_masks([None]))

    mask1 = tensor([True, True])
    mask2 = tensor([True, False])
    assert (and_masks([mask1, None, mask2]) == tensor([True, False])).all()

def test_or_masks():
    assert not exists(or_masks([None]))

    mask1 = tensor([True, True])
    mask2 = tensor([True, False])
    assert (or_masks([mask1, None, mask2]) == tensor([True, True])).all()

def test_masked_mean():
    t = tensor([1., 2., 3., 4.])
    assert torch.allclose(masked_mean(t), tensor(2.5))
    assert torch.allclose(masked_mean(t, dim = 0), tensor(2.5))

    mask = tensor([True, False, True, False])
    assert torch.allclose(masked_mean(t, mask = mask), tensor(2.0))

    mask = tensor([False, False, False, False])
    assert torch.allclose(masked_mean(t, mask = mask), tensor(0.0))

    t = tensor([[1., 2.], [3., 4.]])
    mask = tensor([[True, False], [True, True]])

    assert torch.allclose(masked_mean(t, mask = mask, dim = 0), tensor([2.0, 4.0]))

    assert torch.allclose(masked_mean(t, mask = mask, dim = 1), tensor([1.0, 3.5]))

    t = torch.randn(2, 3, 4)
    mask = torch.ones(2, 3, 4).bool()
    mask[0, :, :] = False

    res = masked_mean(t, mask = mask, dim = (1, 2))
    assert res.shape == (2,)
    assert torch.allclose(res[0], tensor(0.0), atol = 1e-4)
    assert torch.allclose(res[1], t[1].mean())

    t = torch.randn(2, 3, 4)
    mask = tensor([True, False])
    res = masked_mean(t, mask = mask, dim = (1, 2))
    assert res.shape == (2,)
    assert torch.allclose(res[0], t[0].mean())
    assert torch.allclose(res[1], tensor(0.0), atol = 1e-4)

def test_slice_at_dim():
    t = torch.randn(3, 4, 5)

    res = slice_at_dim(t, slice(1, 3))
    assert res.shape == (3, 4, 2)
    assert torch.allclose(res, t[:, :, 1:3])

    res = slice_at_dim(t, slice(None, 2), dim = 1)
    assert res.shape == (3, 2, 5)
    assert torch.allclose(res, t[:, :2, :])

    res = slice_at_dim(t, slice(2, None), dim = -2)
    assert res.shape == (3, 2, 5)
    assert torch.allclose(res, t[:, 2:, :])

    res = slice_left_at_dim(t, 2, dim = 1)
    assert res.shape == (3, 2, 5)
    assert torch.allclose(res, t[:, :2, :])

    res = slice_right_at_dim(t, 2, dim = 1)
    assert res.shape == (3, 2, 5)
    assert torch.allclose(res, t[:, -2:, :])

def test_shape_with_replace():
    t = torch.randn(3, 4, 5)
    assert shape_with_replace(t, {1: 2}) == (3, 2, 5)

def test_safe_functions():
    t1 = torch.randn(2, 3)
    t2 = torch.randn(2, 3)

    assert safe_stack([]) is None
    assert safe_stack([None]) is None
    assert (safe_stack([t1]) == t1).all()
    assert (safe_stack([t1, None]) == t1).all()
    assert safe_stack([t1, t2]).shape == (2, 2, 3)

    assert safe_cat([]) is None
    assert safe_cat([None]) is None
    assert (safe_cat([t1]) == t1).all()
    assert (safe_cat([t1, None]) == t1).all()
    assert safe_cat([t1, t2]).shape == (4, 3)
