import torch
from torch import tensor

from torch_einops_utils.torch_einops_utils import (
    exists,
    maybe,
    pad_ndim,
    align_dims_left,
    pad_at_dim,
    pad_left_at_dim,
    pad_right_at_dim,
    pad_sequence,
    lens_to_mask,
    and_masks,
    or_masks,
    tree_flatten_with_inverse,
    pack_with_inverse,
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

def test_tree_flatten_with_inverse():
    tree = (1, (2, 3), 4)
    (first, *rest), inverse = tree_flatten_with_inverse(tree)

    out = inverse((first + 1, *rest))
    assert out == (2, (2, 3), 4)

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
