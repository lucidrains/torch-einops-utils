import pytest
import numpy as np
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
    pad_sequence_and_cat,
    lens_to_mask,
    and_masks,
    or_masks,
    tree_flatten_with_inverse,
    tree_map_tensor,
    pack_with_inverse,
    masked_reduce,
    masked_mean,
    masked_sum,
    exclusive_cumsum,
    slice_at_dim,
    slice_left_at_dim,
    slice_right_at_dim,
    safe_stack,
    safe_cat,
    mask_after,
    mask_before,
    shift_right,
    shift_left,
    reverse_cumsum,
    batched_index_select,
    pad_right_ndim_to_and_expand_as,
    repeat_interleave_to_match,
    detach_tensor,
    tree_map_detach,
    cast_tensor,
    cast_item,
    clamp
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

def test_pad_sequence_uneven_images():
    images = [
        torch.randn(3, 16, 17),
        torch.randn(3, 15, 18),
        torch.randn(3, 17, 16)
    ]

    padded_height = pad_sequence(images, dim = -2, return_stacked = False)
    assert len(padded_height) == 3
    assert all([t.shape[1] == 17 for t in padded_height])

    stacked = pad_sequence_and_cat(padded_height, dim_cat = 0)
    assert stacked.shape == (9, 17, 18)

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

    res_keepdim = masked_mean(t, mask = mask, dim = (1, 2), keepdim = True)
    assert res_keepdim.shape == (2, 1, 1)
    assert torch.allclose(res_keepdim.squeeze(), res)

    res_no_mask_keepdim = masked_mean(t, dim = (1, 2), keepdim = True)
    assert res_no_mask_keepdim.shape == (2, 1, 1)
    assert torch.allclose(res_no_mask_keepdim.squeeze(), t.mean(dim = (1, 2)))

    res_slice = masked_mean(t, mask = mask, dim = slice(1, None))
    assert torch.allclose(res_slice, res)

    # einops reduce parity
    from einops import reduce
    assert torch.allclose(masked_mean(t, dim = slice(1, None)), reduce(t, 'b ... -> b', 'mean'))

def test_masked_sum():
    t = tensor([1., 2., 3., 4.])
    assert torch.allclose(masked_sum(t), tensor(10.0))
    assert torch.allclose(masked_sum(t, dim = 0), tensor(10.0))

    mask = tensor([True, False, True, False])
    assert torch.allclose(masked_sum(t, mask = mask), tensor(4.0))

    mask = tensor([False, False, False, False])
    assert torch.allclose(masked_sum(t, mask = mask), tensor(0.0))

    t = tensor([[1., 2.], [3., 4.]])
    mask = tensor([[True, False], [True, True]])

    assert torch.allclose(masked_sum(t, mask = mask, dim = 0), tensor([4.0, 4.0]))
    assert torch.allclose(masked_sum(t, mask = mask, dim = 1), tensor([1.0, 7.0]))

    t = torch.randn(2, 3, 4)
    mask = torch.ones(2, 3, 4).bool()
    mask[0, :, :] = False

    res = masked_sum(t, mask = mask, dim = (1, 2))
    assert res.shape == (2,)
    assert torch.allclose(res[0], tensor(0.0))
    assert torch.allclose(res[1], t[1].sum())

    t = torch.randn(2, 3, 4)
    mask = tensor([True, False])
    res = masked_sum(t, mask = mask, dim = (1, 2))
    assert res.shape == (2,)
    assert torch.allclose(res[0], t[0].sum())
    assert torch.allclose(res[1], tensor(0.0))

    res_keepdim = masked_sum(t, mask = mask, dim = (1, 2), keepdim = True)
    assert res_keepdim.shape == (2, 1, 1)
    assert torch.allclose(res_keepdim.squeeze(), res)

    res_no_mask_keepdim = masked_sum(t, dim = (1, 2), keepdim = True)
    assert res_no_mask_keepdim.shape == (2, 1, 1)
    assert torch.allclose(res_no_mask_keepdim.squeeze(), t.sum(dim = (1, 2)))

    # int tensor test
    t_int = tensor([1, 2, 3, 4])
    mask_int = tensor([True, False, True, False])
    assert masked_sum(t_int, mask = mask_int) == 4
    assert (masked_sum(tensor([[1, 2], [3, 4]]), mask = tensor([[True, False], [True, True]]), dim = -1) == tensor([1, 7])).all()

def test_masked_reduce():
    t = tensor([1., 2., 3., 4.])
    mask = tensor([True, False, True, False])
    assert torch.allclose(masked_reduce(t, mode = 'mean', mask = mask), tensor(2.0))
    assert torch.allclose(masked_reduce(t, mode = 'sum', mask = mask), tensor(4.0))
    assert torch.allclose(masked_reduce(t, mode = 'none', mask = mask), tensor([1., 0., 3., 0.]))
    assert torch.allclose(masked_reduce(t, mode = 'none'), t)

    with pytest.raises(AssertionError):
        masked_reduce(t, mode = 'invalid', mask = mask)

def test_z_score():
    from torch_einops_utils.statistics import z_score

    t = tensor([1., 2., 3., 4., 5.])
    out = z_score(t)
    assert torch.allclose(out.mean(), tensor(0.0), atol = 1e-6)
    assert torch.allclose(out.std(correction = 0), tensor(1.0), atol = 1e-3)

    mask = tensor([True, True, True, False, False])
    out = z_score(t, mask = mask)
    assert torch.allclose(out[3], tensor(0.0))
    assert torch.allclose(out[4], tensor(0.0))
    assert torch.allclose(out[:3].mean(), tensor(0.0), atol = 1e-6)

    t = torch.randn(3, 4)
    out = z_score(t, dim = 1)
    assert out.shape == t.shape
    assert torch.allclose(out.mean(dim = 1), torch.zeros(3), atol = 1e-5)
    assert torch.allclose(out.std(dim = 1, correction = 0), torch.ones(3), atol = 1e-3)

    out_all = z_score(t)
    assert out_all.shape == t.shape

    # unaligned mask dimension test (2D mask with 3D tensor)
    t_3d = torch.randn(2, 3, 4)
    mask_2d = torch.tensor([[True, False, True], [False, True, False]])
    out_3d = z_score(t_3d, mask = mask_2d, dim = (1, 2))
    assert out_3d.shape == (2, 3, 4)
    assert (out_3d[0, 1] == 0.0).all()
    assert (out_3d[1, 0] == 0.0).all()
    assert (out_3d[1, 2] == 0.0).all()

    # 1D mask with 3D tensor test
    mask_1d = torch.tensor([True, False])
    out_3d_1dmask = z_score(t_3d, mask = mask_1d, dim = (1, 2))
    assert out_3d_1dmask.shape == (2, 3, 4)
    assert (out_3d_1dmask[1] == 0.0).all()

def test_exclusive_cumsum():
    t = tensor([1., 2., 3., 4.])
    assert torch.allclose(exclusive_cumsum(t), tensor([0., 1., 3., 6.]))

    t = tensor([[1., 2.], [3., 4.]])
    assert torch.allclose(exclusive_cumsum(t, dim = 0), tensor([[0., 0.], [1., 2.]]))
    assert torch.allclose(exclusive_cumsum(t, dim = 1), tensor([[0., 1.], [0., 3.]]))

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
    assert safe_stack([t1]).shape == (1, 2, 3)
    assert safe_stack([t1, t2]).shape == (2, 2, 3)

    assert safe_cat([]) is None
    assert safe_cat([None]) is None
    assert (safe_cat([t1]) == t1).all()
    assert (safe_cat([t1, None]) == t1).all()
    assert safe_cat([t1, t2]).shape == (4, 3)

def test_mask_after_before():
    t = tensor([[1, 2, 3, 4, 5], [1, 3, 2, 3, 5]])

    assert mask_after(t, 3).tolist() == [
        [True, True, True, False, False],
        [True, True, False, False, False]
    ]

    assert mask_after(t, 3, inclusive = False).tolist() == [
        [True, True, False, False, False],
        [True, False, False, False, False]
    ]

    assert mask_before(t, 3).tolist() == [
        [False, False, True, True, True],
        [False, False, False, True, True]
    ]

    assert mask_before(t, 3, inclusive = False).tolist() == [
        [False, False, False, True, True],
        [False, False, False, False, True]
    ]

    assert mask_after(t.T, 3, dim = 0).tolist() == mask_after(t, 3).T.tolist()
    assert mask_before(t.T, 3, dim = 0).tolist() == mask_before(t, 3).T.tolist()

def test_eos_id_masking():
    seq = tensor([
        [1, 4, 5, 2, 0, 0],
        [1, 6, 2, 0, 0, 0],
        [1, 7, 8, 9, 2, 0]
    ])

    assert mask_after(seq, 2).tolist() == [
        [True, True, True, True, False, False],
        [True, True, True, False, False, False],
        [True, True, True, True, True, False]
    ]

    assert mask_after(seq, 2, inclusive = False).tolist() == [
        [True, True, True, False, False, False],
        [True, True, False, False, False, False],
        [True, True, True, True, False, False]
    ]

def test_shift():
    t = tensor([1, 2, 3])
    assert shift_right(t).tolist() == [0, 1, 2]
    assert shift_left(t).tolist() == [2, 3, 0]
    assert shift_right(t, pad_value = -1).tolist() == [-1, 1, 2]

def test_reverse_cumsum():
    t = tensor([1, 2, 3])
    assert reverse_cumsum(t).tolist() == [6, 5, 3]

def test_pad_right_ndim_to_and_expand_as():
    target = torch.randn(2, 8, 64)
    source = torch.randint(0, 8, (2, 4))
    assert pad_right_ndim_to_and_expand_as(source, target).shape == (2, 4, 64)

    dest = torch.zeros(2, 8, 64)
    source = torch.arange(4).unsqueeze(0).expand(2, -1)

    scattered = dest.scatter(1, pad_right_ndim_to_and_expand_as(source, dest), torch.ones(2, 4, 64))

    assert (scattered[:, :4] == 1.).all()
    assert (scattered[:, 4:] == 0.).all()

def test_repeat_interleave_to_match():
    time_lens = torch.tensor([2, 3])

    out = repeat_interleave_to_match(time_lens, torch.randn(4, 512))
    assert out.tolist() == [2, 2, 3, 3]

    out2 = repeat_interleave_to_match(time_lens, torch.randn(6, 128))
    assert out2.tolist() == [2, 2, 2, 3, 3, 3]

    out3 = repeat_interleave_to_match(time_lens, 6)
    assert out3.tolist() == [2, 2, 2, 3, 3, 3]

def test_batched_index_select():
    values = torch.randn(2, 5, 4)
    indices = torch.tensor([1, 3])

    out = batched_index_select(values, indices)
    assert out.shape == (2, 4)
    assert torch.allclose(out[0], values[0, 1])
    assert torch.allclose(out[1], values[1, 3])

    indices_2d = torch.tensor([[1, 2], [3, 4]])
    out2 = batched_index_select(values, indices_2d)
    assert out2.shape == (2, 2, 4)
    assert torch.allclose(out2[0, 0], values[0, 1])
    assert torch.allclose(out2[1, 1], values[1, 4])

    v = torch.randn(2, 3, 5, 4)
    i = torch.tensor([[[0, 1], [1, 2], [3, 4]], [[4, 3], [2, 1], [0, 0]]])
    out3 = batched_index_select(v, i, dim=2)
    assert out3.shape == (2, 3, 2, 4)
    assert torch.allclose(out3[0, 1, 0], v[0, 1, i[0, 1, 0]])
    assert torch.allclose(out3[1, 2, 1], v[1, 2, i[1, 2, 1]])

def test_detach_tensor():
    t = torch.randn(3, requires_grad=True)
    out = detach_tensor(t)
    assert not out.requires_grad
    assert t.requires_grad

    out_preserve = detach_tensor(t, preserve_requires_grad=True)
    assert out_preserve.requires_grad
    assert out_preserve.data_ptr() != t.data_ptr() or out_preserve is not t

    out_clone = detach_tensor(t, clone=True)
    assert not out_clone.requires_grad
    assert out_clone.data_ptr() != t.data_ptr()

def test_tree_map_detach():
    t1 = torch.randn(3, requires_grad=True)
    t2 = torch.randn(4, requires_grad=True)
    tree = (t1, [t2, {'a': torch.randn(5)}])

    out_tree = tree_map_detach(tree)
    assert not out_tree[0].requires_grad
    assert not out_tree[1][0].requires_grad
    assert not out_tree[1][1]['a'].requires_grad

    out_tree_preserve = tree_map_detach(tree, preserve_requires_grad=True)
    assert out_tree_preserve[0].requires_grad
    assert out_tree_preserve[1][0].requires_grad

def test_cast_tensor():
    assert cast_tensor(tensor([1, 2, 3]), dtype = torch.float32).dtype == torch.float32
    assert cast_tensor(tensor([1, 2, 3]), dtype = torch.float32, device = 'cpu').device.type == 'cpu'
    assert cast_tensor(1, dtype = torch.float32).dtype == torch.float32
    assert cast_tensor(1.0, dtype = torch.long).dtype == torch.long
    assert cast_tensor([1, 2, 3]).shape == (3,)
    assert cast_tensor((1.0, 2.0), dtype = torch.float32).dtype == torch.float32
    assert cast_tensor([1, 2, 3], device = 'cpu').device.type == 'cpu'
    assert cast_tensor(tensor([1, 2, 3]), device = torch.device('cpu')).device.type == 'cpu'

    assert cast_tensor(None) is None
    assert cast_tensor('hello') == 'hello'

    with pytest.raises(TypeError):
        cast_tensor('hello', error = True)

def test_cast_item():
    assert cast_item(tensor(3)) == 3
    assert cast_item(tensor(3.)) == 3.0
    assert cast_item(3) == 3
    assert cast_item(3.0) == 3.0
    assert cast_item('hello') == 'hello'
    assert cast_item(None) is None

@pytest.mark.parametrize('t, lo, hi, expected, inplace', [
    (5, 0, 3, 3, False),
    (1.5, 0., 1., 1.0, False),
    (5, 10, None, 10, False),
    (5, None, 3, 3, False),
    (5, None, None, 5, False),
    (5, 0, 3, 3, True),
    (tensor([1., 2., 3., 4.]), 2., 3., tensor([2., 2., 3., 3.]), False),
    (tensor([1., 2., 3., 4.]), 2., 3., tensor([2., 2., 3., 3.]), True),
    (tensor([1, 2, 3, 4]), 2, 3, tensor([2, 2, 3, 3]), True),
    (tensor([1., 2., 3., 4.]), None, 3., tensor([1., 2., 3., 3.]), True),
    (tensor([1., 2., 3., 4.]), None, None, tensor([1., 2., 3., 4.]), False),
    (np.array([1., 2., 3., 4.]), 2., 3., np.array([2., 2., 3., 3.]), False),
    (np.array([1., 2., 3., 4.]), 2., 3., np.array([2., 2., 3., 3.]), True),
    (np.array([1., 2., 3., 4.]), None, 3., np.array([1., 2., 3., 3.]), True),
])

def test_clamp(t, lo, hi, expected, inplace):
    out = clamp(t, lo = lo, hi = hi, inplace = inplace)

    if isinstance(out, (int, float)):
        assert out == expected
    elif torch.is_tensor(out):
        assert torch.allclose(out, expected)
    elif isinstance(out, np.ndarray):
        assert np.array_equal(out, expected)

    if inplace and not isinstance(t, (int, float)):
        assert out is t
