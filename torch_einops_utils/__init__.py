# isort: split
from __future__ import annotations

from typing import ParamSpec, Protocol, TypedDict, TypeVar

from torch_einops_utils._helpers import (
    compact as compact,
    default as default,
    exists as exists,
    first as first,
    identity as identity,
    safe as safe
)

# isort: split
from torch_einops_utils._slicing import (
    shape_with_replace as shape_with_replace,
    slice_at_dim as slice_at_dim,
    slice_left_at_dim as slice_left_at_dim,
    slice_right_at_dim as slice_right_at_dim
)

# isort: split
from torch_einops_utils._dimensions import (
    align_dims_left as align_dims_left,
    pad_left_ndim as pad_left_ndim,
    pad_left_ndim_to as pad_left_ndim_to,
    pad_ndim as pad_ndim,
    pad_right_ndim as pad_right_ndim,
    pad_right_ndim_to as pad_right_ndim_to
)

# isort: split
from torch_einops_utils._masking import (
    and_masks as and_masks,
    lens_to_mask as lens_to_mask,
    or_masks as or_masks,
    reduce_masks as reduce_masks
)

# isort: split
from torch_einops_utils._padding import (
    pad_at_dim as pad_at_dim,
    pad_left_at_dim as pad_left_at_dim,
    pad_left_at_dim_to as pad_left_at_dim_to,
    pad_right_at_dim as pad_right_at_dim,
    pad_right_at_dim_to as pad_right_at_dim_to,
    pad_sequence as pad_sequence,
    pad_sequence_and_cat as pad_sequence_and_cat
)

# isort: split
from torch_einops_utils._cat_stack import (
    safe_cat as safe_cat,
    safe_stack as safe_stack
)

# isort: split
from torch_einops_utils.torch_einops_utils import (
    masked_mean as masked_mean,
    maybe as maybe
)

# isort: split
from torch_einops_utils.torch_einops_utils import (
    tree_flatten_with_inverse as tree_flatten_with_inverse,
    tree_map_tensor as tree_map_tensor
)

# isort: split
from torch_einops_utils.torch_einops_utils import (
    pack_with_inverse as pack_with_inverse
)

DVar = TypeVar("DVar")
TVar = TypeVar("TVar")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

PSpec = ParamSpec("PSpec")


class DimAndValue(TypedDict, total=False):
    dim: int
    value: float


class DimValueLeft(DimAndValue, total=False):
    left: bool


class SupportsGetItem(Protocol[T_co]):
    def __getitem__(self, index: int) -> T_co: ...


class SupportsMod(Protocol[T_contra, T_co]):
    def __mod__(self, other: T_contra, /) -> T_co: ...


class SupportsRMod(Protocol[T_contra, T_co]):
    def __rmod__(self, other: T_contra, /) -> T_co: ...
