from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import wraps
from typing import Concatenate, TypeGuard, overload

from torch import Tensor

from torch_einops_utils import (
    DVar,
    IdentityCallable,
    PSpec,
    SupportsIntIndex,
    T_co
)


def compact(arr: Iterable[T_co | None]) -> list[T_co]:
    """Filter `None` values from `arr` and return the remaining elements as a `list`.

    You can use `compact` to remove all `None` values from an iterable, producing a typed `list` of
    non-`None` elements. `compact` applies `exists` [1] as the predicate for `filter` [2]. The `safe`
    [3] decorator uses `compact` to strip `None` values from a `Sequence` of `Tensor` before passing
    the result to the decorated function.

    Parameters
    ----------
    arr : Iterable[TVar | None]
        An iterable that may contain a mix of non-`None` values and `None` values to discard.

    Returns
    -------
    compacted : list[TVar]
        A `list` containing only the non-`None` elements of `arr`, in iteration order.

    See Also
    --------
    exists : Test whether a value is not `None`.
    safe : Decorator that applies `compact` to filter `None` values before calling the wrapped function.

    References
    ----------
    [1] torch_einops_utils._helpers.exists

    [2] filter - Python documentation
        https://docs.python.org/3/library/functions.html#filter
    [3] torch_einops_utils._helpers.safe
    """
    return [*filter(exists, arr)]


def default(v: T_co | None, d: T_co) -> T_co:
    """Return `v` when `v` is not `None`, or `d` when `v` is `None`.

    You can use `default` to supply a fallback value for optional parameters and accumulators.
    `default` calls `exists` [1] to test `v`. When `exists(v)` is `True`, `default` returns `v`
    unchanged. When `v` is `None`, `default` returns `d`. Two overloads ensure the return type
    narrows correctly under static analysis.

    Parameters
    ----------
    v : TVar | None
        The primary value to test.
    d : DVar
        The fallback value returned when `v` is `None`.

    Returns
    -------
    result : TVar | DVar
        `v` when `v` is not `None`, otherwise `d`.

    See Also
    --------
    exists : Test whether a value is not `None`.

    References
    ----------
    [1] torch_einops_utils._helpers.exists
    """
    return v if exists(v) else d


def divisible_by(num: float, den: float) -> bool:
    """Test whether `num` is evenly divisible by `den`.

    You can use `divisible_by` to check divisibility without raising a `ZeroDivisionError` when `den`
    is zero. `divisible_by` returns `False` whenever `den` is zero, and otherwise returns `True` when
    `num % den == 0`.

    Parameters
    ----------
    num : float
        The numerator to test.
    den : float
        The denominator. When `den` is `0`, `divisible_by` returns `False` without evaluating `num %
        den`.

    Returns
    -------
    is_divisible : bool
        `True` when `den != 0` and `num % den == 0`, otherwise `False`.
    """
    return (den != 0) and ((num % den) == 0)


def exists(v: T_co | None) -> TypeGuard[T_co]:
    """Test whether `v` is not `None`.

    You can use `exists` as a `None`-guard throughout this package. `exists` returns `True` for any
    value that is not `None`, including falsy values such as `0`, `False`, and empty collections. The
    return type is annotated as `TypeGuard[TVar]` [1] so that static analyzers narrow the type of `v`
    to `TVar` in branches guarded by `exists`.

    Parameters
    ----------
    v : TVar | None
        The value to test.

    Returns
    -------
    result : bool
        `True` when `v is not None`, otherwise `False`.

    See Also
    --------
    default : Return a fallback value when `v` is `None`.
    compact : Filter `None` values from an iterable.

    Examples
    --------
    From `torch_einops_utils._masking.lens_to_mask` [2], guarding optional parameter `max_len` before
    use:

        ```python
        if not exists(max_len):
            max_len = int(lens.amax().item())
        ```

    References
    ----------
    [1] TypeGuard - Python typing documentation
        https://docs.python.org/3/library/typing.html#typing.TypeGuard
    [2] torch_einops_utils._masking.lens_to_mask
    """
    return v is not None


def first(arr: SupportsIntIndex[T_co]) -> T_co:
    """Return the element at index `0` of `arr`.

    You can use `first` to retrieve the first element of any sequence that supports integer indexing
    via `SupportsIntIndex` [1]. `first` delegates to `arr[0]`.

    Parameters
    ----------
    arr : SupportsIntIndex[T_co]
        A sequence that supports integer indexing. Access with index `0` must be valid.

    Returns
    -------
    element : T_co
        The element at position `0` in `arr`.

    References
    ----------
    [1] torch_einops_utils._types.SupportsIntIndex
    """
    return arr[0]


def identity(t: T_co, *args: object, **kwargs: object) -> T_co:  # noqa: ARG001
    """Return `t` unchanged, ignoring all other arguments.

    You can use `identity` as a no-op callable in contexts that require a function but no
    transformation. `identity` accepts and discards any additional positional or keyword arguments,
    making `identity` a drop-in substitute for any unary or multi-argument callable.
    `torch_einops_utils.torch_einops_utils.maybe` [1] returns `identity` when its `fn` argument is
    `None`.

    Parameters
    ----------
    t : TVar
        The value to return unchanged.
    *args : Any
        Additional positional arguments. Accepted and discarded.
    **kwargs : Any
        Additional keyword arguments. Accepted and discarded.

    Returns
    -------
    result : TVar
        `t`, the same object that was passed as the first argument.

    See Also
    --------
    maybe : Return `identity` when `fn` is `None`, otherwise wrap `fn` to skip `None` inputs.

    References
    ----------
    [1] torch_einops_utils._helpers.identity
    """
    return t


@overload
def maybe(fn: Callable[Concatenate[DVar, PSpec], T_co]) -> Callable[Concatenate[DVar | None, PSpec], T_co | None]: ...
@overload
def maybe(fn: None) -> IdentityCallable: ...
def maybe(
    fn: Callable[Concatenate[DVar, PSpec], T_co] | None,
) -> Callable[Concatenate[DVar | None, PSpec], T_co | None] | IdentityCallable:
    """Wrap `fn` so that the wrapped function returns `None` when the first argument is `None`.

    You can use this function to conditionally apply `fn` without adding an explicit `if`/`else`
    guard at every call site. The returned callable passes all positional and keyword arguments to
    `fn` unchanged when the first argument is not `None`, and returns `None` immediately when the
    first argument is `None`. When `fn` is `None`, the function returns `identity` [1], which passes
    its first argument through without modification.

    Parameters
    ----------
    fn : Callable[Concatenate[DVar, PSpec], TVar] | None
        The callable to wrap, or `None`. The first positional parameter of `fn` is the value that may
        be `None`. Pass `None` to receive an identity function.

    Returns
    -------
    wrapped : Callable[Concatenate[DVar | None, PSpec], TVar | None] | Callable[..., TVar]
        A wrapped version of `fn` that short-circuits to `None` when the first argument is `None`, or
        `identity` when `fn` is `None`.

    See Also
    --------
    identity : Return the first argument unchanged.

    Examples
    --------
    Skip the function when the first argument is `None` [2]:

        ```python
        from torch_einops_utils import maybe

        result = maybe(lambda t: t + 1)(None)
        # result is None
        ```

    Pass `None` as `fn` to receive an identity function [2]:

        ```python
        result = maybe(None)(1)
        # result == 1
        ```

    Conditionally convert episode lengths to a mask [3]:

        ```python
        from torch_einops_utils import maybe, lens_to_mask

        mask = maybe(lens_to_mask)(episode_lens, seq_len)
        # mask is None when episode_lens is None,
        # otherwise mask == lens_to_mask(episode_lens, seq_len)
        ```

    References
    ----------
    [1] torch_einops_utils._helpers.identity

    [2] tests.test_utils.test_maybe

    [3] metacontroller.metacontroller.ratio_loss
        https://context7.com/lucidrains/metacontroller
    """
    if not exists(fn):
        return identity

    @wraps(fn)
    def inner(t: DVar | None, *args: PSpec.args, **kwargs: PSpec.kwargs) -> T_co | None:
        if not exists(t):
            return None

        return fn(t, *args, **kwargs)

    return inner


def safe(
    fn: Callable[Concatenate[Sequence[Tensor], PSpec], Tensor | None],
) -> Callable[Concatenate[Sequence[Tensor | None], PSpec], Tensor | None]:
    """Wrap `fn` so that `None` values are filtered from the first argument before the call.

    You can use `safe` as a decorator to make a function that accepts a `Sequence[Tensor]` tolerate
    `None` values in the sequence. The decorated function accepts a `Sequence[Tensor | None]`. Before
    calling `fn`, `safe` compacts [1] the sequence to remove all `None` values. When the compacted
    sequence is empty, `safe` returns `None` without calling `fn`. When at least one non-`None`
    `Tensor` remains, `safe` passes the compacted sequence to `fn`.

    `safe` is applied as a decorator to `safe_stack` [2] and `safe_cat` [3] to produce null-safe
    stacking and concatenation. `safe` is also applied to `reduce_masks` [4] to produce null-safe
    mask reduction.

    Parameters
    ----------
    fn : Callable[Concatenate[Sequence[Tensor], PSpec], Tensor | None]
        A callable whose first argument is a `Sequence[Tensor]` with at least one element. `fn` must
        handle a compacted sequence of any length ≥ 1.

    Returns
    -------
    wrapped : Callable[Concatenate[Sequence[Tensor | None], PSpec], Tensor | None]
        A wrapped version of `fn` that accepts `None` values in the first argument and returns `None`
        when no non-`None` `Tensor` values are present.

    See Also
    --------
    compact : Filter `None` values from an iterable.
    safe_cat : Concatenate tensors along an existing dimension, skipping `None` values.
    safe_stack : Stack tensors along a new dimension, skipping `None` values.

    References
    ----------
    [1] torch_einops_utils._helpers.compact

    [2] torch_einops_utils._cat_stack.safe_stack

    [3] torch_einops_utils._cat_stack.safe_cat

    [4] torch_einops_utils._masking.reduce_masks
    """

    @wraps(fn)
    def inner(tensors: Sequence[Tensor | None], *args: PSpec.args, **kwargs: PSpec.kwargs) -> Tensor | None:
        compacted: list[Tensor] = compact(tensors)
        if len(compacted) == 0:
            output: Tensor | None = None
        else:
            output = fn(compacted, *args, **kwargs)
        return output

    return inner
