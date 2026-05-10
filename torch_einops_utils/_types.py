from os import PathLike
from torch import Tensor
from torch.nn import Module
from typing import Any, Literal, ParamSpec, Protocol, TypeAlias, TypedDict, TypeVar

ConfigArgsKwargs:	TypeAlias = tuple[tuple[Any, ...], dict[Any, Any]]
"""Represent the positional arguments and keyword arguments of a decorated constructor.

`ConfigArgsKwargs` is a `TypeAlias` for a two-element `tuple` whose first element holds positional
arguments as a `tuple` and whose second element holds keyword arguments as a `dict`. The `save_load`
[1] decorator stores each decorated `torch.nn.Module` [2] instance's constructor arguments as a
`ConfigArgsKwargs` under a configurable attribute name. `dehydrate_config` [3] and `rehydrate_config`
[4] both accept and return `ConfigArgsKwargs` values when traversing checkpoint configuration
payloads.

See Also
--------
DehydratedCheckpoint : Represent the full serialized checkpoint, including a pickled
    `ConfigArgsKwargs` as `config`.
DehydratedTorchNNModule : Embed a `ConfigArgsKwargs` as the `config` field of a reconstruction
    record.

References
----------
[1] torch_einops_utils.save_load.save_load

[2] torch.nn.Module - PyTorch documentation
    https://pytorch.org/docs/stable/generated/torch.nn.Module.html
[3] torch_einops_utils.save_load.dehydrate_config

[4] torch_einops_utils.save_load.rehydrate_config
"""
PSpec =				ParamSpec("PSpec")
"""Capture the full parameter specification of a callable for higher-order type signatures.

`PSpec` is a `ParamSpec` [1] that preserves the complete parameter signature of a callable through
decorator transformations. `move_inputs_to_device` [2] and `move_inputs_to_module_device` [3] each
use `PSpec` so the wrapped callable retains its original parameter signature. `maybe` [4] uses
`PSpec` to forward all positional and keyword arguments beyond the first to the wrapped callable.

See Also
--------
RVar : Represent the return type paired with `PSpec` in higher-order callable signatures.
TVar : Represent the input or return type paired with `PSpec` in identity-preserving signatures.

References
----------
[1] ParamSpec - Python typing documentation
    https://docs.python.org/3/library/typing.html#typing.ParamSpec
[2] torch_einops_utils.device.move_inputs_to_device

[3] torch_einops_utils.device.move_inputs_to_module_device

[4] torch_einops_utils.maybe
"""
RVar =				TypeVar("RVar")
"""Represent a generic return type in higher-order callable signatures.

`RVar` is an unconstrained `TypeVar` [1] used to annotate the return type of an inner callable in
decorator and wrapper signatures. `maybe` [2] uses `RVar` to type the return value of the callable
being wrapped, keeping `RVar` distinct from `TVar` [3], which types the first argument.

See Also
--------
TVar : Represent the input type paired with `RVar` in wrapper signatures.
PSpec : Capture the parameter specification of the callable paired with `RVar`.

References
----------
[1] TypeVar - Python typing documentation
    https://docs.python.org/3/library/typing.html#typing.TypeVar
[2] torch_einops_utils.maybe

[3] torch_einops_utils.TVar
"""
StrPath:			TypeAlias =	str | PathLike[str]
"""Accept either a `str` or a `PathLike[str]` filesystem path value.

`StrPath` is a `TypeAlias` for `str | PathLike[str]` [1], intentionally copying the definition of
`_typeshed.StrPath` [2] from typeshed and mirroring the annotation accepted by `pathlib.Path` [3].
`StrPath` provides a single annotation for parameters that accept either a plain string path or any
path-like object. The generated `save` and `load` instance methods and the generated `init_and_load`
classmethod produced by `save_load` [4] each annotate their `path` parameter with `StrPath`.

References
----------
[1] os.PathLike - Python documentation
    https://docs.python.org/3/library/os.html#os.PathLike
[2] _typeshed.StrPath - typeshed
    https://github.com/python/typeshed/blob/main/stdlib/_typeshed/__init__.pyi
[3] pathlib.Path - Python documentation
    https://docs.python.org/3/library/pathlib.html#pathlib.Path
[4] torch_einops_utils.save_load.save_load
"""
T_co =				TypeVar("T_co", covariant=True)
"""Represent a covariant element type in read-only container protocols.

`T_co` is a covariant `TypeVar` [1] used in container protocols and functions that produce but do not
consume elements of a given type. `SupportsIntIndex` [2] uses `T_co` to annotate the element type
returned by `__getitem__`. `compact` [3] uses `T_co` for the element type in its input iterable and
its output `list`, preserving the element type while filtering `None` values.

See Also
--------
SupportsIntIndex : Container protocol parametrized by `T_co`.
TVar : Unconstrained TypeVar for operations where covariance is not required.

References
----------
[1] TypeVar - Python typing documentation
    https://docs.python.org/3/library/typing.html#typing.TypeVar
[2] torch_einops_utils.SupportsIntIndex

[3] torch_einops_utils.compact
"""
TorchNNModule =		TypeVar("TorchNNModule", bound=Module)
"""Represent a concrete `torch.nn.Module` subclass in decorator and method signatures.

`TorchNNModule` is a `TypeVar` [1] bound to `torch.nn.Module` [2] that lets static analyzers track
which specific subclass passes through a generic decorator or generic method. The `save_load` [3]
decorator uses `TorchNNModule` so that decorating a `Module` subclass preserves the concrete class
type in the return annotation rather than widening it to `type[Module]`.
`move_inputs_to_module_device` [4] uses `TorchNNModule` to preserve the specific `Module` subclass
type through its decorator.

See Also
--------
PSpec : Paired with `TorchNNModule` in `move_inputs_to_module_device` to capture the remaining
    parameters.

References
----------
[1] TypeVar - Python typing documentation
    https://docs.python.org/3/library/typing.html#typing.TypeVar
[2] torch.nn.Module - PyTorch documentation
    https://pytorch.org/docs/stable/generated/torch.nn.Module.html
[3] torch_einops_utils.save_load.save_load

[4] torch_einops_utils.device.move_inputs_to_module_device
"""
TVar =				TypeVar("TVar")
"""Represent an unconstrained generic type for identity-preserving operations.

`TVar` is an unconstrained `TypeVar` [1] used to annotate functions and protocols that accept a value
of any type and return a value of the same type. `exists` [2], `default` [3], `identity` [4], and
`map_values` [5] all use `TVar` to express that their output type matches their input type.
`IdentityCallable` [6] uses `TVar` to annotate both the input and output of its `__call__` method.
`first` [7] uses `TVar` via `SupportsIntIndex[TVar]` to propagate the element type of its argument to
the return type.

See Also
--------
T_co : Covariant TypeVar for read-only container protocols requiring covariance.
RVar : Separate TypeVar for the return type in higher-order callable signatures where input and
    return types differ.
IdentityCallable : Protocol using `TVar` to type identity-preserving callables.

References
----------
[1] TypeVar - Python typing documentation
    https://docs.python.org/3/library/typing.html#typing.TypeVar
[2] torch_einops_utils.exists

[3] torch_einops_utils.default

[4] torch_einops_utils.identity

[5] torch_einops_utils.map_values

[6] torch_einops_utils.IdentityCallable

[7] torch_einops_utils.first
"""

class DehydratedCheckpoint(TypedDict):
    """Represent the checkpoint dictionary written and read by the `save_load` decorator.

    `DehydratedCheckpoint` is a `TypedDict` [1] that defines the structure of the dictionary
    serialized by `torch.save` [2] and deserialized by `torch.load` [3] during checkpoint operations.
    The `save_load` [4] decorator generates instance methods and a classmethod that build and consume
    `DehydratedCheckpoint` values. The generated classmethod reads `config` to reconstruct
    constructor arguments before restoring model state.

    Attributes
    ----------
    model : dict[str, Tensor]
        The model state dictionary produced by `torch.nn.Module.state_dict` [5], mapping parameter
        and buffer names to their `Tensor` values.
    config : bytes
        The constructor arguments serialized with `pickle.dumps` [6]. The bytes encode a pickled
        `ConfigArgsKwargs` [7] value after `dehydrate_config` [8] has replaced any nested decorated
        `Module` instances with `DehydratedTorchNNModule` [9] reconstruction records.
    version : str | None
        An optional version string written at save time. When both the stored version and the
        `save_load` `version` argument are set and differ under `packaging.version.parse` [10], the
        generated load method prints a notice but still restores model state.

    See Also
    --------
    DehydratedTorchNNModule : Reconstruction record that may appear as a leaf in the pickled
        `config` bytes.
    ConfigArgsKwargs : Type of the value unpickled from `config`.

    References
    ----------
    [1] TypedDict - Python typing documentation
        https://docs.python.org/3/library/typing.html#typing.TypedDict
    [2] torch.save - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.save.html
    [3] torch.load - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.load.html
    [4] torch_einops_utils.save_load.save_load

    [5] torch.nn.Module.state_dict - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict
    [6] pickle.dumps - Python documentation
        https://docs.python.org/3/library/pickle.html#pickle.dumps
    [7] torch_einops_utils.ConfigArgsKwargs

    [8] torch_einops_utils.save_load.dehydrate_config

    [9] torch_einops_utils.DehydratedTorchNNModule

    [10] packaging.version - packaging documentation
        https://packaging.pypa.io/en/stable/version.html
    """
    model: dict[str, Tensor]
    config: bytes
    version: str | None

class DehydratedTorchNNModule(TypedDict):
    """Represent the reconstruction record for a single decorated `torch.nn.Module` instance.

    `DehydratedTorchNNModule` is a `TypedDict` [1] that encodes everything needed to reconstruct a
    decorated `torch.nn.Module` [2] instance from a stored checkpoint. `dehydrate_config` [3]
    produces a `DehydratedTorchNNModule` for each nested module it encounters during traversal.
    `rehydrate_config` [4] recognizes a `DehydratedTorchNNModule` by the `__save_load_module__`
    sentinel and reconstructs the original module by calling `klass` with the unpacked `config`.

    Attributes
    ----------
    __save_load_module__ : Literal[True]
        A sentinel flag with value `True` that `rehydrate_config` uses to distinguish a
        `DehydratedTorchNNModule` record from other values during traversal.
    klass : type[Module]
        The decorated `torch.nn.Module` subclass to instantiate during rehydration.
    config : ConfigArgsKwargs
        The constructor arguments stored as a `ConfigArgsKwargs` value. `config` itself may contain
        further nested `DehydratedTorchNNModule` records that `rehydrate_config` will recursively
        reconstruct.

    See Also
    --------
    DehydratedCheckpoint : The outer checkpoint structure whose `config` bytes may contain pickled
        `DehydratedTorchNNModule` values.
    ConfigArgsKwargs : Type of the `config` field.

    References
    ----------
    [1] TypedDict - Python typing documentation
        https://docs.python.org/3/library/typing.html#typing.TypedDict
    [2] torch.nn.Module - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    [3] torch_einops_utils.save_load.dehydrate_config

    [4] torch_einops_utils.save_load.rehydrate_config
    """
    __save_load_module__: Literal[True]
    klass: type[Module]
    config: ConfigArgsKwargs

class DimAndValue(TypedDict, total=False):
    """Carry the optional `dim` and `value` keyword arguments for padding operations.

    `DimAndValue` is a `TypedDict` [1] with `total=False`, meaning both `dim` and `value` are
    optional. Padding functions in this package accept `**kwargs: Unpack[DimAndValue]` [2] to forward
    these two keyword arguments to `pad_at_dim` [3]. `pad_left_at_dim` [4] and `pad_right_at_dim` [5]
    both accept and forward `DimAndValue` keyword arguments.

    Attributes
    ----------
    dim : int
        The dimension along which padding is applied. When absent, `pad_at_dim` uses a default of
        `-1`, meaning the last dimension.
    value : float
        The fill value used for the padded positions. When absent, `pad_at_dim` uses a default of
        `0.0`.

    See Also
    --------
    pad_left_at_dim : Pad the left side of a tensor, accepting `**kwargs: Unpack[DimAndValue]`.
    pad_right_at_dim : Pad the right side of a tensor, accepting `**kwargs: Unpack[DimAndValue]`.

    References
    ----------
    [1] TypedDict - Python typing documentation
        https://docs.python.org/3/library/typing.html#typing.TypedDict
    [2] Unpack - Python typing documentation
        https://docs.python.org/3/library/typing.html#typing.Unpack
    [3] torch_einops_utils.pad_at_dim

    [4] torch_einops_utils.pad_left_at_dim

    [5] torch_einops_utils.pad_right_at_dim
    """
    dim: int
    value: float

class IdentityCallable(Protocol):
    """Match any callable that returns its first positional argument unchanged.

    `IdentityCallable` is a `Protocol` [1] that any callable satisfies when its `__call__` method
    accepts a positional argument `value` of type `TVar` and returns the same `TVar` value,
    regardless of any additional positional or keyword arguments. `maybe` [2] returns an
    `IdentityCallable` when its `fn` argument is `None`, and `identity` [3] is the concrete
    implementation of `IdentityCallable` used throughout this package.

    See Also
    --------
    identity : Concrete implementation of `IdentityCallable` used by `maybe` when `fn` is `None`.
    maybe : Returns an `IdentityCallable` when called with `fn=None`. TVar : TypeVar parametrizing
    the input and output type of `IdentityCallable.__call__`.

    References
    ----------
    [1] Protocol - Python typing documentation
        https://docs.python.org/3/library/typing.html#typing.Protocol
    [2] torch_einops_utils.maybe

    [3] torch_einops_utils.identity
    """
    def __call__(self, value: TVar, /, *args: object, **kwargs: object) -> TVar: ...

class SupportsIntIndex(Protocol[T_co]):
    """Match any object that supports integer indexing via `__getitem__`.

    `SupportsIntIndex` is a `Protocol` [1] that any object satisfies when it implements `__getitem__`
    with a single `int` index and returns a value of type `T_co`. `first` [2] uses `SupportsIntIndex`
    to accept any sequence-like argument without requiring it to be a `Sequence` or `list`
    specifically.

    See Also
    --------
    first : Uses `SupportsIntIndex` to accept any object that supports integer indexing.
    T_co : Covariant TypeVar that parametrizes the element type returned by `__getitem__`.

    References
    ----------
    [1] Protocol - Python typing documentation
        https://docs.python.org/3/library/typing.html#typing.Protocol
    [2] torch_einops_utils.first
    """
    def __getitem__(self, index: int, /) -> T_co: ...

