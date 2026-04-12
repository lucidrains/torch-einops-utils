from __future__ import annotations
from pathlib import Path
from packaging import version as packaging_version

import pickle
from functools import wraps

import torch
from torch.nn import Module
from torch import Tensor

from collections.abc import Callable
from os import PathLike
from typing import Any, Literal, TypeAlias, TypedDict, TypeGuard, TypeVar, cast, overload

TorchNNModule = TypeVar('TorchNNModule', bound=Module)
TVar = TypeVar('TVar')
StrPath: TypeAlias = str | PathLike[str]

ConfigArgsKwargs: TypeAlias = tuple[tuple[Any, ...], dict[Any, Any]]

class DehydratedTorchNNModule(TypedDict):
    __save_load_module__: Literal[True]
    klass: type[Module]
    config: ConfigArgsKwargs

class DehydratedCheckpoint(TypedDict):
    model: dict[str, Tensor]
    config: bytes
    version: str | None

# helpers

def exists(v: TVar | None) -> TypeGuard[TVar]:
    return v is not None

def map_values(fn: Callable[[TVar], TVar], v: TVar) -> TVar:
    """Apply `fn` to every leaf value in a nested `list`, `tuple`, or `dict` structure.

    You can use `map_values` to transform the leaf values of an arbitrarily nested container without
    changing its shape. `map_values` recurses into `list` and `tuple` elements and into `dict`
    values, reassembling each container using its original type. Any value that is not a `list`,
    `tuple`, or `dict` is treated as a leaf and passed directly to `fn`. `dehydrate_config` [1] and
    `rehydrate_config` [2] both use `map_values` to traverse nested checkpoint configuration
    structures.

    Parameters
    ----------
    fn : Callable[[TVar], TVar]
        The function to apply to each leaf value. `fn` receives each non-container value and must
        return a value of the same type.
    v : TVar
        The value to transform. May be a `list`, `tuple`, `dict`, or any leaf value.

    Returns
    -------
    transformed : TVar
        The input structure with all leaf values replaced by the results of `fn`.

    See Also
    --------
    dehydrate_config : Serialize nested `Module` instances using `map_values`.
    rehydrate_config : Reconstruct nested `Module` instances using `map_values`.

    References
    ----------
    [1] torch_einops_utils.save_load.dehydrate_config

    [2] torch_einops_utils.save_load.rehydrate_config

    [3] tests.test_helpers.test_map_values_transforms_structure
    """
    if isinstance(v, (list, tuple)):
        return type(v)(map_values(fn, el) for el in v)

    if isinstance(v, dict):
        v = {key: map_values(fn, val) for key, val in v.items()}

    return fn(v)

def dehydrate_config(config: TVar, config_instance_var_name: str) -> TVar:
    """Convert nested decorated modules in `config` into reconstruction records.

    You can use `dehydrate_config` to replace each nested `torch.nn.Module` [1] in `config` that
    carries `config_instance_var_name` with a dictionary containing the module `class` and recorded
    constructor configuration. `dehydrate_config` preserves the surrounding `tuple`, `list`, and
    `dict` structure by delegating traversal to `map_values` [2]. The `save_load` decorator [3] uses
    `dehydrate_config` before serializing checkpoint configuration payloads.

    Parameters
    ----------
    config : TVar
        The constructor configuration to transform. `config` may contain plain Python values, nested
        containers, and decorated module instances.
    config_instance_var_name : str
        The attribute name that stores constructor arguments on each decorated module instance.

    Returns
    -------
    dehydrated_config : TVar
        A value with the same container structure as `config`, where each decorated module instance
        has been replaced by a reconstruction record.

    See Also
    --------
    rehydrate_config : Reconstruct module instances from dehydrated configuration records.
    save_load : Decorate a module class so constructor configuration can be dehydrated and restored.

    torch
    -----
    Only module instances that have an attribute named `config_instance_var_name` are converted.
    `save_load` [3] writes that attribute when the decorated `torch.nn.Module` [1] subclass records
    its constructor arguments. Undecorated module instances remain unchanged.

    Examples
    --------
    From `tests.test_save_load_extended.test_dehydrate_config_respects_config_instance_var_name` [4]:

        ```python
        from torch_einops_utils.save_load import dehydrate_config
        from tests.test_save_load_extended import SaveLoadExtendedCustomNamedModel

        custom_model = SaveLoadExtendedCustomNamedModel(13)
        config_args_kwargs = ((custom_model,), {})
        dehydrated_config = dehydrate_config(config_args_kwargs, 'stored_config')
        ```

    References
    ----------
    [1] torch.nn.Module - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    [2] torch_einops_utils.save_load.map_values

    [3] torch_einops_utils.save_load.save_load

    [4] tests.test_save_load_extended.test_dehydrate_config_respects_config_instance_var_name
    """
    @overload
    def dehydrate(v: Module) -> DehydratedTorchNNModule: ...
    @overload
    def dehydrate(v: TVar) -> TVar: ...
    def dehydrate(v: Module | TVar) -> DehydratedTorchNNModule | TVar:
        """Convert a candidate value to a reconstruction record when it is a decorated module instance.

        The function converts `v` to a `DehydratedTorchNNModule` [1] when `v` is a `torch.nn.Module`
        [2] that carries the `config_instance_var_name` attribute written by `save_load` [3]. All
        other values pass through unchanged.

        Parameters
        ----------
        v : Module | TVar
            The value to inspect and potentially convert.

        Returns
        -------
        converted_value : DehydratedTorchNNModule | TVar
            A reconstruction record when `v` is a decorated module instance, or `v` unchanged for all
            other values.

        References
        ----------
        [1] torch_einops_utils.DehydratedTorchNNModule

        [2] torch.nn.Module - PyTorch documentation
            https://pytorch.org/docs/stable/generated/torch.nn.Module.html

        [3] torch_einops_utils.save_load.save_load
        """
        if isinstance(v, Module) and hasattr(v, config_instance_var_name):
            return DehydratedTorchNNModule(
                __save_load_module__ = True,
                klass = v.__class__,
                config = dehydrate_config(getattr(v, config_instance_var_name), config_instance_var_name)
            )

        return cast(TVar, v)

    return map_values(dehydrate, config)

def rehydrate_config(config: ConfigArgsKwargs) -> ConfigArgsKwargs:
    """Reconstruct nested decorated modules from checkpoint configuration records.

    You can use `rehydrate_config` to replace each dictionary emitted by `dehydrate_config` [1] with
    a fresh `torch.nn.Module` [2] instance. `rehydrate_config` preserves the surrounding `tuple`,
    `list`, and `dict` structure by delegating traversal to `map_values` [3]. The classmethod
    generated by `save_load` [4] uses `rehydrate_config` to rebuild constructor arguments before it
    restores parameter values from the checkpoint.

    Parameters
    ----------
    config : ConfigArgsKwargs
        The dehydrated constructor configuration to transform.

    Returns
    -------
    rehydrated_config : ConfigArgsKwargs
        A configuration tuple whose module reconstruction records have been replaced by newly
        instantiated module objects.

    See Also
    --------
    dehydrate_config : Convert decorated module instances into reconstruction records.
    save_load : Decorate a module class so constructor configuration can be rehydrated during load.

    torch
    -----
    Each reconstruction record stores `klass` together with nested `(args, kwargs)` data.
    `rehydrate_config` calls `klass(*args, **kwargs)` to rebuild the module graph. `rehydrate_config`
    does not restore parameter tensors. The generated load method added by `save_load` [4] performs
    that state restoration after construction.

    Examples
    --------
    From `tests.test_save_load_extended` [5]:

        ```python
        from torch_einops_utils.save_load import rehydrate_config
        from tests.test_save_load_extended import SaveLoadExtendedLinearModel

        config_args_kwargs = (
            (
                {
                    '__save_load_module__': True,
                    'klass': SaveLoadExtendedLinearModel,
                    'config': ((7, 11), {}),
                },
            ),
            {'tag': 'manual-dehydrated-config'},
        )

        rehydrated_config = rehydrate_config(config_args_kwargs)
        ```

    References
    ----------
    [1] torch_einops_utils.save_load.dehydrate_config

    [2] torch.nn.Module - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    [3] torch_einops_utils.save_load.map_values

    [4] torch_einops_utils.save_load.save_load

    [5] tests.test_save_load_extended.test_rehydrate_config_instantiates_manual_dehydrated_modules
    """
    @overload
    def rehydrate(v: ConfigArgsKwargs) -> ConfigArgsKwargs: ...
    @overload
    def rehydrate(v: DehydratedTorchNNModule) -> Module: ...
    def rehydrate(v: DehydratedTorchNNModule | ConfigArgsKwargs) -> Module | ConfigArgsKwargs:
        """Instantiate a module from its reconstruction record when the value carries the `__save_load_module__` marker.

        The function calls `klass(*args, **kwargs)` to rebuild the module when `v` is a
        `DehydratedTorchNNModule` [1] dictionary. All other values pass through unchanged.

        Parameters
        ----------
        v : DehydratedTorchNNModule | ConfigArgsKwargs
            The value to inspect and potentially reconstruct as a module.

        Returns
        -------
        reconstructed : Module | ConfigArgsKwargs
            A newly instantiated `torch.nn.Module` [2] when `v` is a reconstruction record, or `v`
            unchanged for all other values.

        References
        ----------
        [1] torch_einops_utils.DehydratedTorchNNModule

        [2] torch.nn.Module - PyTorch documentation
            https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        """
        if isinstance(v, dict) and v.get('__save_load_module__', False):
            klass = v['klass']
            args, kwargs = v['config']
            return klass(*args, **kwargs)

        return cast(ConfigArgsKwargs, v)

    return map_values(rehydrate, config)

def save_load(
    save_method_name: str = 'save',
    load_method_name: str = 'load',
    config_instance_var_name: str = '_config',
    init_and_load_classmethod_name: str = 'init_and_load',
    version: str | None = None,
) -> Callable[[type[TorchNNModule]], type[TorchNNModule]]:
    """Decorate a `torch.nn.Module` subclass with checkpoint save and restore helpers.

    You can use `save_load` to wrap a `torch.nn.Module` [1] subclass so each instance records its
    constructor arguments and gains instance methods for checkpoint save and checkpoint load plus a
    classmethod that reconstructs a new instance from a checkpoint. The generated checkpoint payload
    stores model state, serialized constructor configuration, and an optional version string.
    `save_load` also uses `dehydrate_config` [2] and `rehydrate_config` [3] so constructor graphs
    that contain other decorated modules can round-trip through a checkpoint [4].

    Parameters
    ----------
    save_method_name : str = 'save'
        The name assigned to the generated instance method that writes a checkpoint.
    load_method_name : str = 'load'
        The name assigned to the generated instance method that reads model state from a checkpoint.
    config_instance_var_name : str = '_config'
        The attribute name used to store constructor arguments on each decorated instance.
    init_and_load_classmethod_name : str = 'init_and_load'
        The name assigned to the generated classmethod that instantiates a model from checkpoint
        configuration and then loads model state.
    version : str | None = None
        An optional version string written into each checkpoint and compared during load.

    Returns
    -------
    decorator : Callable[[type[TorchNNModule]], type[TorchNNModule]]
        A class decorator that mutates a `torch.nn.Module` subclass in place and returns the same
        class object.

    Raises
    ------
    TypeError
        Raised when the returned decorator is applied to a class that is not a subclass of
        `torch.nn.Module` [1].

    See Also
    --------
    dehydrate_config : Convert decorated module instances into checkpoint-safe configuration records.
    rehydrate_config : Reconstruct decorated module instances from stored configuration records.

    Generated Methods
    -----------------
    `save_method_name` : instance method
        Added to each instance. The generated method serializes the current model state, the
        dehydrated constructor configuration, and `version` to `path`. The generated method raises
        `FileExistsError` when `overwrite` is `False` and `path` already exists.
    `load_method_name` : instance method
        Added to each instance. The generated method reads the checkpoint at `path`, optionally
        prints a version-mismatch notice, and restores model state. The generated method raises
        `FileNotFoundError` when `path` does not exist.
    `init_and_load_classmethod_name` : classmethod
        Added to the decorated class. The generated classmethod reads constructor configuration from
        the checkpoint, rebuilds the module with `rehydrate_config` [3], and then restores model
        state. The generated classmethod raises `KeyError` when the checkpoint does not contain a
        `config` entry.

    torch
    -----
    The generated save method writes a dictionary that is compatible with `torch.save` [5], and the
    generated load paths read the dictionary with `torch.load` [6] on CPU before restoring state.
    When `version` and the stored checkpoint version both exist and differ under
    `packaging.version.parse` [7], the generated load method prints a notice but still restores the
    stored model state.

    Examples
    --------
    From `tests.test_save_load.test_init_and_load` [8]:

        ```python
        from pathlib import Path

        from torch import nn
        from torch_einops_utils.save_load import save_load

        @save_load()
        class SimpleNet(nn.Module):
            def __init__(self, dim, hidden_dim):
                super().__init__()
                self.dim = dim
                self.hidden_dim = hidden_dim
                self.net = nn.Linear(dim, hidden_dim)

        path = Path('test_model_init.pt')
        model = SimpleNet(10, 20)
        model.save(str(path))
        restored_model = SimpleNet.init_and_load(str(path))
        ```

    From `tests.test_save_load_extended` [9]:

        ```python
        import torch
        from torch import nn
        from torch_einops_utils.save_load import save_load

        @save_load(
            save_method_name='store',
            load_method_name='restore',
            config_instance_var_name='stored_config',
            init_and_load_classmethod_name='create_and_restore',
        )
        class SaveLoadExtendedCustomNamedModel(nn.Module):
            def __init__(self, width):
                super().__init__()
                self.width = width
                self.weight = nn.Parameter(torch.randn(width))

        model = SaveLoadExtendedCustomNamedModel(13)
        model.store('save-load-custom-methods.pt')
        restored_model = SaveLoadExtendedCustomNamedModel.create_and_restore(
            'save-load-custom-methods.pt'
        )
        ```

    References
    ----------
    [1] torch.nn.Module - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    [2] torch_einops_utils.save_load.dehydrate_config

    [3] torch_einops_utils.save_load.rehydrate_config

    [4] tests.test_save_load_extended.test_save_load_init_and_load_rehydrates_nested_modules

    [5] torch.save - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.save.html
    [6] torch.load - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.load.html
    [7] packaging.version - packaging documentation
        https://packaging.pypa.io/en/stable/version.html
    [8] tests.test_save_load.test_init_and_load

    [9] tests.test_save_load_extended.test_save_load_supports_custom_method_names_and_config_storage
    """
    def _save_load(klass: type[TorchNNModule]) -> type[TorchNNModule]:
        if not issubclass(klass, Module):
            message: str = 'save_load should decorate a subclass of torch.nn.Module'
            raise TypeError(message)

        _orig_init: Callable[..., None] = klass.__init__

        @wraps(_orig_init)
        def __init__(self: TorchNNModule, *args: Any, **kwargs: Any) -> None:
            setattr(self, config_instance_var_name, (args, kwargs))
            _orig_init(self, *args, **kwargs)

        def _save(self: TorchNNModule, path: StrPath, overwrite: bool = True) -> None:
            """Save the current model state and constructor configuration to a checkpoint file.

            You can use this method to persist a decorated module instance. The method dehydrates
            constructor configuration via `dehydrate_config` [1], packs it with the current state
            dict and the version string captured at decoration time, and writes the bundle via
            `torch.save` [2].

            Parameters
            ----------
            path : StrPath
                The filesystem path to write the checkpoint to.
            overwrite : bool = True
                When `False`, raise `FileExistsError` if a file already exists at `path`. When
                `True`, an existing file at `path` is silently replaced.

            Returns
            -------
            None

            Raises
            ------
            FileExistsError
                Raised when `overwrite` is `False` and a file already exists at `path`.

            References
            ----------
            [1] torch_einops_utils.save_load.dehydrate_config

            [2] torch.save - PyTorch documentation
                https://pytorch.org/docs/stable/generated/torch.save.html
            """
            path = Path(path)
            if not overwrite and path.exists():
                message: str = f'I received `{path = }`, but the file already exists and `overwrite` is `False`.'
                raise FileExistsError(message)

            config = getattr(self, config_instance_var_name)
            pkg = DehydratedCheckpoint(
                model = self.state_dict(),
                config = pickle.dumps(dehydrate_config(config, config_instance_var_name)),
                version = version,
            )

            torch.save(pkg, str(path))

        def _load(self: TorchNNModule, path: StrPath | Path, strict: bool = True) -> None:
            """Restore model state from a checkpoint file.

            You can use this method to load parameter values into an already-constructed decorated
            module instance. The method reads the checkpoint via `torch.load` [1] on CPU, emits a
            `UserWarning` when the stored version and the decoration-time version both exist and
            differ under `packaging.version.parse` [2], and then restores parameter values via
            `load_state_dict`.

            Parameters
            ----------
            path : StrPath | Path
                The filesystem path to read the checkpoint from.
            strict : bool = True
                Forwarded to `load_state_dict`. When `True`, the key sets of the checkpoint and the
                current model must match exactly.

            Returns
            -------
            None

            Raises
            ------
            FileNotFoundError
                Raised when no file exists at `path`.

            Warns
            -----
            UserWarning
                Emitted when the checkpoint's stored version and the decoration-time version both
                exist and differ under `packaging.version.parse` [2].

            References
            ----------
            [1] torch.load - PyTorch documentation
                https://pytorch.org/docs/stable/generated/torch.load.html

            [2] packaging.version - packaging documentation
                https://packaging.pypa.io/en/stable/version.html
            """
            path = Path(path)
            if not path.exists():
                message: str = f'I received `{path = }`, but no file exists at that path.'
                raise FileNotFoundError(message)

            pkg: DehydratedCheckpoint = torch.load(str(path), map_location = 'cpu')

            if exists(version) and exists(pkg['version']) and packaging_version.parse(version) != packaging_version.parse(pkg['version']):
                message: str = f'loading saved model at version {pkg["version"]}, but current package version is {version}'
                print(message)

            self.load_state_dict(pkg['model'], strict = strict)

        @classmethod
        def _init_and_load_from(cls: type[TorchNNModule], path: StrPath | Path, strict: bool = True) -> TorchNNModule:
            """Construct a new instance of the decorated class and restore its state from a checkpoint file.

            You can use this classmethod to reconstruct a model that was previously saved with the
            corresponding save method. The classmethod reads the checkpoint, unpickles the `config`
            entry, rebuilds the module graph via `rehydrate_config` [1], and then restores parameter
            values from the same checkpoint.

            Parameters
            ----------
            path : StrPath | Path
                The filesystem path to read the checkpoint from.
            strict : bool = True
                Forwarded to `load_state_dict`. When `True`, the key sets of the checkpoint and the
                reconstructed model must match exactly.

            Returns
            -------
            model : TorchNNModule
                A newly instantiated and state-restored instance of the decorated class.

            Raises
            ------
            FileNotFoundError
                Raised when no file exists at `path`.
            KeyError
                Raised when the checkpoint does not contain a `config` entry.

            References
            ----------
            [1] torch_einops_utils.save_load.rehydrate_config
            """
            path = Path(path)
            if not path.exists():
                message: str = f'I received `{path = }`, but no file exists at that path.'
                raise FileNotFoundError(message)
            pkg: DehydratedCheckpoint = torch.load(str(path), map_location = 'cpu')

            if 'config' not in pkg:
                message: str = 'model configs were not found in this saved checkpoint'
                raise KeyError(message)

            config: ConfigArgsKwargs = pickle.loads(pkg['config'])
            args, kwargs = rehydrate_config(config)
            model: TorchNNModule = cls(*args, **kwargs)

            _load(model, path, strict = strict)
            return model

        # set decorated init as well as save, load, and init_and_load

        klass.__init__ = __init__
        setattr(klass, save_method_name, _save)
        setattr(klass, load_method_name, _load)
        setattr(klass, init_and_load_classmethod_name, _init_and_load_from)

        return klass

    return _save_load
