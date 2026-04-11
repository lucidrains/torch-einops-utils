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
    if isinstance(v, (list, tuple)):
        return type(v)(map_values(fn, el) for el in v)

    if isinstance(v, dict):
        v = {key: map_values(fn, val) for key, val in v.items()}

    return fn(v)

def dehydrate_config(config: TVar, config_instance_var_name: str) -> TVar:
    @overload
    def dehydrate(v: Module) -> DehydratedTorchNNModule: ...
    @overload
    def dehydrate(v: TVar) -> TVar: ...
    def dehydrate(v: Module | TVar) -> DehydratedTorchNNModule | TVar:
        # if the value is a save_load decorated module, convert it to its reconstruction metadata
        if isinstance(v, Module) and hasattr(v, config_instance_var_name):
            return DehydratedTorchNNModule(
                __save_load_module__ = True,
                klass = v.__class__,
                config = dehydrate_config(getattr(v, config_instance_var_name), config_instance_var_name)
            )

        return cast(TVar, v)

    return map_values(dehydrate, config)

def rehydrate_config(config: ConfigArgsKwargs) -> ConfigArgsKwargs:
    @overload
    def rehydrate(v: ConfigArgsKwargs) -> ConfigArgsKwargs: ...
    @overload
    def rehydrate(v: DehydratedTorchNNModule) -> Module: ...
    def rehydrate(v: DehydratedTorchNNModule | ConfigArgsKwargs) -> Module | ConfigArgsKwargs:
        # if the value is reconstruction metadata, instantiate the module using its class and configuration
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
            path = Path(path)
            if not path.exists():
                message: str = f'I received `{path = }`, but no file exists at that path.'
                raise FileNotFoundError(message)

            pkg: DehydratedCheckpoint = torch.load(str(path), map_location = 'cpu')

            if exists(version) and exists(pkg['version']) and packaging_version.parse(version) != packaging_version.parse(pkg['version']):
                message: str = f'loading saved model at version {pkg["version"]}, but current package version is {version}'
                print(message)

            self.load_state_dict(pkg['model'], strict = strict)

        # init and load from
        # looks for a `config` key in the stored checkpoint, instantiating the model as well as loading the state dict

        @classmethod
        def _init_and_load_from(cls: type[TorchNNModule], path: StrPath | Path, strict: bool = True) -> TorchNNModule:
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
