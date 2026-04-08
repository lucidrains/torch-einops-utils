from __future__ import annotations

import pickle
from functools import wraps
from pathlib import Path

import torch
from torch.nn import Module

from torch_einops_utils import exists

from packaging import version as packaging_version


def map_values(fn, v):
    if isinstance(v, (list, tuple)):
        return type(v)(map_values(fn, el) for el in v)

    if isinstance(v, dict):
        v = {key: map_values(fn, val) for key, val in v.items()}

    return fn(v)


def dehydrate_config(config, config_instance_var_name):
    def dehydrate(v):
        # if the value is a save_load decorated module, convert it to its reconstruction metadata
        if isinstance(v, Module) and hasattr(v, config_instance_var_name):
            return dict(  # noqa: C408
                __save_load_module__=True,
                klass=v.__class__,
                config=dehydrate_config(getattr(v, config_instance_var_name), config_instance_var_name),
            )

        return v

    return map_values(dehydrate, config)


def rehydrate_config(config):
    def rehydrate(v):
        # if the value is reconstruction metadata, instantiate the module using its class and configuration
        if isinstance(v, dict) and v.get("__save_load_module__", False):
            klass = v["klass"]
            args, kwargs = v["config"]
            return klass(*args, **kwargs)

        return v

    return map_values(rehydrate, config)


def save_load(
    maybe_fn=None,
    *,
    save_method_name: str = "save",
    load_method_name: str = "load",
    config_instance_var_name: str = "_config",
    init_and_load_classmethod_name: str = "init_and_load",
    version: str | None = None,
):
    def _save_load(klass):
        if not issubclass(klass, Module):
            message = "save_load should decorate a subclass of torch.nn.Module"
            raise TypeError(message)

        _orig_init = klass.__init__

        @wraps(_orig_init)
        def __init__(self, *args, **kwargs):
            setattr(self, config_instance_var_name, (args, kwargs))
            _orig_init(self, *args, **kwargs)

        def _save(self, path, overwrite: bool = True) -> None:
            path = Path(path)
            if not overwrite and path.exists():
                message: str = f"I received `{path = }`, but this path already exists and `overwrite` is `False`."
                raise FileExistsError(message)

            config = getattr(self, config_instance_var_name)
            pkg = dict(  # noqa: C408
                model=self.state_dict(),
                config=pickle.dumps(dehydrate_config(config, config_instance_var_name)),
                version=version,
            )

            torch.save(pkg, str(path))

        def _load(self, path, strict: bool = True) -> None:
            path = Path(path)
            if not path.exists():
                message: str = f"I could not find a file at `{path = }`."
                raise FileNotFoundError(message)

            pkg = torch.load(str(path), map_location="cpu")

            if exists(version) and exists(pkg["version"]) and packaging_version.parse(version) != packaging_version.parse(pkg["version"]):
                print(f"loading saved model at version {pkg['version']}, but current package version is {version}")  # noqa: T201

            self.load_state_dict(pkg["model"], strict=strict)

        # init and load from
        # looks for a `config` key in the stored checkpoint, instantiating the model as well as loading the state dict

        @classmethod
        def _init_and_load_from(cls, path, strict: bool = True):
            path = Path(path)
            if not path.exists():
                message: str = f"I could not find a file at `{path = }`."
                raise FileNotFoundError(message)
            pkg = torch.load(str(path), map_location="cpu")

            if "config" not in pkg:
                message = "model configs were not found in this saved checkpoint"
                raise ValueError(message)

            config = pickle.loads(pkg["config"])
            args, kwargs = rehydrate_config(config)
            model = cls(*args, **kwargs)

            _load(model, path, strict=strict)
            return model

        # set decorated init as well as save, load, and init_and_load

        klass.__init__ = __init__
        setattr(klass, save_method_name, _save)
        setattr(klass, load_method_name, _load)
        setattr(klass, init_and_load_classmethod_name, _init_and_load_from)

        return klass

    # if already decorating a function then just return

    if exists(maybe_fn):
        return _save_load(maybe_fn)

    return _save_load
