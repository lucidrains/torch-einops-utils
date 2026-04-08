"""Inspect and route tensor arguments based on module or explicit device assignment.

You can use this module to determine the device on which a `torch.nn.Module` resides and to decorate
functions or methods so that tensor arguments are automatically moved to the appropriate device
before each call. These utilities remove explicit `.to(device)` calls from each call site and
centralize device routing in a single decorator.

Contents
--------
Functions
    module_device
        Infer the `torch.device` of a module from its first parameter or buffer.
    move_inputs_to_device
        Create a decorator that moves all tensor arguments to a fixed target device.
    move_inputs_to_module_device
        Create a decorator that moves all tensor arguments to the device of the module.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from itertools import chain
from typing import Concatenate

from torch import Tensor, device
from torch.nn import Module
from torch.types import Device

from torch_einops_utils import PSpec, T_co, TypeModule, exists, tree_map_tensor


def module_device(m: Module) -> device | None:
    """Infer the `torch.device` of a module from its first parameter or buffer.

    You can use this function to determine the device of a `torch.nn.Module` [1] without inspecting
    its internals directly. The function chains the module's parameters and buffers [2] and returns
    the device of the first one found. Parameters are checked before buffers. If the module has no
    parameters or buffers, the function returns `None`.

    Parameters
    ----------
    m : Module
        The module whose device is to be inferred.

    Returns
    -------
    inferredDevice : torch.device | None
        The device of the first parameter or buffer in the module, or `None` if the module has no
        parameters or buffers.

    See Also
    --------
    move_inputs_to_module_device : Decorator that uses this function to route tensor inputs to the module device.

    Examples
    --------
    Retrieve the device of a module with parameters [3]:

        ```python
        import torch
        from torch import nn
        from torch_einops_utils.device import module_device

        linear = nn.Linear(3, 5)
        inferredDevice = module_device(linear)
        # inferredDevice == torch.device('cpu')
        ```

    Returns `None` for a module with no parameters or buffers:

        ```python
        empty = nn.Identity()
        inferredDevice = module_device(empty)
        # inferredDevice is None
        ```

    References
    ----------
    [1] torch.nn.Module - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    [2] torch.nn.Module.buffers - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.buffers
    [3] tests.test_device.test_module_device_returns_expected_device

    """

    first_param_or_buffer: Tensor | None = next(chain(m.parameters(), m.buffers()), None)

    if not exists(first_param_or_buffer):
        return None

    return first_param_or_buffer.device


def move_inputs_to_device(device: Device) -> Callable[[Callable[PSpec, T_co]], Callable[PSpec, T_co]]:
    """Create a decorator that moves all tensor arguments to a target device before each call.

    You can use this function to wrap a callable so that every `torch.Tensor` [1] in the positional
    and keyword arguments is moved to `device` before the wrapped callable is invoked. Non-tensor
    arguments are passed through without modification. The function uses `tree_map_tensor` [2] to
    recurse into containers such as tuples and dictionaries, so tensors nested inside those
    containers are moved as well.

    Parameters
    ----------
    device : Device
        The target device to which all tensor arguments are moved. Accepts any value valid for
        `torch.Tensor.to` [3], such as `"cpu"`, `"cuda"`, or a `torch.device` instance.

    Returns
    -------
    decorator : Callable[[Callable[PSpec, TVar]], Callable[PSpec, TVar]]
        A decorator that wraps the given callable, moving its tensor arguments to `device` before
        each call while preserving the original signature.

    See Also
    --------
    move_inputs_to_module_device : Infers the target device from the module itself.

    Examples
    --------
    Wrap a function so all tensor arguments are moved to the `meta` device [4]:

        ```python
        import torch
        from torch_einops_utils.device import move_inputs_to_device

        targetDevice = torch.device("meta")


        @move_inputs_to_device(targetDevice)
        def collectDeviceTypes(
            positionTensor: torch.Tensor,
            nestedTuple: tuple[torch.Tensor, str],
            *,
            keywordTensor: torch.Tensor,
        ) -> tuple[torch.device, torch.device, torch.device]:
            return positionTensor.device, nestedTuple[0].device, keywordTensor.device


        cpuTensor = torch.tensor([1.0, 2.0])
        result = collectDeviceTypes(cpuTensor, (cpuTensor, "north"), keywordTensor=cpuTensor)
        # result[0] == torch.device("meta")
        # result[1] == torch.device("meta")
        # result[2] == torch.device("meta")
        ```

    References
    ----------
    [1] torch.Tensor - PyTorch documentation
        https://pytorch.org/docs/stable/tensors.html
    [2] torch_einops_utils.torch_einops_utils.tree_map_tensor

    [3] torch.Tensor.to - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.Tensor.to.html
    [4] tests.test_device.test_move_inputs_to_device_moves_tensor_arguments_in_nested_structures

    """

    def decorator(fn: Callable[PSpec, T_co]) -> Callable[PSpec, T_co]:
        @wraps(fn)
        def inner(*args: PSpec.args, **kwargs: PSpec.kwargs) -> T_co:
            args, kwargs = tree_map_tensor(lambda t: t.to(device), (args, kwargs))

            return fn(*args, **kwargs)

        return inner

    return decorator


def move_inputs_to_module_device(
    fn: Callable[Concatenate[TypeModule, PSpec], T_co],
) -> Callable[Concatenate[TypeModule, PSpec], T_co]:
    """Create a decorator that moves all tensor arguments to the device of the module.

    You can use this function as a decorator on methods of `torch.nn.Module` subclasses [1] to
    automatically move all tensor arguments to the device of the module. The decorator inspects the
    first argument (`self`) using `module_device` [2] to determine the target device, then uses
    `tree_map_tensor` [3] to move every `torch.Tensor` found in the positional and keyword arguments.
    If the module has no parameters or buffers, the tensor arguments are passed through without
    modification.

    The first argument of the decorated callable must be a `torch.nn.Module` instance.
    `typing.Concatenate` [4] expresses the constraint that the first argument is the module while the
    remaining arguments form `PSpec`.

    Parameters
    ----------
    fn : Callable[Concatenate[TypeModule, PSpec], TVar]
        The callable to wrap. The first positional argument must be a `torch.nn.Module` instance
        whose device is used as the routing target.

    Returns
    -------
    wrappedMethod : Callable[Concatenate[TypeModule, PSpec], TVar]
        The wrapped callable with the same signature as `fn`, where all tensor arguments are moved to
        the module device before each invocation.

    See Also
    --------
    module_device : Infer the device of a module from its first parameter or buffer.
    move_inputs_to_device : Move tensor arguments to an explicit target device.

    Examples
    --------
    Decorate a method so all tensor inputs are moved to the module's device [5]:

        ```python
        import torch
        from torch import nn, Tensor
        from torch_einops_utils.device import move_inputs_to_module_device


        class EchoModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = nn.Parameter(torch.tensor([2.0], device=torch.device("meta")))

            @move_inputs_to_module_device
            def forward(self, tensorValue: Tensor) -> Tensor:
                return tensorValue


        module = EchoModule()
        cpuTensor = torch.tensor([1.0, 2.0])
        result = module.forward(cpuTensor)
        # result.device == torch.device("meta")
        ```

    Decorate a standalone function and attach it to a class as a method [6]:

        ```python
        @move_inputs_to_module_device
        def policy_loss(model, state, old_log_probs, actions, advantages, mask=None): ...


        MetaController.policy_loss = policy_loss
        ```

    References
    ----------
    [1] torch.nn.Module - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    [2] torch_einops_utils.device.module_device

    [3] torch_einops_utils.torch_einops_utils.tree_map_tensor

    [4] typing.Concatenate - Python documentation
        https://docs.python.org/3/library/typing.html#typing.Concatenate
    [5] tests.test_device._echo_module_with_parameter

    [6] metacontroller.metacontroller.policy_loss

    """

    @wraps(fn)
    def inner(self: TypeModule, *args: PSpec.args, **kwargs: PSpec.kwargs) -> T_co:
        device: device | None = module_device(self)

        if exists(device):
            args, kwargs = tree_map_tensor(lambda t: t.to(device), (args, kwargs))

        return fn(self, *args, **kwargs)

    return inner
