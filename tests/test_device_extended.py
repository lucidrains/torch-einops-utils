from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from torch_einops_utils.device import (
    module_device,
    move_inputs_to_device,
    move_inputs_to_module_device
)

import pytest


@pytest.mark.parametrize(
    ("module_factory", "expected_device", "description"),
    [
        pytest.param(lambda: nn.Linear(3, 5), torch.device("cpu"), "parameter-device", id="parameter-device"),
        pytest.param(nn.Identity, None, "no-parameter-no-buffer", id="no-parameter-no-buffer"),
        pytest.param(
            lambda: _buffer_only_module(torch.device("meta")), torch.device("meta"), "buffer-only-module", id="buffer-only-module"
        ),
        pytest.param(
            lambda: _parameter_and_buffer_module(torch.device("cpu"), torch.device("meta")),
            torch.device("cpu"),
            "parameter-prioritized-before-buffer",
            id="parameter-prioritized-before-buffer",
        ),
    ],
)
def test_module_device_returns_expected_device(
    module_factory: Callable[[], nn.Module], expected_device: torch.device | None, description: str
) -> None:
    module_instance = module_factory()
    result = module_device(module_instance)
    assert result == expected_device, f"module_device returned {result}, expected {expected_device} for {description}."


@pytest.mark.parametrize(
    "target_device",
    [pytest.param(torch.device("meta"), id="target-meta")],
)
def test_move_inputs_to_device_moves_tensor_arguments_in_nested_structures(target_device: torch.device) -> None:
    @move_inputs_to_device(target_device)
    def collect_device_types(
        positional_tensor: Tensor, nested_tuple: tuple[Tensor, str], *, keyword_tensor: Tensor, keyword_dictionary: dict[str, Tensor | str]
    ) -> tuple[torch.device, torch.device, torch.device, str, str]:
        return (positional_tensor.device, nested_tuple[0].device, keyword_tensor.device, nested_tuple[1], str(keyword_dictionary["tag"]))

    cpu_tensor_left = torch.tensor([2.0, 3.0, 5.0])
    cpu_tensor_right = torch.tensor([7.0, 11.0, 13.0])
    cpu_tensor_keyword = torch.tensor([17.0, 19.0, 23.0])

    result = collect_device_types(
        cpu_tensor_left,
        (cpu_tensor_right, "north"),
        keyword_tensor=cpu_tensor_keyword,
        keyword_dictionary={"payload": torch.tensor([29.0, 31.0]), "tag": "east"},
    )

    assert result[0] == target_device, f"move_inputs_to_device did not move positional tensor to {target_device}; received {result[0]}."
    assert result[1] == target_device, (
        f"move_inputs_to_device did not move nested positional tensor to {target_device}; received {result[1]}."
    )
    assert result[2] == target_device, f"move_inputs_to_device did not move keyword tensor to {target_device}; received {result[2]}."
    assert result[3] == "north", f"move_inputs_to_device changed non-tensor positional value to {result[3]}, expected 'north'."
    assert result[4] == "east", f"move_inputs_to_device changed non-tensor keyword value to {result[4]}, expected 'east'."


@pytest.mark.parametrize(
    "module_factory",
    [
        pytest.param(lambda: _echo_module_with_parameter(torch.device("meta")), id="module-with-parameter-meta-device"),
        pytest.param(lambda: _echo_module_with_buffer(torch.device("meta")), id="module-with-buffer-meta-device"),
    ],
)
def test_move_inputs_to_module_device_uses_module_device_when_present(module_factory: Callable[[], nn.Module]) -> None:
    module_instance = module_factory()
    source_tensor = torch.tensor([37.0, 41.0])
    source_keyword_tensor = torch.tensor([43.0, 47.0])

    result_tensor, result_dictionary = module_instance.forward(
        source_tensor, payload={"keyword_tensor": source_keyword_tensor, "label": "south"}
    )

    assert result_tensor.device == torch.device("meta"), (
        f"move_inputs_to_module_device did not move positional tensor to meta; received {result_tensor.device}."
    )
    assert isinstance(result_dictionary["keyword_tensor"], Tensor), (
        "move_inputs_to_module_device changed tensor payload to a non-tensor value."
    )
    keyword_tensor = result_dictionary["keyword_tensor"]
    assert isinstance(keyword_tensor, Tensor)
    assert keyword_tensor.device == torch.device("meta"), (
        f"move_inputs_to_module_device did not move keyword tensor to meta; received {keyword_tensor.device}."
    )
    assert result_dictionary["label"] == "south", (
        f"move_inputs_to_module_device changed non-tensor payload to {result_dictionary['label']}, expected 'south'."
    )


@pytest.mark.parametrize("module_instance", [pytest.param(None, id="module-without-parameter-and-buffer")])
def test_move_inputs_to_module_device_skips_move_when_module_has_no_device(module_instance: None) -> None:
    del module_instance
    module_without_state = _stateless_echo_module()
    source_tensor = torch.tensor([53.0, 59.0])
    result_tensor = module_without_state.forward(source_tensor)

    assert result_tensor.device == source_tensor.device, (
        f"move_inputs_to_module_device moved tensor to {result_tensor.device}, expected {source_tensor.device}."
    )


def _buffer_only_module(buffer_device: torch.device) -> nn.Module:
    class BufferOnlyModule(nn.Module):
        def __init__(self, buffer_device: torch.device) -> None:
            super().__init__()
            self.register_buffer("prime_buffer", torch.ones((2,), device=buffer_device))

    return BufferOnlyModule(buffer_device)


def _parameter_and_buffer_module(parameter_device: torch.device, buffer_device: torch.device) -> nn.Module:
    class ParameterAndBufferModule(nn.Module):
        def __init__(self, parameter_device: torch.device, buffer_device: torch.device) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2).to(parameter_device)
            self.register_buffer("prime_buffer", torch.ones((2,), device=buffer_device))

    return ParameterAndBufferModule(parameter_device, buffer_device)


def _echo_module_with_parameter(parameter_device: torch.device) -> nn.Module:
    class EchoModuleWithParameter(nn.Module):
        def __init__(self, parameter_device: torch.device) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor([2.0], device=parameter_device))

        @move_inputs_to_module_device
        def forward(self, tensor_value: Tensor, *, payload: dict[str, Tensor | str]) -> tuple[Tensor, dict[str, Tensor | str]]:
            return tensor_value, payload

    return EchoModuleWithParameter(parameter_device)


def _echo_module_with_buffer(buffer_device: torch.device) -> nn.Module:
    class EchoModuleWithBuffer(nn.Module):
        def __init__(self, buffer_device: torch.device) -> None:
            super().__init__()
            self.register_buffer("buffer_anchor", torch.tensor([3.0], device=buffer_device))

        @move_inputs_to_module_device
        def forward(self, tensor_value: Tensor, *, payload: dict[str, Tensor | str]) -> tuple[Tensor, dict[str, Tensor | str]]:
            return tensor_value, payload

    return EchoModuleWithBuffer(buffer_device)


def _stateless_echo_module() -> nn.Module:
    class StatelessEchoModule(nn.Module):
        @move_inputs_to_module_device
        def forward(self, tensor_value: Tensor) -> Tensor:
            return tensor_value

    return StatelessEchoModule()
