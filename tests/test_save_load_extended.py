from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from typing import Protocol, TypeAlias, cast

import torch
from torch import nn
from torch.nn import Module

import pytest

from torch_einops_utils.save_load import dehydrate_config, rehydrate_config, save_load, map_values


ConfigArgsKwargsObject: TypeAlias = tuple[tuple[object, ...], dict[object, object]]


class SaveLoadModelProtocol(Protocol):
    def save(self, path: str, overwrite: bool = True) -> None: ...

    def load(self, path: str, strict: bool = True) -> None: ...


class SaveLoadLinearClassProtocol(Protocol):
    @classmethod
    def init_and_load(cls, path: str, strict: bool = True) -> SaveLoadModelProtocol: ...


class SaveLoadCustomModelProtocol(Protocol):
    width: int
    weight: torch.Tensor
    stored_config: object

    def store(self, path: str, overwrite: bool = True) -> None: ...

    def restore(self, path: str, strict: bool = True) -> None: ...


class SaveLoadCustomClassProtocol(Protocol):
    def __call__(self, width: int) -> SaveLoadCustomModelProtocol: ...

    def create_and_restore(self, path: str, strict: bool = True) -> SaveLoadCustomModelProtocol: ...


class LinearWeightContainerProtocol(Protocol):
    weight: torch.Tensor


class SaveLoadVersionedModelProtocol(SaveLoadModelProtocol, Protocol):
    net: LinearWeightContainerProtocol


class SaveLoadNestedLeafProtocol(Protocol):
    marker: str


class SaveLoadNestedBranchProtocol(Protocol):
    label: str
    gate: torch.Tensor
    leaf: SaveLoadNestedLeafProtocol | None


class SaveLoadNestedRootProtocol(SaveLoadModelProtocol, Protocol):
    primary: SaveLoadNestedBranchProtocol
    secondary: SaveLoadNestedBranchProtocol | None
    output: LinearWeightContainerProtocol


class SaveLoadNestedClassProtocol(Protocol):
    @classmethod
    def init_and_load(cls, path: str, strict: bool = True) -> SaveLoadNestedRootProtocol: ...


@save_load()
class SaveLoadExtendedLinearModel(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(dim, hidden_dim)


@save_load(save_method_name='store', load_method_name='restore', config_instance_var_name='stored_config', init_and_load_classmethod_name='create_and_restore')
class SaveLoadExtendedCustomNamedModel(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.weight = nn.Parameter(torch.randn(width))


@save_load()
class SaveLoadExtendedLeaf(nn.Module):
    def __init__(self, width: int, marker: str) -> None:
        super().__init__()
        self.width = width
        self.marker = marker
        self.projection = nn.Linear(width, width, bias=False)


@save_load()
class SaveLoadExtendedBranch(nn.Module):
    def __init__(self, leaf: SaveLoadExtendedLeaf | None = None, label: str = 'branch') -> None:
        super().__init__()
        self.leaf = leaf
        self.label = label
        self.gate = nn.Parameter(torch.randn(1))


@save_load()
class SaveLoadExtendedRoot(nn.Module):
    def __init__(self, primary: SaveLoadExtendedBranch, secondary: SaveLoadExtendedBranch | None = None) -> None:
        super().__init__()
        self.primary = primary
        self.secondary = secondary
        self.output = nn.Linear(5, 3, bias=False)


@save_load(version='1.3.5')
class SaveLoadExtendedVersionedWriter(nn.Module):
    def __init__(self, dim: int = 5) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Linear(dim, dim, bias=False)


@save_load(version='9.7.1')
class SaveLoadExtendedVersionedReader(nn.Module):
    def __init__(self, dim: int = 5) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Linear(dim, dim, bias=False)


@save_load()
class SaveLoadExtendedStrictWriter(nn.Module):
    def __init__(self, dim: int = 5) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Linear(dim, dim, bias=False)


@save_load()
class SaveLoadExtendedStrictReader(nn.Module):
    def __init__(self, dim: int = 5) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Linear(dim, dim, bias=False)
        self.extra = nn.Parameter(torch.randn(dim))


class SaveLoadExtendedInvalidTarget:
    pass


def _build_save_load_extended_nested_model() -> SaveLoadExtendedRoot:
    leaf_primary = SaveLoadExtendedLeaf(width=5, marker='leaf-primary')
    leaf_secondary = SaveLoadExtendedLeaf(width=5, marker='leaf-secondary')
    branch_primary = SaveLoadExtendedBranch(leaf=leaf_primary, label='branch-primary')
    branch_secondary = SaveLoadExtendedBranch(leaf=leaf_secondary, label='branch-secondary')
    return SaveLoadExtendedRoot(primary=branch_primary, secondary=branch_secondary)


def _build_save_load_extended_config_case(config_case_key: str) -> tuple[ConfigArgsKwargsObject, int]:
    if config_case_key == 'nested-modules':
        nested_model = _build_save_load_extended_nested_model()
        linear_model = SaveLoadExtendedLinearModel(7, 11)
        config: ConfigArgsKwargsObject = ((nested_model,), {'auxiliary': [linear_model, {'token': 'north'}], 'offsets': (13, 21)})
        return config, 2

    if config_case_key == 'plain-values':
        config = ((13, 21), {'token': 'south', 'offsets': [34, 55]})
        return config, 0

    message = f'Unknown save/load config case key: {config_case_key}.'
    raise ValueError(message)


def _collect_dehydrated_module_records(config_value: object) -> list[dict[str, object]]:
    if isinstance(config_value, dict):
        dictionary_value = cast(dict[object, object], config_value)
        records: list[dict[str, object]] = []
        if bool(dictionary_value.get('__save_load_module__', False)):
            records.append(cast(dict[str, object], dictionary_value))

        for nested_value in dictionary_value.values():
            records.extend(_collect_dehydrated_module_records(nested_value))

        return records

    if isinstance(config_value, (list, tuple)):
        sequence_value = cast(list[object] | tuple[object, ...], config_value)
        records = []
        for nested_value in sequence_value:
            records.extend(_collect_dehydrated_module_records(nested_value))

        return records

    return []


def _collect_module_instances(config_value: object) -> list[Module]:
    if isinstance(config_value, Module):
        return [config_value]

    if isinstance(config_value, dict):
        dictionary_value = cast(dict[object, object], config_value)
        instances: list[Module] = []
        for nested_value in dictionary_value.values():
            instances.extend(_collect_module_instances(nested_value))

        return instances

    if isinstance(config_value, (list, tuple)):
        sequence_value = cast(list[object] | tuple[object, ...], config_value)
        instances = []
        for nested_value in sequence_value:
            instances.extend(_collect_module_instances(nested_value))

        return instances

    return []


@pytest.mark.parametrize(
    ('config_case_key', 'expected_min_dehydrated_count'),
    [pytest.param('nested-modules', 2, id='config-case-nested-modules'), pytest.param('plain-values', 0, id='config-case-plain-values')],
)
@pytest.mark.parametrize('config_instance_var_name', [pytest.param('_config', id='config-instance-var-default')])
def test_dehydrate_config_marks_save_load_modules(config_case_key: str, expected_min_dehydrated_count: int, config_instance_var_name: str) -> None:
    config_args_kwargs, expected_count_from_builder = _build_save_load_extended_config_case(config_case_key)

    assert expected_count_from_builder == expected_min_dehydrated_count, (
        f'save/load config builder returned {expected_count_from_builder}, expected {expected_min_dehydrated_count} for {config_case_key=}.'
    )

    dehydrated_config = dehydrate_config(config_args_kwargs, config_instance_var_name)
    dehydrated_records = _collect_dehydrated_module_records(dehydrated_config)
    remaining_module_instances = _collect_module_instances(dehydrated_config)

    if expected_min_dehydrated_count == 0:
        assert len(dehydrated_records) == 0, f'dehydrate_config produced {len(dehydrated_records)} dehydrated records, expected 0 for {config_case_key=}.'
    else:
        assert len(dehydrated_records) >= expected_min_dehydrated_count, (
            f'dehydrate_config produced {len(dehydrated_records)} dehydrated records, expected at least {expected_min_dehydrated_count} for {config_case_key=}.'
        )

    assert len(remaining_module_instances) == 0, f'dehydrate_config left module instances in output: {remaining_module_instances!r}.'

    for record_index, dehydrated_record in enumerate(dehydrated_records):
        assert dehydrated_record.get('__save_load_module__', False) is True, f'dehydrate_config emitted record {record_index} without __save_load_module__ marker: {dehydrated_record!r}.'
        assert 'klass' in dehydrated_record, f'dehydrate_config emitted record {record_index} without klass: {dehydrated_record!r}.'
        assert 'config' in dehydrated_record, f'dehydrate_config emitted record {record_index} without config: {dehydrated_record!r}.'


@pytest.mark.parametrize('config_case_key', [pytest.param('nested-modules', id='config-case-nested-modules'), pytest.param('plain-values', id='config-case-plain-values')])
@pytest.mark.parametrize('config_instance_var_name', [pytest.param('_config', id='config-instance-var-default')])
def test_rehydrate_config_round_trip_matches_dehydrated_payload(config_case_key: str, config_instance_var_name: str) -> None:
    config_args_kwargs, _expected_min_dehydrated_count = _build_save_load_extended_config_case(config_case_key)

    dehydrated_config = dehydrate_config(config_args_kwargs, config_instance_var_name)
    rehydrated_config = rehydrate_config(dehydrated_config)
    redehydrated_config = dehydrate_config(rehydrated_config, config_instance_var_name)

    assert redehydrated_config == dehydrated_config, f'rehydrate_config round trip changed dehydrated payload for {config_case_key=}.'


@pytest.mark.parametrize(
    ('config_instance_var_name', 'expected_dehydrated_count'),
    [pytest.param('_config', 0, id='config-instance-var-default'), pytest.param('stored_config', 1, id='config-instance-var-custom')],
)
@pytest.mark.parametrize('model_width', [pytest.param(13, id='custom-width-13')])
def test_dehydrate_config_respects_config_instance_var_name(config_instance_var_name: str, expected_dehydrated_count: int, model_width: int) -> None:
    custom_model = SaveLoadExtendedCustomNamedModel(model_width)
    config_args_kwargs: ConfigArgsKwargsObject = ((custom_model,), {})

    dehydrated_config = dehydrate_config(config_args_kwargs, config_instance_var_name)
    dehydrated_records = _collect_dehydrated_module_records(dehydrated_config)

    assert len(dehydrated_records) == expected_dehydrated_count, (
        f'dehydrate_config produced {len(dehydrated_records)} dehydrated records, expected {expected_dehydrated_count} for {config_instance_var_name=}.'
    )


@pytest.mark.parametrize(('dim', 'hidden_dim', 'config_tag'), [pytest.param(7, 11, 'manual-dehydrated-config', id='manual-dehydrated-config')])
def test_rehydrate_config_instantiates_manual_dehydrated_modules(dim: int, hidden_dim: int, config_tag: str) -> None:
    config_args_kwargs: ConfigArgsKwargsObject = (({'__save_load_module__': True, 'klass': SaveLoadExtendedLinearModel, 'config': ((dim, hidden_dim), {})},), {'tag': config_tag})

    rehydrated_config = rehydrate_config(config_args_kwargs)
    args, kwargs = rehydrated_config
    instantiated_model = args[0]

    assert isinstance(instantiated_model, SaveLoadExtendedLinearModel), f'rehydrate_config returned {type(instantiated_model).__name__}, expected SaveLoadExtendedLinearModel.'
    assert getattr(instantiated_model, 'dim', None) == dim, f'rehydrate_config produced model dim {getattr(instantiated_model, "dim", None)}, expected {dim}.'
    assert getattr(instantiated_model, 'hidden_dim', None) == hidden_dim, (
        f'rehydrate_config produced model hidden_dim {getattr(instantiated_model, "hidden_dim", None)}, expected {hidden_dim}.'
    )
    assert kwargs.get('tag') == config_tag, f'rehydrate_config changed non-module kwargs: got {kwargs!r}, expected tag {config_tag!r}.'


@pytest.mark.parametrize('invalid_target_class', [pytest.param(SaveLoadExtendedInvalidTarget, id='non-module-save-load-target')])
def test_save_load_rejects_non_module_targets(invalid_target_class: type) -> None:
    with pytest.raises(TypeError) as error_info:
        save_load()(invalid_target_class)

    assert 'subclass of torch.nn.Module' in str(error_info.value), f'save_load raised unexpected error message: {error_info.value!r}.'


@pytest.mark.parametrize(('checkpoint_name', 'overwrite_flag'), [pytest.param('save-load-checkpoint.pt', False, id='overwrite-disabled')])
def test_save_load_save_respects_overwrite_flag(temporary_artifact_path_builder: Callable[[str], Path], checkpoint_name: str, overwrite_flag: bool) -> None:
    linear_model = cast(SaveLoadModelProtocol, SaveLoadExtendedLinearModel(7, 11))
    checkpoint_path = temporary_artifact_path_builder(checkpoint_name)

    linear_model.save(str(checkpoint_path))

    with pytest.raises(FileExistsError) as error_info:
        linear_model.save(str(checkpoint_path), overwrite=overwrite_flag)

    assert 'overwrite' in str(error_info.value), f'save_load.save raised unexpected error message for overwrite guard: {error_info.value!r}.'


@pytest.mark.parametrize('missing_checkpoint_name', [pytest.param('missing-save-load-checkpoint.pt', id='missing-save-load-checkpoint')])
def test_save_load_load_and_init_and_load_raise_for_missing_paths(temporary_artifact_path_builder: Callable[[str], Path], missing_checkpoint_name: str) -> None:
    linear_model = cast(SaveLoadModelProtocol, SaveLoadExtendedLinearModel(7, 11))
    linear_model_class = cast(SaveLoadLinearClassProtocol, SaveLoadExtendedLinearModel)
    missing_checkpoint_path = temporary_artifact_path_builder(missing_checkpoint_name)

    with pytest.raises(FileNotFoundError) as load_error_info:
        linear_model.load(str(missing_checkpoint_path))

    assert 'no file exists at that path' in str(load_error_info.value), f'save_load.load raised unexpected missing-path message: {load_error_info.value!r}.'
    assert missing_checkpoint_path.name in str(load_error_info.value), f'save_load.load missing-path message omitted filename {missing_checkpoint_path.name!r}: {load_error_info.value!r}.'

    with pytest.raises(FileNotFoundError) as init_load_error_info:
        linear_model_class.init_and_load(str(missing_checkpoint_path))

    assert 'no file exists at that path' in str(init_load_error_info.value), f'save_load.init_and_load raised unexpected missing-path message: {init_load_error_info.value!r}.'
    assert missing_checkpoint_path.name in str(init_load_error_info.value), (
        f'save_load.init_and_load missing-path message omitted filename {missing_checkpoint_path.name!r}: {init_load_error_info.value!r}.'
    )


@pytest.mark.parametrize('checkpoint_name', [pytest.param('checkpoint-without-config.pt', id='checkpoint-without-config')])
def test_save_load_init_and_load_requires_config_key(torch_artifact_writer: Callable[[str, object], Path], checkpoint_name: str) -> None:
    linear_model = SaveLoadExtendedLinearModel(7, 11)
    linear_model_class = cast(SaveLoadLinearClassProtocol, SaveLoadExtendedLinearModel)
    checkpoint_path_without_config = torch_artifact_writer(checkpoint_name, {'model': linear_model.state_dict(), 'version': None})

    with pytest.raises(KeyError) as error_info:
        linear_model_class.init_and_load(str(checkpoint_path_without_config))

    assert 'model configs were not found' in str(error_info.value), f'save_load.init_and_load raised unexpected config-missing message: {error_info.value!r}.'


@pytest.mark.parametrize('checkpoint_name', [pytest.param('save-load-custom-methods.pt', id='save-load-custom-methods')])
def test_save_load_supports_custom_method_names_and_config_storage(temporary_artifact_path_builder: Callable[[str], Path], checkpoint_name: str) -> None:
    custom_model = cast(SaveLoadCustomModelProtocol, SaveLoadExtendedCustomNamedModel(13))
    custom_model_class = cast(SaveLoadCustomClassProtocol, SaveLoadExtendedCustomNamedModel)
    checkpoint_path = temporary_artifact_path_builder(checkpoint_name)

    assert hasattr(custom_model, 'store'), "custom save_load model is missing configured save method 'store'."
    assert hasattr(custom_model, 'restore'), "custom save_load model is missing configured load method 'restore'."
    assert hasattr(custom_model_class, 'create_and_restore'), "custom save_load class is missing configured classmethod 'create_and_restore'."
    assert hasattr(custom_model, 'stored_config'), "custom save_load model is missing configured config storage attribute 'stored_config'."

    custom_model.store(str(checkpoint_path))

    loaded_by_classmethod = custom_model_class.create_and_restore(str(checkpoint_path))
    assert loaded_by_classmethod.width == custom_model.width, f'custom save_load classmethod returned width {loaded_by_classmethod.width}, expected {custom_model.width}.'
    assert torch.allclose(loaded_by_classmethod.weight, custom_model.weight), 'custom save_load classmethod failed to restore parameter values from checkpoint.'

    loaded_by_instance_method = custom_model_class(custom_model.width)
    loaded_by_instance_method.restore(str(checkpoint_path))
    assert torch.allclose(loaded_by_instance_method.weight, custom_model.weight), 'custom save_load instance restore failed to load parameter values from checkpoint.'


@pytest.mark.parametrize('checkpoint_name', [pytest.param('save-load-versioned.pt', id='save-load-versioned')])
def test_save_load_version_mismatch_emits_notice_and_loads_state(temporary_artifact_path_builder: Callable[[str], Path], checkpoint_name: str, capsys: pytest.CaptureFixture[str]) -> None:
    writer_model = cast(SaveLoadVersionedModelProtocol, SaveLoadExtendedVersionedWriter(dim=5))
    reader_model = cast(SaveLoadVersionedModelProtocol, SaveLoadExtendedVersionedReader(dim=5))
    checkpoint_path = temporary_artifact_path_builder(checkpoint_name)

    writer_model.save(str(checkpoint_path))

    reader_model.load(str(checkpoint_path))
    captured_output = capsys.readouterr()

    assert 'loading saved model at version' in captured_output.out, f'save_load did not emit version mismatch notice. Captured output: {captured_output.out!r}.'
    assert '1.3.5' in captured_output.out, f'save_load version mismatch notice omitted saved version. Captured output: {captured_output.out!r}.'
    assert '9.7.1' in captured_output.out, f'save_load version mismatch notice omitted current version. Captured output: {captured_output.out!r}.'
    assert torch.allclose(writer_model.net.weight, reader_model.net.weight), 'save_load failed to load state_dict values after version mismatch notice.'


@pytest.mark.parametrize('checkpoint_name', [pytest.param('save-load-strict.pt', id='save-load-strict')])
def test_save_load_strict_flag_controls_missing_key_behavior(temporary_artifact_path_builder: Callable[[str], Path], checkpoint_name: str) -> None:
    writer_model = cast(SaveLoadVersionedModelProtocol, SaveLoadExtendedStrictWriter(dim=5))
    reader_model = cast(SaveLoadVersionedModelProtocol, SaveLoadExtendedStrictReader(dim=5))
    checkpoint_path = temporary_artifact_path_builder(checkpoint_name)

    writer_model.save(str(checkpoint_path))

    with pytest.raises(RuntimeError) as strict_error_info:
        reader_model.load(str(checkpoint_path), strict=True)

    assert 'Missing key(s) in state_dict' in str(strict_error_info.value), f'save_load strict=True raised unexpected error message: {strict_error_info.value!r}.'

    reader_model.load(str(checkpoint_path), strict=False)
    assert torch.allclose(writer_model.net.weight, reader_model.net.weight), 'save_load strict=False failed to load shared keys from state_dict.'


@pytest.mark.parametrize('checkpoint_name', [pytest.param('save-load-nested.pt', id='save-load-nested')])
def test_save_load_init_and_load_rehydrates_nested_modules(temporary_artifact_path_builder: Callable[[str], Path], checkpoint_name: str) -> None:
    nested_model = cast(SaveLoadNestedRootProtocol, _build_save_load_extended_nested_model())
    nested_model_class = cast(SaveLoadNestedClassProtocol, SaveLoadExtendedRoot)
    checkpoint_path = temporary_artifact_path_builder(checkpoint_name)

    nested_model.save(str(checkpoint_path))
    restored_model = nested_model_class.init_and_load(str(checkpoint_path))

    assert restored_model.primary.label == nested_model.primary.label, f'nested save_load changed primary label from {nested_model.primary.label!r} to {restored_model.primary.label!r}.'
    assert restored_model.secondary is not None, 'nested save_load restored model with missing secondary branch.'
    assert nested_model.secondary is not None, 'nested save_load test fixture unexpectedly created source model without secondary branch.'
    assert restored_model.primary.leaf is not None, 'nested save_load restored model with missing primary leaf module.'
    assert restored_model.secondary.leaf is not None, 'nested save_load restored model with missing secondary leaf module.'
    assert nested_model.primary.leaf is not None, 'nested save_load test fixture unexpectedly created source model without primary leaf module.'
    assert nested_model.secondary.leaf is not None, 'nested save_load test fixture unexpectedly created source model without secondary leaf module.'
    assert restored_model.primary.leaf.marker == nested_model.primary.leaf.marker, (
        f'nested save_load changed primary leaf marker from {nested_model.primary.leaf.marker!r} to {restored_model.primary.leaf.marker!r}.'
    )
    assert restored_model.secondary.leaf.marker == nested_model.secondary.leaf.marker, (
        f'nested save_load changed secondary leaf marker from {nested_model.secondary.leaf.marker!r} to {restored_model.secondary.leaf.marker!r}.'
    )
    assert torch.allclose(restored_model.output.weight, nested_model.output.weight), 'nested save_load failed to restore root module state_dict values.'
    assert torch.allclose(restored_model.primary.gate, nested_model.primary.gate), 'nested save_load failed to restore primary branch parameter values.'
    assert torch.allclose(restored_model.secondary.gate, nested_model.secondary.gate), 'nested save_load failed to restore secondary branch parameter values.'



@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        pytest.param(13, 26, id="leaf-int"),
        pytest.param([2, 3, 5], [4, 6, 10], id="flat-list"),
        pytest.param((7, 11, 13), (14, 22, 26), id="flat-tuple"),
        pytest.param({"north": 3, "east": 5}, {"north": 6, "east": 10}, id="flat-dict"),
        pytest.param(
            {"alpha": [2, 3], "beta": (5, 7)},
            {"alpha": [4, 6], "beta": (10, 14)},
            id="nested-dict-with-sequences",
        ),
        pytest.param([[2, 3], [5, 7]], [[4, 6], [10, 14]], id="nested-lists"),
        pytest.param(
            {"outer": {"inner": 11, "sibling": 13}},
            {"outer": {"inner": 22, "sibling": 26}},
            id="nested-dicts",
        ),
    ],
)
def test_map_values_transforms_structure(input_value: object, expected: object) -> None:
    def double_if_int(value: object) -> object:
        if isinstance(value, int):
            return value * 2
        return value

    result = map_values(double_if_int, input_value)  # type: ignore[arg-type]
    assert result == expected, (
        f"map_values returned {result!r}, expected {expected!r} for {input_value=}."
    )
