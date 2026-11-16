import re

import pytest
import torch
from torch import nn

from torch_einops_utils import resolve_modules, resolve_parameters


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8))
        self.head = nn.Linear(8, 2)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.head(self.encoder(x))

def names(pairs):
    return [name for name, _ in pairs]

def test_resolve_parameters_all():
    model = Model()
    assert names(resolve_parameters(model)) == names(model.named_parameters())

def test_resolve_parameters_glob():
    model = Model()

    assert names(resolve_parameters(model, '*.bias')) == [
        'encoder.0.bias',
        'encoder.1.bias',
        'head.bias'
    ]

    assert names(resolve_parameters(model, 'encoder.*')) == [
        'encoder.0.weight',
        'encoder.0.bias',
        'encoder.1.weight',
        'encoder.1.bias'
    ]

def test_resolve_parameters_module_type():
    model = Model()
    assert names(resolve_parameters(model, nn.LayerNorm)) == [
        'encoder.1.weight',
        'encoder.1.bias'
    ]

def test_resolve_parameters_module_instance():
    model = Model()

    assert names(resolve_parameters(model, model.encoder[0])) == [
        'encoder.0.weight',
        'encoder.0.bias'
    ]

    assert names(resolve_parameters(model, model)) == names(model.named_parameters())

def test_resolve_parameters_parameter():
    model = Model()
    assert names(resolve_parameters(model, model.head.weight)) == ['head.weight']

def test_resolve_parameters_regex():
    model = Model()
    assert names(resolve_parameters(model, re.compile(r'^head\.'))) == ['head.weight', 'head.bias']

def test_resolve_parameters_predicate():
    model = Model()

    assert names(resolve_parameters(model, lambda name, param: param.ndim == 1)) == [
        'scale',
        'encoder.0.bias',
        'encoder.1.weight',
        'encoder.1.bias',
        'head.bias'
    ]

def test_resolve_parameters_flattens_and_dedupes():
    model = Model()

    assert names(resolve_parameters(model, ['*.bias', nn.LayerNorm])) == [
        'encoder.0.bias',
        'encoder.1.weight',
        'encoder.1.bias',
        'head.bias'
    ]

def test_resolve_parameters_strict():
    model = Model()

    with pytest.raises(ValueError):
        resolve_parameters(model, 'missing.*')

    assert resolve_parameters(model, 'missing.*', strict = False) == []

    with pytest.raises(TypeError):
        resolve_parameters(model, 1)

def test_resolve_parameters_default_all():
    model = Model()

    assert names(resolve_parameters(model, [])) == names(model.named_parameters())
    assert resolve_parameters(model, [], default_all = False) == []

def test_resolve_modules():
    model = Model()

    assert names(resolve_modules(model, 'encoder.*')) == ['encoder.0', 'encoder.1']
    assert names(resolve_modules(model, nn.Linear)) == ['encoder.0', 'head']
    assert names(resolve_modules(model, model.encoder[1])) == ['encoder.1']
    assert names(resolve_modules(model, re.compile('head$'))) == ['head']
    assert names(resolve_modules(model, lambda name, m: isinstance(m, nn.LayerNorm))) == ['encoder.1']

def test_resolve_modules_strict():
    model = Model()

    with pytest.raises(ValueError):
        resolve_modules(model, 'missing.*')

    assert resolve_modules(model, 'missing.*', strict = False) == []

    with pytest.raises(TypeError):
        resolve_modules(model, 1)
