import pytest
import torch
from torch import nn

from torch_einops_utils import (
    freeze,
    unfreeze,
    freeze_except,
    trainable_parameters,
    frozen
)

def make_model():
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))

def test_freeze_all():
    model = make_model()

    assert freeze(model) is model
    assert not any(p.requires_grad for p in model.parameters())

    unfreeze(model)
    assert all(p.requires_grad for p in model.parameters())

def test_freeze_by_glob():
    model = make_model()
    freeze(model, '*.bias')
    assert all(p.requires_grad == name.endswith('weight') for name, p in model.named_parameters())

    model = make_model()
    freeze(model, '0.*')
    assert not any(p.requires_grad for p in model[0].parameters())
    assert all(p.requires_grad for p in model[1].parameters())

def test_freeze_by_module():
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 2))

    freeze(model, nn.LayerNorm)
    assert not any(p.requires_grad for p in model[1].parameters())
    assert all(p.requires_grad for p in model[0].parameters())

    model = make_model()
    freeze(model, model[1])
    assert not any(p.requires_grad for p in model[1].parameters())
    assert all(p.requires_grad for p in model[0].parameters())

def test_freeze_strict():
    model = make_model()

    with pytest.raises(ValueError):
        freeze(model, 'missing.*')

    freeze(model, 'missing.*', strict = False)
    assert all(p.requires_grad for p in model.parameters())

    with pytest.raises(TypeError):
        freeze(model, 1)

def test_freeze_except():
    model = make_model()
    freeze_except(model, '1.*')
    assert all(p.requires_grad == name.startswith('1.') for name, p in model.named_parameters())

    model = make_model()
    freeze_except(model)
    assert not any(p.requires_grad for p in model.parameters())

def test_trainable_parameters():
    model = make_model()
    freeze(model, '0.*')

    assert trainable_parameters(model) == [model[1].weight, model[1].bias]

def test_trainable_parameters_selectors():
    model = make_model()
    freeze(model, '0.*')

    assert trainable_parameters(model, '1.*') == [model[1].weight, model[1].bias]
    assert trainable_parameters(model, '0.*') == []
    assert trainable_parameters(model, model[0]) == []

def test_frozen_restores_state():
    model = make_model()
    model[1].requires_grad_(False)

    with frozen(model) as m:
        assert m is model
        assert model.training
        assert not any(p.requires_grad for p in model.parameters())

    assert all(p.requires_grad for p in model[0].parameters())
    assert not any(p.requires_grad for p in model[1].parameters())

def test_frozen_selectors():
    model = make_model()

    with frozen(model, '*.bias'):
        assert all(p.requires_grad == name.endswith('weight') for name, p in model.named_parameters())

    assert all(p.requires_grad for p in model.parameters())

def test_frozen_eval():
    model = make_model()
    model[0].eval()

    with frozen(model, eval = True):
        assert not any(m.training for m in model.modules())

    assert model.training
    assert not model[0].training
    assert model[1].training

def test_frozen_decorator():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)
            self.seen_frozen = False

        @frozen
        def forward(self, x):
            self.seen_frozen = not any(p.requires_grad for p in self.parameters())
            return self.linear(x)

    model = Model()
    model(torch.randn(3, 4))

    assert model.seen_frozen
    assert all(p.requires_grad for p in model.parameters())

def test_frozen_decorator_with_module():
    model = make_model()
    seen = {}

    @frozen(model)
    def fn():
        seen['frozen'] = not any(p.requires_grad for p in model.parameters())

    fn()

    assert seen['frozen']
    assert all(p.requires_grad for p in model.parameters())
