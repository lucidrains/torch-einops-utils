import os
from pathlib import Path

import torch
from torch import nn
from torch_einops_utils.save_load import save_load

@save_load()
class SimpleNet(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.net = nn.Linear(dim, hidden_dim)

    def forward(self, x):
        return self.net(x)

def test_save_load():
    model = SimpleNet(10, 20)
    path = Path('test_model.pt')
    
    # Save the model
    model.save(str(path))
    
    # Create another model with different weights
    model2 = SimpleNet(10, 20)
    
    # Ensure weights are different initially
    assert not torch.allclose(model.net.weight, model2.net.weight)
    
    # Load back
    model2.load(str(path))
    
    # Validate weights are the same
    assert torch.allclose(model.net.weight, model2.net.weight)
    
    # Cleanup
    if path.exists():
        os.remove(path)

def test_init_and_load():
    dim, hidden_dim = 16, 32
    model = SimpleNet(dim, hidden_dim)
    path = Path('test_model_init.pt')
    
    # Save the model
    model.save(str(path))
    
    # Initialize and load from file
    model2 = SimpleNet.init_and_load(str(path))
    
    # Validate attributes and weights
    assert model2.dim == dim
    assert model2.hidden_dim == hidden_dim
    assert torch.allclose(model.net.weight, model2.net.weight)
    
    # Cleanup
    if path.exists():
        os.remove(path)
