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

# nested save load

@save_load()
class GrandChild(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.param = nn.Parameter(torch.randn(dim))

@save_load()
class Child(nn.Module):
    def __init__(self, grandchild = None, name = "child"):
        super().__init__()
        self.grandchild = grandchild
        self.name = name
        self.param = nn.Parameter(torch.randn(1))

@save_load()
class Parent(nn.Module):
    def __init__(self, child1, child2 = None):
        super().__init__()
        self.child1 = child1
        self.child2 = child2
        self.param = nn.Parameter(torch.randn(1))

@save_load()
class GrandParent(nn.Module):
    def __init__(self, p1, p2):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.param = nn.Parameter(torch.randn(1))

def test_sophisticated_nested_save_load():
    gc = GrandChild(dim = 8)
    c1 = Child(name = "c1")
    c2 = Child(name = "c2")
    c_nest = Child(grandchild = gc, name = "c_nest")
    
    p1 = Parent(child1 = c1, child2 = c2)
    p2 = Parent(child1 = c_nest)
    
    gp = GrandParent(p1 = p1, p2 = p2)
    
    path = Path('sophisticated_test.pt')
    
    # Save
    gp.save(str(path))
    
    # Load
    gp2 = GrandParent.init_and_load(str(path))
    
    # Verify structure
    assert gp2.p1.child1.name == "c1"
    assert gp2.p1.child2.name == "c2"
    assert gp2.p2.child1.name == "c_nest"
    assert gp2.p2.child1.grandchild.dim == 8
    
    # Verify weight parity
    assert torch.allclose(gp.param, gp2.param)
    assert torch.allclose(gp.p1.param, gp2.p1.param)
    assert torch.allclose(gp.p1.child1.param, gp2.p1.child1.param)
    assert torch.allclose(gp.p2.child1.grandchild.param, gp2.p2.child1.grandchild.param)
 
    if path.exists():
        os.remove(path)

def test_save_load_without_parens():

    @save_load
    class SimpleNet(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, 1)

    model = SimpleNet(10)
    model.save('./simple.pt')
    reloaded_model = SimpleNet.init_and_load('./simple.pt')

    assert torch.allclose(reloaded_model.linear.weight, model.linear.weight)
