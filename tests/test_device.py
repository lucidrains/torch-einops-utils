import torch
from torch import nn

def test_module_device():
    from torch_einops_utils.device import module_device

    assert module_device(nn.Linear(3, 3)) == torch.device('cpu')
    assert module_device(nn.Identity()) is None

def test_move_input_to_device():
    from torch_einops_utils.device import move_inputs_to_device

    def fn(t):
        return t

    decorated = move_inputs_to_device(torch.device('cpu'))(fn)
    decorated(torch.randn(3))

def test_tree_map_tensor_to_device():
    from torch_einops_utils.device import tree_map_tensor_to_device

    tree = (torch.randn(3), [torch.randn(4), {'a': torch.randn(5)}])
    device = torch.device('meta')

    out_tree = tree_map_tensor_to_device(tree, device)

    assert out_tree[0].device == device
    assert out_tree[1][0].device == device
    assert out_tree[1][1]['a'].device == device

def test_fallback_from():
    from torch_einops_utils.device import fallback_from

    @fallback_from('mps')
    def fn(t):
        assert t.device.type == 'cpu'
        return t * 2

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    t = torch.randn(3, device = device, requires_grad = True)

    out = fn(t)

    assert out.device.type == device

    out.sum().backward()
    assert t.grad is not None

def test_device_dtype():
    from torch_einops_utils import device_dtype as dd

    t = torch.randn(3, dtype = torch.float64)
    assert dd(t) == dict(device = torch.device('cpu'), dtype = torch.float64)
    assert dd(t, dtype = torch.float32) == dict(device = torch.device('cpu'), dtype = torch.float32)

    out = torch.zeros(4, **dd(t))
    assert out.device == t.device and out.dtype == t.dtype

    linear = nn.Linear(3, 3, dtype = torch.float16)
    assert dd(linear) == dict(device = torch.device('cpu'), dtype = torch.float16)
