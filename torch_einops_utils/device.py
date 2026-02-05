from itertools import chain

import torch
from torch.nn import Module

# helpers

def exists(v):
    return v is not None

# infer the device for a module

def module_device(m: Module):

    first_param_or_buffer = next(chain(m.parameters(), m.buffers()), None)

    if not exists(first_param_or_buffer):
        return None

    return first_param_or_buffer.device
