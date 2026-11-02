from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.func import functional_call, vjp, vmap
from torch.nn import Module
from torch.utils._pytree import tree_flatten, tree_unflatten

from torch_einops_utils.torch_einops_utils import default, exists

# straight through estimator

def straight_through(src, tgt):
    # forward -> tgt, gradient -> src
    return src + (tgt - src).detach()

def zero_with_grad(tgt):
    # forward -> 0, gradient -> tgt
    return tgt - tgt.detach()

def mask_grad(t, mask):
    # forward -> t, gradient -> mask * gradient
    # bool mask drops grads, float mask scales them
    masked = t * mask
    return t.detach() + masked - masked.detach()

# frozen subnetwork
# deepcopy once so the live module can keep training independently
# params are frozen, but gradients still flow to inputs

class Frozen(Module):
    def __init__(self, module):
        super().__init__()
        self.module = deepcopy(module)
        self.module.requires_grad_(False)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

# functional form
# frozen_head = frozen_grad(head)
# out = frozen_head(x)

def frozen_grad(module):
    return Frozen(module)

# per sample gradients
# callback(per_sample_grads, ctx) -> dict[name, grad]
#   per_sample_grads -> name -> (batch, *param_shape)
#   grad -> name -> param_shape
# missing or None -> sum per sample grads
# extra args are constants

@dataclass
class GradContext:
    module: Module
    params: dict[str, Tensor]
    inputs: tuple
    kwargs: dict
    outputs: Any
    batch_size: int
    batch_dim: int

def sum_per_sample_grads(per_sample_grads, ctx = None):
    return {name: grad.sum(dim = 0) for name, grad in per_sample_grads.items()}

# custom autograd function so the transformed per sample gradients
# become the layer's parameter gradients

class _PerSampleGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, meta, *flat):
        module, callback, batch_dim, trainable_names, frozen, buffers, args, kwargs = meta

        num_trainable = len(trainable_names)
        trainable = dict(zip(trainable_names, flat[:num_trainable]))
        x = flat[num_trainable]

        batch = x.shape[batch_dim]

        # vmap over trainable params and x -> per sample forward and backward

        def fn(trainable, x):
            params = {**trainable, **frozen}
            return functional_call(module, (params, buffers), (x, *args), kwargs)

        batched_params = {
            name: param.unsqueeze(0).repeat(batch, *([1] * param.ndim))
            for name, param in trainable.items()
        }

        in_dims = ({name: 0 for name in trainable_names}, batch_dim)
        output, vjp_fn = vjp(vmap(fn, in_dims = in_dims, randomness = 'different'), batched_params, x)

        flat_output, output_spec = tree_flatten(output)

        if not all(torch.is_tensor(t) for t in flat_output):
            raise TypeError('per sample grad expects the module to return tensors')

        flat_output = [t.movedim(0, batch_dim) for t in flat_output]

        ctx.vjp_fn = vjp_fn
        ctx.callback = callback
        ctx.trainable_names = trainable_names
        ctx.num_outputs = len(flat_output)
        ctx.output_spec = output_spec
        ctx.output_meta = tuple((t.shape, t.dtype, t.device) for t in flat_output)
        ctx.param_shapes = {name: tuple(param.shape) for name, param in trainable.items()}
        ctx.batch_dim = batch_dim
        ctx.batch_size = batch
        ctx.params = trainable
        ctx.inputs = (x, *args)
        ctx.kwargs = kwargs
        ctx.outputs = tree_unflatten(flat_output, output_spec)
        ctx.module = module

        return (*flat_output, output_spec)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grads = []

        for grad, (shape, dtype, device) in zip(grad_outputs[:ctx.num_outputs], ctx.output_meta):
            if not exists(grad):
                grad = torch.zeros(shape, dtype = dtype, device = device)

            grads.append(grad.movedim(ctx.batch_dim, 0))

        per_sample_params, dinput = ctx.vjp_fn(tree_unflatten(grads, ctx.output_spec))

        context = GradContext(
            module = ctx.module,
            params = ctx.params,
            inputs = ctx.inputs,
            kwargs = ctx.kwargs,
            outputs = ctx.outputs,
            batch_size = ctx.batch_size,
            batch_dim = ctx.batch_dim
        )

        transformed = default(ctx.callback(per_sample_params, context), {})

        if not isinstance(transformed, Mapping):
            raise TypeError(f'callback must return a mapping, got {type(transformed)}')

        grad_params = []

        for name in ctx.trainable_names:
            if name in transformed:
                grad = transformed[name]
            else:
                per_sample = per_sample_params[name]
                grad = per_sample.sum(dim = 0) if exists(per_sample) else None

            if exists(grad) and not torch.is_tensor(grad):
                raise TypeError(f'callback must return a tensor or None for {name}, got {type(grad)}')

            if exists(grad) and tuple(grad.shape) != ctx.param_shapes[name]:
                raise ValueError(f'callback must return shape {ctx.param_shapes[name]} for {name}, got {tuple(grad.shape)}')

            grad_params.append(grad)

        return (None, *grad_params, dinput)

class PerSampleGrad(Module):
    def __init__(
        self,
        module: Module,
        callback: Callable | None = None,
        *,
        dim: int = 0
    ):
        super().__init__()
        self.module = module
        self.callback = default(callback, sum_per_sample_grads)
        self.dim = dim

    def forward(self, x, *args, **kwargs):
        if not torch.is_grad_enabled():
            return self.module(x, *args, **kwargs)

        params = dict(self.module.named_parameters())
        trainable = {name: param for name, param in params.items() if param.requires_grad}

        if not trainable:
            return self.module(x, *args, **kwargs)

        if not (torch.is_tensor(x) and (x.is_floating_point() or x.is_complex())):
            raise TypeError('per sample grad expects the first input to be a floating point tensor')

        dim = self.dim
        if dim < 0:
            dim += x.ndim

        if not (0 <= dim < x.ndim):
            raise ValueError(f'dim {self.dim} out of range for input of shape {tuple(x.shape)}')

        if x.shape[dim] == 0:
            return self.module(x, *args, **kwargs)

        frozen = {name: param for name, param in params.items() if not param.requires_grad}
        buffers = dict(self.module.named_buffers())

        meta = (
            self.module,
            self.callback,
            dim,
            tuple(trainable),
            frozen,
            buffers,
            args,
            kwargs
        )

        out = _PerSampleGradFunction.apply(meta, *trainable.values(), x)

        *flat_output, output_spec = out
        return tree_unflatten(flat_output, output_spec)

def per_sample_grad(
    module: Module,
    callback: Callable | None = None,
    *,
    dim: int = 0
):
    return PerSampleGrad(module, callback, dim = dim)

# named grads -> reconcile -> set grads

def named_grads(module: Module, *, remove_none = True):
    grads = {name: param.grad for name, param in module.named_parameters()}

    if not remove_none:
        return grads

    return {name: grad for name, grad in grads.items() if exists(grad)}

def named_grads_from_loss(
    loss: Tensor,
    module: Module,
    *,
    retain_graph = False,
    create_graph = False,
    allow_unused = True
):
    params = {name: param for name, param in module.named_parameters() if param.requires_grad}

    if not params:
        return {}

    grads = torch.autograd.grad(
        loss,
        tuple(params.values()),
        retain_graph = retain_graph,
        create_graph = create_graph,
        allow_unused = allow_unused
    )

    return {name: grad for name, grad in zip(params, grads) if exists(grad)}

def set_named_grads(module: Module, grads: dict[str, Tensor]):
    for name, param in module.named_parameters():
        param.grad = grads.get(name)

# scope requires_grad by name
# wrap forward + backward, requires_grad is read when the graph is built
# freezing weights does not stop grads to earlier layers -> detach for that
# does not clear .grad -> call optimizer.zero_grad() first

@contextmanager
def grad_scope(
    module: Module,
    *,
    disable = (),
    enable = (),
    hard_set: dict | None = None,
    strict = True
):
    params = dict(module.named_parameters())
    original = {name: param.requires_grad for name, param in params.items()}

    def set_grad(names, value):
        names = (names,) if isinstance(names, str) else names

        for name in names:
            if name not in params:
                if strict:
                    raise KeyError(name)
                continue

            params[name].requires_grad_(bool(value))

    try:
        set_grad(disable, False)
        set_grad(enable, True)

        for names, value in default(hard_set, {}).items():
            set_grad(names, value)

        yield
    finally:
        for name, value in original.items():
            params[name].requires_grad_(value)
