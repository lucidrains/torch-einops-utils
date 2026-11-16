## torch-einops-utils

Some utility functions to help myself (and perhaps others) go faster with ML/AI work

## maybe_return

return just the primary output by default, extras only on request

```python
from torch_einops_utils import maybe_return

@maybe_return('memories')
def forward(self, ids, return_memories = False):
    memories = build_memories(ids) if return_memories else None

    return backbone(ids), memories

model(ids)                             # logits
model(ids, return_memories = True)     # Output(logits, memories)
logits, memories = model(ids, return_memories = True)  # unpacks
```

any number of extras, individually or all at once

```python
@maybe_return('memories', 'hiddens')
def forward(self, ids, return_memories = False, return_hiddens = False):
    ...
    return logits, memories, hiddens

model(ids, return_hiddens = True)  # Output(logits, hiddens)
model(ids, return_all = True)      # Output(logits, memories, hiddens)
```

`primary` may be a tuple of names, or a callable receiving the bound call arguments, for when the shape of the primary depends on the mode

```python
def primary_fields(kwargs):
    if not kwargs.get('return_loss', False):
        return ('logits', 'memory') if kwargs.get('return_memory', False) else 'logits'

    return ('loss', 'loss_mask') if not kwargs.get('reduce_loss', False) else 'loss'

@maybe_return('pooled_repr', primary = primary_fields)
def forward(self, ids, return_loss = False, return_memory = False, reduce_loss = False, return_pooled_repr = False):
    ...
    if return_loss:
        return loss, loss_mask, pooled_repr

    return logits, memory, pooled_repr

model(ids)                            # logits
logits, memory = model(ids, return_memory = True)
loss, loss_mask = model(ids, return_loss = True)
loss = model(ids, return_loss = True, reduce_loss = True)
loss, pooled = model(ids, return_loss = True, reduce_loss = True, return_pooled_repr = True)
```

## log_table

clean, aligned logging for a single record or a list of records

```python
from torch_einops_utils import log_table

log_table(ep = 200, reward = 187.04, steps = 1000, avg100 = 66.03)
# ep 200  reward 187.04  steps 1000  avg100 66.03

log_table([
    dict(ep = 200, reward = 187.04, avg100 = 66.03),
    dict(ep = 225, reward = 76.72, avg100 = 89.98),
], formatters = dict(reward = '.2f', avg100 = '.2f'))
```

```text
ep   reward  avg100
200  187.04   66.03
225   76.72   89.98
```

`formatters` takes a format spec or a callable per column, and `format_table` returns the string without printing

## selectors

`freeze` and `grad_scope` share one selector syntax, also available directly

```python
from torch_einops_utils import resolve_parameters, resolve_modules

resolve_parameters(model)                        # all
resolve_parameters(model, '*.bias')              # glob over parameter names
resolve_parameters(model, 'encoder.*')
resolve_parameters(model, nn.LayerNorm)          # module type
resolve_parameters(model, model.encoder)         # module instance
resolve_parameters(model, re.compile('head'))    # regex search
resolve_parameters(model, lambda name, p: p.ndim == 1)  # predicate

resolve_modules(model, nn.Linear)                # (name, module) pairs
```

selectors can be nested in lists, and `strict = False` skips selectors that match nothing

## freeze

freeze and unfreeze parts of a network by any of the selectors above

```python
from torch_einops_utils import freeze, unfreeze, freeze_except, trainable_parameters, frozen

freeze(model)                       # all params
freeze(model, '*.bias')             # glob over param names
freeze(model, 'encoder.*')
freeze(model, nn.LayerNorm)         # by module type
freeze(model, model.encoder)        # by module instance

unfreeze(model, 'encoder.*')
freeze_except(model, 'head.*')      # freeze all but the head

optimizer = AdamW(trainable_parameters(model), lr = 1e-4)

with frozen(model, 'encoder.*'):
    ...                             # frozen only for this block, restored after

with frozen(model, 'encoder.*', eval = True):
    ...                             # also eval, training flags restored on exit
```

`frozen` also works as a decorator, on a method or with an explicit module

```python
@frozen
def forward(self, x): ...

@frozen(model)
def evaluate(): ...
```

`grad_scope` accepts the same selectors for `disable`, `enable`, and `hard_set`

```python
from torch_einops_utils import grad_scope

with grad_scope(model, disable = 'encoder.*'):
    ...
```

note: `Frozen` in the grad utilities is different, it deepcopies a module into a non-training snapshot
