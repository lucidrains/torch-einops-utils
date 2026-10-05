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
