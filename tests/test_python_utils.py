import torch
from torch import nn

from torch_einops_utils.python_utils import maybe_return

def test_maybe_return():
    class Dummy(nn.Module):
        @maybe_return('memories', 'hiddens')
        def forward(
            self,
            tokens,
            return_memories = False,
            return_hiddens = False
        ):
            # only compute what was asked for

            self.computed = [
                field for field, requested in (('memories', return_memories), ('hiddens', return_hiddens)) if requested
            ]

            memories = tokens + 1 if return_memories else None
            hiddens = tokens + 2 if return_hiddens else None

            return tokens, memories, hiddens

    dummy = Dummy()
    tokens = torch.randn(2, 4, 8)

    # primary only by default

    out = dummy(tokens)
    assert torch.equal(out, tokens)
    assert dummy.computed == []

    # fine grained returns, along with the primary

    out = dummy(tokens, return_memories = True)
    assert torch.equal(out.tokens, tokens)
    assert torch.equal(out.memories, tokens + 1)
    assert dummy.computed == ['memories']

    out = dummy(tokens, return_hiddens = True)
    assert torch.equal(out.hiddens, tokens + 2)
    assert dummy.computed == ['hiddens']

    # all fields

    out = dummy(tokens, return_all = True)
    assert torch.equal(out.memories, tokens + 1)
    assert torch.equal(out.hiddens, tokens + 2)
    assert dummy.computed == ['memories', 'hiddens']

    # plain functions work too

    @maybe_return('hiddens')
    def fn(tokens, return_hiddens = False):
        return tokens, (tokens + 1 if return_hiddens else None)

    out = fn(tokens, return_hiddens = True)
    assert torch.equal(out.tokens, tokens)
    assert torch.equal(out.hiddens, tokens + 1)

def test_maybe_return_advanced():
    # scalar tensor return when not requested
    @maybe_return('loss_breakdown', primary = lambda kw: 'loss' if kw.get('return_loss', True) else 'logits')
    def model(x, return_loss = True, return_loss_breakdown = False):
        if not return_loss:
            return torch.randn(2, 4)
        loss = torch.tensor(1.5)
        if not return_loss_breakdown:
            return loss
        return loss, {'loss_breakdown': [1.0, 0.5]}

    out1 = model(None)
    assert isinstance(out1, torch.Tensor) and out1.item() == 1.5

    out2 = model(None, return_loss_breakdown = True)
    assert out2.loss.item() == 1.5
    assert out2.loss_breakdown == [1.0, 0.5]
    loss, breakdown = out2
    assert loss.item() == 1.5

    out3 = model(None, return_loss = False)
    assert out3.shape == (2, 4)

    out4 = model(None, return_loss = False, return_loss_breakdown = True)
    assert hasattr(out4, 'logits')
