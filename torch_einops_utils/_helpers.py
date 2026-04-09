from __future__ import annotations

def divisible_by(num: float, den: float) -> bool:
    """Test whether `num` is evenly divisible by `den`.

    You can use `divisible_by` to check divisibility without raising a `ZeroDivisionError` when `den`
    is zero. `divisible_by` returns `False` whenever `den` is zero, and otherwise returns `True` when
    `num % den == 0`.

    Parameters
    ----------
    num : float
        The numerator to test.
    den : float
        The denominator. When `den` is `0`, `divisible_by` returns `False` without evaluating `num %
        den`.

    Returns
    -------
    is_divisible : bool
        `True` when `den != 0` and `num % den == 0`, otherwise `False`.
    """
    return (den != 0) and ((num % den) == 0)
