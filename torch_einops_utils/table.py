from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any, Callable

import torch

# helpers

def exists(v):
    return v is not None

def to_scalar(v: Any) -> Any:
    if torch.is_tensor(v) and v.numel() == 1:
        return v.item()

    return v

def format_value(v: Any, fmt: str | Callable = '.6g') -> str:
    v = to_scalar(v)

    if callable(fmt):
        return str(fmt(v))

    if isinstance(v, Real) and not isinstance(v, bool):
        return format(v, fmt)

    return str(v)

def is_numeric(v: Any) -> bool:
    v = to_scalar(v)
    return isinstance(v, Real) and not isinstance(v, bool)

# formatting

def format_table(
    rows: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    /,
    *,
    columns: Sequence[str] | None = None,
    header: bool | None = None,
    sep: str = '  ',
    float_fmt: str = '.6g',
    formatters: Mapping[str, str | Callable] | None = None,
    align: str | Mapping[str, str] = 'auto',
    title: str | None = None,
    **kwargs: Any
) -> str:
    if rows is None:
        rows = [kwargs]
    elif isinstance(rows, Mapping):
        rows = [{**rows, **kwargs}]
    else:
        assert not kwargs, 'extra keyword arguments are only supported for a single row'
        rows = [dict(row) for row in rows]

    if not rows:
        return ''

    if columns is None:
        columns = [*dict.fromkeys(k for row in rows for k in row)]
    else:
        columns = list(columns)

    formatters = formatters or {}

    def format_cell(key: str, v: Any) -> str:
        fmt = formatters.get(key, float_fmt)
        return format_value(v, fmt)

    formatted_rows = [[format_cell(k, row.get(k, '')) for k in columns] for row in rows]

    header_is_auto = not exists(header)
    header = (len(rows) > 1) if header_is_auto else header

    # a single row reads best as `key value` pairs

    if len(rows) == 1 and header_is_auto:
        line = sep.join(f'{k} {cell}' for k, cell in zip(columns, formatted_rows[0]))
        return '\n'.join(part for part in (title, line) if exists(part))

    if header:
        widths = [
            max(len(k), *(len(row[idx]) for row in formatted_rows))
            for idx, k in enumerate(columns)
        ]
    else:
        widths = [
            max(len(row[idx]) for row in formatted_rows)
            for idx in range(len(columns))
        ]

    def resolve_align(key: str, idx: int) -> str:
        if isinstance(align, Mapping):
            return align.get(key, 'left')

        if align != 'auto':
            return align

        return 'right' if all(is_numeric(row.get(key)) for row in rows) else 'left'

    aligns = [resolve_align(k, idx) for idx, k in enumerate(columns)]

    def render_row(cells: Sequence[str], is_header = False) -> str:
        out = []

        for cell, width, alignment in zip(cells, widths, aligns):
            if is_header:
                out.append(cell.ljust(width))
            elif alignment == 'right':
                out.append(cell.rjust(width))
            else:
                out.append(cell.ljust(width))

        return sep.join(out).rstrip()

    lines = []

    if exists(title):
        lines.append(title)

    if header:
        lines.append(render_row(columns, is_header = True))

    lines.extend(render_row(row) for row in formatted_rows)

    return '\n'.join(lines)

def log_table(
    rows: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    /,
    *,
    columns: Sequence[str] | None = None,
    header: bool | None = None,
    sep: str = '  ',
    float_fmt: str = '.6g',
    formatters: Mapping[str, str | Callable] | None = None,
    align: str | Mapping[str, str] = 'auto',
    title: str | None = None,
    flush: bool = True,
    **kwargs: Any
) -> str:
    text = format_table(
        rows,
        columns = columns,
        header = header,
        sep = sep,
        float_fmt = float_fmt,
        formatters = formatters,
        align = align,
        title = title,
        **kwargs
    )

    print(text, flush = flush)

    return text
