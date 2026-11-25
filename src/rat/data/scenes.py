"""Scenes of the controlled pair generator (Appendix B.1).

A scene holds structured content, a rendering specification and a question whose answer is
computed from the content.  An edit changes one answer-determining variable; the answer and
every dependent visual element are recomputed from the edited content, while the question,
the geometry and all rendering choices stay fixed within the pair.

Three task families are generated: bar charts, numerical tables and text cards.  Numbers are
stored as integers in units of ``10 ** -decimals`` so that answers are exact strings.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from typing import Any

from . import vocab
from .fonts import FAMILIAR_FONTS, HELDOUT_FONTS
from .rng import Rng

FAMILIES: tuple[str, ...] = ("chart", "table", "card")
RENDERINGS: tuple[str, ...] = ("familiar", "heldout")

#: Layout variants per task family; the held-out variants appear only in the shifted half of
#: the paired test sets.
LAYOUTS: dict[str, dict[str, tuple[str, ...]]] = {
    "chart": {"familiar": ("vertical", "vertical_grid"), "heldout": ("horizontal",)},
    "table": {"familiar": ("ruled", "zebra"), "heldout": ("boxed",)},
    "card": {"familiar": ("two_column", "stacked"), "heldout": ("inline_boxed",)},
}

#: Number formats ``(low, high, decimals)`` in stored integer units.
NUMBER_RANGES: dict[str, tuple[tuple[int, int, int], ...]] = {
    "familiar": ((10, 99, 0), (100, 999, 0), (10, 999, 1)),
    "heldout": ((1000, 9999, 0), (100, 9999, 2)),
}

#: Digits of identifiers on text cards.
ID_DIGITS: dict[str, tuple[int, int]] = {"familiar": (4, 5), "heldout": (6, 8)}

CANVAS_LONG_SIDES: tuple[int, ...] = (560, 784, 1008, 1232, 1456, 1680, 1904)
ASPECTS: dict[str, tuple[tuple[int, int], ...]] = {
    "chart": ((4, 3), (3, 2), (16, 10)),
    "table": ((4, 3), (3, 2), (1, 1)),
    "card": ((4, 3), (1, 1), (3, 4)),
}
#: Body text size in native pixels, drawn log-uniformly.  Together with the canvas sizes this
#: spreads the projected edit scale from details that vanish in the low-resolution mode to
#: content that is legible in both modes.
FONT_PX_RANGE: tuple[int, int] = (7, 30)

BACKGROUNDS: tuple[tuple[int, int, int], ...] = (
    (255, 255, 255),
    (250, 250, 247),
    (244, 246, 250),
    (252, 248, 240),
    (245, 249, 245),
)
INKS: tuple[tuple[int, int, int], ...] = ((20, 20, 24), (34, 40, 49), (45, 35, 30))
ACCENTS: tuple[tuple[int, int, int], ...] = (
    (38, 110, 160),
    (190, 80, 50),
    (60, 140, 110),
    (120, 90, 170),
    (200, 140, 40),
)

#: Probability that an edit is fine-grained (one digit, one character, a small mark change).
FINE_EDIT_PROBABILITY = 0.6

_STREAM_SPEC, _STREAM_CONTENT, _STREAM_EDIT = 1, 2, 3


@dataclass(frozen=True)
class RenderSpec:
    """Rendering choices; sampled independently of the answer and shared within a pair."""

    width: int
    height: int
    font: str
    font_px: int
    layout: str
    background: tuple[int, int, int]
    ink: tuple[int, int, int]
    accent: tuple[int, int, int]
    rendering: str


@dataclass(frozen=True)
class Scene:
    """One image of a pair: content, rendering specification and question."""

    family: str
    seed: int
    spec: RenderSpec
    content: dict[str, Any]
    question: str
    target: dict[str, Any]

    @property
    def answer(self) -> str:
        return compute_answer(self.family, self.content, self.target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "seed": self.seed,
            "spec": asdict(self.spec),
            "content": self.content,
            "question": self.question,
            "target": self.target,
        }


# --------------------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------------------


def format_number(value: int, decimals: int) -> str:
    """Render a stored integer with ``decimals`` fractional digits, e.g. ``(1234, 2) -> 12.34``."""
    if decimals == 0:
        return str(value)
    sign = "-" if value < 0 else ""
    whole, frac = divmod(abs(value), 10**decimals)
    return f"{sign}{whole}.{frac:0{decimals}d}"


def nice_ceiling(value: int) -> int:
    """Smallest of 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10 times a power of ten that is >= value."""
    if value <= 0:
        raise ValueError("value must be positive")
    if value <= 10:
        return 10
    magnitude = 10 ** (len(str(value)) - 1)
    for step in (10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100):
        candidate = step * magnitude // 10
        if candidate >= value:
            return candidate
    raise AssertionError("unreachable")


def edit_number(rng: Rng, value: int, low: int, high: int, fine: bool) -> int:
    """Return a different number in ``[low, high]``.

    A fine edit replaces exactly one digit and keeps the number of digits; a coarse edit
    redraws the number until at least two digits differ (or the length changes).
    """
    if high <= low:
        raise ValueError("edit range must contain at least two numbers")
    digits = str(value)
    if fine:
        for _ in range(64):
            position = rng.randint(0, len(digits) - 1)
            options = [d for d in vocab.DIGITS if d != digits[position]]
            if position == 0 and len(digits) > 1:
                options = [d for d in options if d != "0"]
            candidate = int(digits[:position] + rng.choice(options) + digits[position + 1 :])
            if low <= candidate <= high and candidate != value:
                return candidate
    for _ in range(256):
        candidate = rng.randint(low, high)
        other = str(candidate)
        differing = (
            len(other) != len(digits)
            or sum(a != b for a, b in zip(other, digits, strict=True)) >= 2
        )
        if candidate != value and differing:
            return candidate
    return low if value != low else high


# --------------------------------------------------------------------------------------
# Rendering specification
# --------------------------------------------------------------------------------------


def _round_to(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _stream(seed: int, stream: int, family: str, rendering: str) -> Rng:
    """Independent stream per seed, purpose, task family and rendering family."""
    return Rng(seed, stream, FAMILIES.index(family), RENDERINGS.index(rendering))


def sample_spec(family: str, seed: int, rendering: str) -> RenderSpec:
    rng = _stream(seed, _STREAM_SPEC, family, rendering)
    long_side = rng.choice(CANVAS_LONG_SIDES)
    aspect_w, aspect_h = rng.choice(ASPECTS[family])
    if aspect_w >= aspect_h:
        width, height = long_side, _round_to(long_side * aspect_h / aspect_w, 28)
    else:
        width, height = _round_to(long_side * aspect_w / aspect_h, 28), long_side
    fonts = FAMILIAR_FONTS if rendering == "familiar" else HELDOUT_FONTS
    low_px, high_px = FONT_PX_RANGE
    return RenderSpec(
        width=width,
        height=height,
        font=rng.choice(fonts),
        font_px=int(round(low_px * (high_px / low_px) ** rng.uniform())),
        layout=rng.choice(LAYOUTS[family][rendering]),
        background=rng.choice(BACKGROUNDS),
        ink=rng.choice(INKS),
        accent=rng.choice(ACCENTS),
        rendering=rendering,
    )


# --------------------------------------------------------------------------------------
# Bar charts
# --------------------------------------------------------------------------------------


def _distinct_values(rng: Rng, count: int, low: int, high: int) -> list[int]:
    values: list[int] = []
    while len(values) < count:
        candidate = rng.randint(low, high)
        if candidate not in values:
            values.append(candidate)
    return values


def _sample_chart(rng: Rng, rendering: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    set_name = rng.choice(sorted(vocab.CATEGORY_SETS))
    count = rng.randint(3, 8)
    categories = rng.sample(vocab.CATEGORY_SETS[set_name], count)
    low, high, decimals = rng.choice(NUMBER_RANGES[rendering])
    # Keep bars visible: the smallest value is at least a tenth of the largest possible one.
    floor = max(low, high // 10)
    values = _distinct_values(rng, count, floor, high)
    kind = rng.weighted(("value", "argmax", "argmin", "count"), (0.45, 0.25, 0.15, 0.15))
    content = {
        "title": rng.choice(vocab.CHART_TITLES),
        "noun": vocab.CATEGORY_NOUN[set_name],
        "categories": categories,
        "values": values,
        "decimals": decimals,
        "annotate": kind == "value" or rng.chance(0.5),
        "axis_max": nice_ceiling((max(values) * 13 + 9) // 10),
        "floor": floor,
        "hidden": None,
    }
    if kind == "value":
        index = rng.randint(0, count - 1)
        question = f"What is the value of {categories[index]}?"
        target = {"type": "value", "index": index}
    elif kind == "argmax":
        question = f"Which {content['noun']} has the highest value?"
        target = {"type": "argmax"}
    elif kind == "argmin":
        question = f"Which {content['noun']} has the lowest value?"
        target = {"type": "argmin"}
    else:
        question = "How many bars are shown in the chart?"
        target = {"type": "count"}
    return content, question, target


def _edit_chart(rng: Rng, content: dict[str, Any], target: dict[str, Any], fine: bool) -> None:
    values: list[int] = content["values"]
    ceiling = content["axis_max"] * 97 // 100
    kind = target["type"]
    if kind == "value":
        index = target["index"]
        values[index] = edit_number(rng, values[index], content["floor"], ceiling, fine)
    elif kind in ("argmax", "argmin"):
        extreme = max(values) if kind == "argmax" else min(values)
        others = [i for i, v in enumerate(values) if v != extreme]
        index = rng.choice(others)
        small = max(1, content["axis_max"] * 3 // 100)
        step = rng.randint(1, small) if fine else rng.randint(2 * small, 4 * small)
        if kind == "argmax":
            values[index] = min(extreme + step, ceiling)
        else:
            values[index] = max(extreme - step, max(1, content["floor"] // 2))
    else:
        content["hidden"] = rng.randint(0, len(values) - 1)


# --------------------------------------------------------------------------------------
# Numerical tables
# --------------------------------------------------------------------------------------


def _sample_table(rng: Rng, rendering: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    set_name = rng.choice(sorted(vocab.CATEGORY_SETS))
    rows = rng.sample(vocab.CATEGORY_SETS[set_name], rng.randint(3, 8))
    columns = rng.sample(vocab.TABLE_COLUMNS, rng.randint(2, 4))
    low, high, decimals = rng.choice(NUMBER_RANGES[rendering])
    cells = [_distinct_values(rng, len(columns), low, high) for _ in rows]
    kind = rng.weighted(("cell", "total", "argmax"), (0.5, 0.25, 0.25))
    column = rng.randint(0, len(columns) - 1)
    if kind == "argmax":
        # A unique maximum in the queried column.
        column_values = _distinct_values(rng, len(rows), low, high)
        for row, value in zip(cells, column_values, strict=True):
            row[column] = value
    content = {
        "title": rng.choice(vocab.TABLE_TITLES),
        "noun": vocab.CATEGORY_NOUN[set_name],
        "rows": rows,
        "columns": columns,
        "cells": cells,
        "decimals": decimals,
        "show_total": kind == "total" or rng.chance(0.3),
        "low": low,
        "high": high,
    }
    if kind == "cell":
        row = rng.randint(0, len(rows) - 1)
        question = f"What is the {columns[column]} value for {rows[row]}?"
        target = {"type": "cell", "row": row, "col": column}
    elif kind == "total":
        question = f"What is the total of the {columns[column]} column?"
        target = {"type": "total", "col": column}
    else:
        question = f"Which {content['noun']} has the highest {columns[column]} value?"
        target = {"type": "argmax", "col": column}
    return content, question, target


def _edit_table(rng: Rng, content: dict[str, Any], target: dict[str, Any], fine: bool) -> None:
    cells: list[list[int]] = content["cells"]
    low, high = content["low"], content["high"]
    kind, column = target["type"], target["col"]
    if kind == "cell":
        row = target["row"]
        cells[row][column] = edit_number(rng, cells[row][column], low, high, fine)
    elif kind == "total":
        row = rng.randint(0, len(cells) - 1)
        cells[row][column] = edit_number(rng, cells[row][column], low, high, fine)
    else:
        column_values = [row[column] for row in cells]
        best = max(column_values)
        if best >= high:
            # No room above the maximum: move the old maximum below the runner-up instead.
            runner_up = max(v for v in column_values if v != best)
            step = rng.randint(1, 9) if fine else rng.randint(10, 99)
            cells[column_values.index(best)][column] = max(runner_up - step, low)
            return
        row = rng.choice([i for i, v in enumerate(column_values) if v != best])
        span = high - best
        step = rng.randint(1, min(9, span)) if fine else rng.randint(1, span)
        cells[row][column] = best + step


# --------------------------------------------------------------------------------------
# Text cards
# --------------------------------------------------------------------------------------


def _digits(rng: Rng, count: int) -> str:
    first = rng.choice(vocab.DIGITS[1:])
    return first + "".join(rng.choice(vocab.DIGITS) for _ in range(count - 1))


def _letters(rng: Rng, count: int) -> str:
    return "".join(rng.choice(vocab.LETTERS) for _ in range(count))


def _field_value(rng: Rng, kind: str, rendering: str) -> str:
    """Draw the value of one askable field."""
    digits = rng.randint(*ID_DIGITS[rendering])
    if kind == "number":
        return _digits(rng, digits)
    if kind == "code":
        return f"{_letters(rng, 2)}-{_digits(rng, digits - 1)}"
    if kind == "room":
        return f"{rng.choice(vocab.LETTERS)}{_digits(rng, 3)}"
    if kind == "seat":
        return f"{rng.randint(1, 48)}{rng.choice('ABCDEF')}"
    if kind == "time":
        return f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"
    if kind == "date":
        return f"20{rng.randint(19, 27):02d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    if kind == "count":
        return str(rng.randint(2, 96))
    raise KeyError(kind)


#: card kind -> heading, list of (label, question phrase, value kind or None for filler)
CARD_FIELDS: dict[str, tuple[str, tuple[tuple[str, str, str | None], ...]]] = {
    "order": (
        "Order summary",
        (
            ("Order ID", "order ID", "number"),
            ("Date", "date", "date"),
            ("Customer", "customer", None),
            ("Items", "number of items", "count"),
            ("Tracking", "tracking code", "code"),
        ),
    ),
    "badge": (
        "Visitor badge",
        (
            ("Name", "name", None),
            ("Badge No", "badge number", "number"),
            ("Room", "room", "room"),
            ("Valid until", "valid-until date", "date"),
        ),
    ),
    "ticket": (
        "Boarding pass",
        (
            ("Passenger", "passenger", None),
            ("Flight", "flight code", "code"),
            ("Seat", "seat", "seat"),
            ("Boarding", "boarding time", "time"),
            ("Ticket No", "ticket number", "number"),
        ),
    ),
    "label": (
        "Shipping label",
        (
            ("SKU", "SKU", "code"),
            ("Batch", "batch number", "number"),
            ("Packed", "packing date", "date"),
            ("Boxes", "number of boxes", "count"),
        ),
    ),
}


def _sample_card(rng: Rng, rendering: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if rng.chance(0.3):
        count = rng.randint(3, 6)
        names = rng.sample(vocab.RECEIPT_ITEMS, count)
        cents_high = 4999 if rendering == "familiar" else 19999
        items = [[name, rng.randint(150, cents_high)] for name in names]
        content = {
            "kind": "receipt",
            "heading": rng.choice(vocab.SHOP_NAMES),
            "fields": [["Date", _field_value(rng, "date", rendering), None]],
            "items": items,
            "decimals": 2,
            "cents_high": cents_high,
            "rendering": rendering,
        }
        return content, "What is the total on the receipt?", {"type": "total"}
    kind = rng.choice(sorted(CARD_FIELDS))
    heading, specification = CARD_FIELDS[kind]
    fields: list[list[Any]] = []
    askable: list[int] = []
    for label, _phrase, value_kind in specification:
        if value_kind is None:
            value = f"{rng.choice(vocab.FIRST_NAMES)} {rng.choice(vocab.LAST_NAMES)}"
        else:
            value = _field_value(rng, value_kind, rendering)
            askable.append(len(fields))
        fields.append([label, value, value_kind])
    index = rng.choice(askable)
    phrase = specification[index][1]
    content = {
        "kind": kind,
        "heading": heading,
        "fields": fields,
        "items": [],
        "decimals": 2,
        "rendering": rendering,
    }
    return content, f"What is the {phrase}?", {"type": "field", "index": index}


def _edit_character(rng: Rng, value: str) -> str:
    """Replace one alphanumeric character by another of the same class."""
    positions = [i for i, ch in enumerate(value) if ch.isalnum()]
    for _ in range(64):
        position = rng.choice(positions)
        current = value[position]
        pool = vocab.DIGITS if current.isdigit() else vocab.LETTERS
        options = [ch for ch in pool if ch != current]
        if current.isdigit() and position == 0:
            options = [ch for ch in options if ch != "0"]
        candidate = value[:position] + rng.choice(options) + value[position + 1 :]
        if _valid_field(candidate, value):
            return candidate
    raise RuntimeError(f"no single-character edit found for {value!r}")


def _valid_field(candidate: str, reference: str) -> bool:
    """Keep times and dates well-formed after a character edit."""
    if candidate == reference:
        return False
    if len(reference) == 5 and reference[2] == ":":
        return int(candidate[:2]) <= 23 and int(candidate[3:]) <= 59
    if len(reference) == 10 and reference[4] == "-" and reference[7] == "-":
        return 1 <= int(candidate[5:7]) <= 12 and 1 <= int(candidate[8:]) <= 28
    return True


def _edit_card(rng: Rng, content: dict[str, Any], target: dict[str, Any], fine: bool) -> None:
    if target["type"] == "total":
        item = rng.choice(content["items"])
        item[1] = edit_number(rng, item[1], 150, content["cents_high"], fine)
        return
    field = content["fields"][target["index"]]
    if fine:
        field[1] = _edit_character(rng, field[1])
        return
    for _ in range(64):
        candidate = _field_value(rng, field[2], content["rendering"])
        if candidate != field[1]:
            field[1] = candidate
            return
    field[1] = _edit_character(rng, field[1])


# --------------------------------------------------------------------------------------
# Public interface
# --------------------------------------------------------------------------------------

_SAMPLERS = {"chart": _sample_chart, "table": _sample_table, "card": _sample_card}
_EDITORS = {"chart": _edit_chart, "table": _edit_table, "card": _edit_card}


def compute_answer(family: str, content: dict[str, Any], target: dict[str, Any]) -> str:
    """Compute the answer of a question from the structured content."""
    kind = target["type"]
    if family == "chart":
        visible = [i for i in range(len(content["values"])) if i != content["hidden"]]
        values = content["values"]
        if kind == "value":
            if target["index"] not in visible:
                raise ValueError("the queried bar is hidden")
            return format_number(values[target["index"]], content["decimals"])
        if kind == "count":
            return str(len(visible))
        chooser = max if kind == "argmax" else min
        best = chooser(visible, key=lambda i: values[i])
        if sum(values[i] == values[best] for i in visible) != 1:
            raise ValueError("the extreme bar is not unique")
        return content["categories"][best]
    if family == "table":
        column = [row[target["col"]] for row in content["cells"]]
        if kind == "cell":
            return format_number(
                content["cells"][target["row"]][target["col"]], content["decimals"]
            )
        if kind == "total":
            return format_number(sum(column), content["decimals"])
        if column.count(max(column)) != 1:
            raise ValueError("the maximum of the column is not unique")
        return content["rows"][column.index(max(column))]
    if family == "card":
        if kind == "total":
            return format_number(sum(price for _name, price in content["items"]), 2)
        return str(content["fields"][target["index"]][1])
    raise KeyError(f"unknown family {family!r}")


def sample_scene(family: str, seed: int, rendering: str = "familiar") -> Scene:
    """Draw the base scene of ``seed`` for a task family and rendering family."""
    if family not in FAMILIES:
        raise KeyError(f"unknown family {family!r}; expected one of {FAMILIES}")
    if rendering not in RENDERINGS:
        raise KeyError(f"unknown rendering {rendering!r}; expected one of {RENDERINGS}")
    spec = sample_spec(family, seed, rendering)
    rng = _stream(seed, _STREAM_CONTENT, family, rendering)
    content, question, target = _SAMPLERS[family](rng, rendering)
    return Scene(family, seed, spec, content, question, target)


def sample_edit(scene: Scene) -> tuple[Scene, bool]:
    """Edit one answer-determining variable of ``scene``.

    Returns the edited scene and whether the edit was drawn as fine-grained.  The rendering
    specification, the question and the target are those of the base scene.
    """
    rng = _stream(scene.seed, _STREAM_EDIT, scene.family, scene.spec.rendering)
    fine = rng.chance(FINE_EDIT_PROBABILITY)
    content = copy.deepcopy(scene.content)
    _EDITORS[scene.family](rng, content, scene.target, fine)
    return replace(scene, content=content), fine
