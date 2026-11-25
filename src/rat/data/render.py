"""Rasterisation of scenes with Pillow.

Every drawing call that carries information records an :class:`Element` (role, pixel box,
text, reference into the content).  The consistency checks of Appendix B.1 work on these
records: marks must agree with their annotations, required elements must lie inside the
canvas, and elements must not overlap unintentionally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image, ImageDraw

from .fonts import load_font
from .scenes import RenderSpec, Scene, format_number

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Element:
    """One drawn item.  ``box`` is ``(x0, y0, x1, y1)`` with exclusive right and bottom."""

    role: str
    box: Box
    text: str | None = None
    ref: Any = None


@dataclass
class Rendering:
    image: Image.Image
    elements: list[Element] = field(default_factory=list)
    #: Quantities the checks need to recompute mark geometry, e.g. the plot extent.
    geometry: dict[str, Any] = field(default_factory=dict)

    def by_role(self, role: str) -> list[Element]:
        return [element for element in self.elements if element.role == role]


def bar_pixels(value: int, axis_max: int, extent: int) -> int:
    """Length in pixels of a bar for ``value`` on an axis ``[0, axis_max]`` of ``extent`` pixels."""
    return max(1, int(round(value / axis_max * extent)))


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], weight: float) -> tuple[int, int, int]:
    """Blend ``a`` towards ``b`` by ``weight`` in ``[0, 1]``."""
    return tuple(int(round(x + (y - x) * weight)) for x, y in zip(a, b, strict=True))


class Canvas:
    """Thin wrapper around ``ImageDraw`` that records what was drawn."""

    def __init__(self, spec: RenderSpec) -> None:
        self.spec = spec
        self.image = Image.new("RGB", (spec.width, spec.height), spec.background)
        self.draw = ImageDraw.Draw(self.image)
        self.elements: list[Element] = []
        self.faint = _mix(spec.ink, spec.background, 0.82)
        self.soft = _mix(spec.ink, spec.background, 0.45)

    def text(
        self,
        xy: tuple[float, float],
        text: str,
        size: int,
        role: str,
        anchor: str = "la",
        bold: bool = False,
        fill: tuple[int, int, int] | None = None,
        ref: Any = None,
    ) -> Box:
        font = load_font(self.spec.font, size, bold)
        position = (int(round(xy[0])), int(round(xy[1])))
        left, top, right, bottom = self.draw.textbbox(position, text, font=font, anchor=anchor)
        self.draw.text(position, text, font=font, fill=fill or self.spec.ink, anchor=anchor)
        box = (int(left), int(top), int(right) + 1, int(bottom) + 1)
        self.elements.append(Element(role, box, text, ref))
        return box

    def width_of(self, text: str, size: int, bold: bool = False) -> int:
        font = load_font(self.spec.font, size, bold)
        left, _top, right, _bottom = self.draw.textbbox((0, 0), text, font=font, anchor="la")
        return int(right - left) + 1

    def rect(
        self,
        box: Box,
        fill: tuple[int, int, int] | None,
        role: str | None = None,
        ref: Any = None,
        outline: tuple[int, int, int] | None = None,
        width: int = 1,
    ) -> None:
        x0, y0, x1, y1 = box
        self.draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=fill, outline=outline, width=width)
        if role is not None:
            self.elements.append(Element(role, box, None, ref))

    def line(
        self, start: tuple[float, float], end: tuple[float, float], fill, width: int = 1
    ) -> None:
        self.draw.line((start, end), fill=fill, width=max(1, width))


def _sizes(spec: RenderSpec) -> tuple[int, int, int, int]:
    """Body, title and small text sizes and the outer padding."""
    body = spec.font_px
    return (
        body,
        int(round(1.3 * body)),
        max(7, int(round(0.85 * body))),
        max(12, int(round(0.045 * min(spec.width, spec.height)))),
    )


# --------------------------------------------------------------------------------------
# Bar charts
# --------------------------------------------------------------------------------------


def _render_chart(scene: Scene) -> Rendering:
    spec, content = scene.spec, scene.content
    canvas = Canvas(spec)
    body, title_size, small, pad = _sizes(spec)
    decimals, axis_max = content["decimals"], content["axis_max"]
    values, categories = content["values"], content["categories"]
    slots = len(values)
    stroke = max(1, body // 10)
    horizontal = spec.layout == "horizontal"

    if spec.layout == "vertical":
        title_box = canvas.text((pad, pad), content["title"], title_size, "title", "la", bold=True)
    else:
        title_box = canvas.text(
            (spec.width / 2, pad), content["title"], title_size, "title", "ma", bold=True
        )

    ticks = [axis_max * k // 4 for k in range(5)]
    tick_labels = [format_number(tick, decimals) for tick in ticks]

    if not horizontal:
        tick_width = max(canvas.width_of(label, small) for label in tick_labels)
        left = pad + tick_width + int(round(0.6 * body))
        right = spec.width - pad
        top = title_box[3] + int(round(1.2 * body))
        bottom = spec.height - pad - int(round(1.8 * body))
        extent = bottom - top
        for tick, label in zip(ticks, tick_labels, strict=True):
            y = bottom - int(round(tick / axis_max * extent))
            if spec.layout == "vertical_grid" and tick:
                canvas.line((left, y), (right, y), canvas.faint, stroke)
            canvas.text((left - 0.4 * body, y), label, small, "tick", "rm", fill=canvas.soft)
        slot = (right - left) / slots
        thickness = max(4, int(round(0.58 * slot)))
        for index, (value, name) in enumerate(zip(values, categories, strict=True)):
            if index == content["hidden"]:
                continue
            x0 = int(round(left + slot * index + (slot - thickness) / 2))
            length = bar_pixels(value, axis_max, extent)
            canvas.rect((x0, bottom - length, x0 + thickness, bottom), spec.accent, "bar", index)
            centre = x0 + thickness / 2
            canvas.text((centre, bottom + 0.45 * body), name, body, "category", "ma", ref=index)
            if content["annotate"]:
                canvas.text(
                    (centre, bottom - length - 0.3 * small),
                    format_number(value, decimals),
                    small,
                    "annotation",
                    "mb",
                    ref=index,
                )
        canvas.line((left, top), (left, bottom), spec.ink, stroke)
        canvas.line((left, bottom), (right, bottom), spec.ink, stroke)
    else:
        label_width = max(canvas.width_of(name, body) for name in categories)
        left = pad + label_width + int(round(0.6 * body))
        right = spec.width - pad - int(round(0.5 * body))
        top = title_box[3] + int(round(1.0 * body))
        bottom = spec.height - pad - int(round(1.7 * small))
        extent = right - left
        for tick, label in zip(ticks, tick_labels, strict=True):
            x = left + int(round(tick / axis_max * extent))
            if tick:
                canvas.line((x, top), (x, bottom), canvas.faint, stroke)
            canvas.text((x, bottom + 0.4 * small), label, small, "tick", "ma", fill=canvas.soft)
        slot = (bottom - top) / slots
        thickness = max(4, int(round(0.58 * slot)))
        for index, (value, name) in enumerate(zip(values, categories, strict=True)):
            if index == content["hidden"]:
                continue
            y0 = int(round(top + slot * index + (slot - thickness) / 2))
            length = bar_pixels(value, axis_max, extent)
            canvas.rect((left, y0, left + length, y0 + thickness), spec.accent, "bar", index)
            centre = y0 + thickness / 2
            canvas.text((left - 0.5 * body, centre), name, body, "category", "rm", ref=index)
            if content["annotate"]:
                canvas.text(
                    (left + length + 0.35 * small, centre),
                    format_number(value, decimals),
                    small,
                    "annotation",
                    "lm",
                    ref=index,
                )
        canvas.line((left, top), (left, bottom), spec.ink, stroke)
        canvas.line((left, bottom), (right, bottom), spec.ink, stroke)

    geometry = {"extent": extent, "horizontal": horizontal, "plot": (left, top, right, bottom)}
    return Rendering(canvas.image, canvas.elements, geometry)


# --------------------------------------------------------------------------------------
# Numerical tables
# --------------------------------------------------------------------------------------


def _render_table(scene: Scene) -> Rendering:
    spec, content = scene.spec, scene.content
    canvas = Canvas(spec)
    body, title_size, _small, pad = _sizes(spec)
    decimals = content["decimals"]
    rows, columns, cells = content["rows"], content["columns"], content["cells"]
    boxed = spec.layout == "boxed"
    stroke = max(1, body // 10)

    totals = [sum(row[c] for row in cells) for c in range(len(columns))]
    texts = [[format_number(value, decimals) for value in row] for row in cells]
    total_texts = [format_number(value, decimals) for value in totals]

    gap = int(round(1.4 * body))
    first_labels = [content["noun"].capitalize(), *rows, "Total"]
    first_width = max(canvas.width_of(label, body, bold=True) for label in first_labels) + gap
    widths = []
    for c, header in enumerate(columns):
        entries = [canvas.width_of(header, body, bold=True)]
        entries += [canvas.width_of(texts[r][c], body) for r in range(len(rows))]
        entries.append(canvas.width_of(total_texts[c], body, bold=True))
        widths.append(max(entries) + gap)
    table_width = first_width + sum(widths)
    row_height = int(round(1.9 * body))

    if boxed:
        x0 = (spec.width - table_width) // 2
        title_box = canvas.text(
            (spec.width / 2, pad), content["title"], title_size, "title", "ma", bold=True
        )
    else:
        x0 = pad
        title_box = canvas.text((pad, pad), content["title"], title_size, "title", "la", bold=True)
    y0 = title_box[3] + int(round(0.9 * body))
    line_count = 1 + len(rows) + (1 if content["show_total"] else 0)
    x1, y1 = x0 + table_width, y0 + row_height * line_count

    edges = [x0 + first_width]
    for width in widths:
        edges.append(edges[-1] + width)

    def number_x(column: int) -> tuple[float, str]:
        if boxed:
            return (edges[column] + edges[column + 1]) / 2, "mm"
        return edges[column + 1] - gap / 2, "rm"

    if boxed:
        canvas.rect((x0, y0, x1, y0 + row_height), _mix(spec.accent, spec.background, 0.75), None)
    if spec.layout == "zebra":
        for r in range(len(rows)):
            if r % 2 == 0:
                top = y0 + row_height * (r + 1)
                canvas.rect((x0, top, x1, top + row_height), canvas.faint, None)

    middle = y0 + row_height / 2
    canvas.text(
        (x0 + gap / 2, middle), content["noun"].capitalize(), body, "corner", "lm", bold=True
    )
    for c, header in enumerate(columns):
        x, anchor = number_x(c)
        canvas.text((x, middle), header, body, "header", anchor, bold=True, ref=c)
    for r, name in enumerate(rows):
        middle = y0 + row_height * (r + 1) + row_height / 2
        canvas.text((x0 + gap / 2, middle), name, body, "row", "lm", ref=r)
        for c in range(len(columns)):
            x, anchor = number_x(c)
            canvas.text((x, middle), texts[r][c], body, "cell", anchor, ref=(r, c))
    if content["show_total"]:
        middle = y0 + row_height * (len(rows) + 1) + row_height / 2
        canvas.text((x0 + gap / 2, middle), "Total", body, "total_label", "lm", bold=True)
        for c in range(len(columns)):
            x, anchor = number_x(c)
            canvas.text((x, middle), total_texts[c], body, "total", anchor, bold=True, ref=c)

    if boxed:
        for k in range(line_count + 1):
            y = y0 + row_height * k
            canvas.line((x0, y), (x1, y), spec.ink, stroke)
        for x in (x0, *edges):
            canvas.line((x, y0), (x, y1), spec.ink, stroke)
    else:
        canvas.line((x0, y0), (x1, y0), spec.ink, stroke)
        canvas.line((x0, y0 + row_height), (x1, y0 + row_height), spec.ink, stroke)
        if content["show_total"]:
            y = y1 - row_height
            canvas.line((x0, y), (x1, y), canvas.soft, stroke)
        canvas.line((x0, y1), (x1, y1), spec.ink, stroke)

    canvas.elements.append(Element("frame", (x0, y0, x1 + 1, y1 + 1)))
    return Rendering(canvas.image, canvas.elements, {"table": (x0, y0, x1, y1)})


# --------------------------------------------------------------------------------------
# Text cards
# --------------------------------------------------------------------------------------


def _render_card(scene: Scene) -> Rendering:
    spec, content = scene.spec, scene.content
    canvas = Canvas(spec)
    body, title_size, small, pad = _sizes(spec)
    stroke = max(1, body // 10)
    boxed = spec.layout == "inline_boxed"
    inner = pad + int(round(0.9 * body))
    left, right = inner, spec.width - inner

    if boxed:
        canvas.rect(
            (pad, pad, spec.width - pad, spec.height - pad),
            None,
            None,
            outline=spec.ink,
            width=stroke,
        )
        heading = canvas.text(
            (spec.width / 2, inner), content["heading"], title_size, "title", "ma", bold=True
        )
    else:
        heading = canvas.text(
            (left, inner), content["heading"], title_size, "title", "la", bold=True
        )
    y = heading[3] + int(round(0.6 * body))
    canvas.line((left, y), (right, y), spec.accent, max(stroke, 2))
    y += int(round(0.9 * body))

    def row(label: str, value: str, index: Any, bold: bool = False, role: str = "value") -> None:
        nonlocal y
        if spec.layout == "stacked" and role == "value":
            canvas.text((left, y), label, small, "key", "la", fill=canvas.soft, ref=index)
            y += int(round(1.25 * small))
            canvas.text((left, y), value, body, role, "la", bold=bold, ref=index)
            y += int(round(1.75 * body))
        elif boxed and role == "value":
            key = canvas.text((left, y), f"{label}:", body, "key", "la", bold=True, ref=index)
            canvas.text((key[2] + 0.5 * body, y), value, body, role, "la", ref=index)
            y += int(round(1.9 * body))
        else:
            canvas.text((left, y), label, body, "key", "la", bold=bold, ref=index)
            canvas.text((right, y), value, body, role, "ra", bold=bold, ref=index)
            y += int(round(1.9 * body))

    for index, (label, value, _kind) in enumerate(content["fields"]):
        row(label, value, index)
    if content["kind"] == "receipt":
        for index, (name, price) in enumerate(content["items"]):
            row(name, format_number(price, 2), index, role="price")
        canvas.line((left, y), (right, y), canvas.soft, stroke)
        y += int(round(0.7 * body))
        total = sum(price for _name, price in content["items"])
        row("Total", format_number(total, 2), None, bold=True, role="total")
    return Rendering(canvas.image, canvas.elements, {"content_bottom": y})


_RENDERERS = {"chart": _render_chart, "table": _render_table, "card": _render_card}


def render_scene(scene: Scene) -> Rendering:
    """Rasterise a scene at its native canvas size."""
    return _RENDERERS[scene.family](scene)
