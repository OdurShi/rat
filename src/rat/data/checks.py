"""Consistency checks of the pair generator (Appendix B.1).

A rendering is accepted only if the question identifies a valid target, the stored answer
matches the scene data, graphical marks agree with their annotations, and the required
elements lie within the canvas without unintended overlap.  :func:`check_scene` returns the
list of violated conditions (empty when the rendering is consistent).
"""

from __future__ import annotations

from .render import Box, Element, Rendering, bar_pixels
from .scenes import Scene, compute_answer, format_number


def _intersects(a: Box, b: Box, slack: int = 1) -> bool:
    """Whether two boxes overlap by more than ``slack`` pixels in both directions."""
    return min(a[2], b[2]) - max(a[0], b[0]) > slack and min(a[3], b[3]) - max(a[1], b[1]) > slack


def _inside(box: Box, width: int, height: int) -> bool:
    return box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height


def _one(elements: list[Element], role: str, ref) -> Element | None:
    matches = [e for e in elements if e.role == role and e.ref == ref]
    return matches[0] if len(matches) == 1 else None


def _check_chart(scene: Scene, rendering: Rendering, problems: list[str]) -> None:
    content, target = scene.content, scene.target
    elements, extent = rendering.elements, rendering.geometry["extent"]
    horizontal = rendering.geometry["horizontal"]
    visible = [i for i in range(len(content["values"])) if i != content["hidden"]]
    if len(set(content["categories"])) != len(content["categories"]):
        problems.append("category labels are not unique")
    lengths: dict[int, int] = {}
    for index in visible:
        value = content["values"][index]
        bar = _one(elements, "bar", index)
        label = _one(elements, "category", index)
        if bar is None or label is None:
            problems.append(f"bar {index} or its label is missing")
            continue
        length = bar.box[2] - bar.box[0] if horizontal else bar.box[3] - bar.box[1]
        lengths[index] = length
        if length != bar_pixels(value, content["axis_max"], extent):
            problems.append(f"bar {index} does not match its value")
        if label.text != content["categories"][index]:
            problems.append(f"label of bar {index} does not match the content")
        if content["annotate"]:
            note = _one(elements, "annotation", index)
            if note is None or note.text != format_number(value, content["decimals"]):
                problems.append(f"annotation of bar {index} does not match its value")
    if max(content["values"][i] for i in visible) > content["axis_max"]:
        problems.append("a bar exceeds the axis range")
    kind = target["type"]
    if kind == "value" and (target["index"] not in visible or not content["annotate"]):
        problems.append("the queried bar has no visible value")
    if kind in ("argmax", "argmin") and lengths:
        ordered = sorted(lengths.values(), reverse=kind == "argmax")
        if len(ordered) < 2 or ordered[0] == ordered[1]:
            problems.append("the extreme bar is not distinguishable in pixels")
    # Annotations and labels must not run into other bars.
    for element in elements:
        if element.role in ("annotation", "category", "title"):
            for bar in rendering.by_role("bar"):
                if bar.ref != element.ref and _intersects(element.box, bar.box):
                    problems.append(f"{element.role} {element.ref} overlaps bar {bar.ref}")


def _check_table(scene: Scene, rendering: Rendering, problems: list[str]) -> None:
    content = scene.content
    elements = rendering.elements
    for r, row in enumerate(content["cells"]):
        for c, value in enumerate(row):
            cell = _one(elements, "cell", (r, c))
            if cell is None or cell.text != format_number(value, content["decimals"]):
                problems.append(f"cell ({r}, {c}) does not match the content")
    if len(set(content["rows"])) != len(content["rows"]):
        problems.append("row labels are not unique")
    if len(set(content["columns"])) != len(content["columns"]):
        problems.append("column headers are not unique")
    if content["show_total"]:
        for c in range(len(content["columns"])):
            expected = format_number(sum(row[c] for row in content["cells"]), content["decimals"])
            total = _one(elements, "total", c)
            if total is None or total.text != expected:
                problems.append(f"total of column {c} does not match the cells")
    elif scene.target["type"] == "total":
        problems.append("the queried total is not displayed")
    frame = rendering.by_role("frame")[0].box
    for element in elements:
        if element.text is not None and element.role != "title":
            box = element.box
            if box[0] < frame[0] or box[2] > frame[2] or box[1] < frame[1] or box[3] > frame[3]:
                problems.append(f"{element.role} {element.ref} leaves the table frame")


def _check_card(scene: Scene, rendering: Rendering, problems: list[str]) -> None:
    content = scene.content
    elements = rendering.elements
    for index, (label, value, _kind) in enumerate(content["fields"]):
        shown = _one(elements, "value", index)
        if shown is None or shown.text != value:
            problems.append(f"field {label!r} does not match the content")
    labels = [label for label, _value, _kind in content["fields"]]
    if len(set(labels)) != len(labels):
        problems.append("field labels are not unique")
    if content["kind"] == "receipt":
        expected = format_number(sum(price for _name, price in content["items"]), 2)
        totals = rendering.by_role("total")
        if len(totals) != 1 or totals[0].text != expected:
            problems.append("the receipt total does not match its items")
        for index, (_name, price) in enumerate(content["items"]):
            shown = _one(elements, "price", index)
            if shown is None or shown.text != format_number(price, 2):
                problems.append(f"price {index} does not match the content")
    if rendering.geometry["content_bottom"] > scene.spec.height:
        problems.append("the card content leaves the canvas")


_CHECKS = {"chart": _check_chart, "table": _check_table, "card": _check_card}


def check_scene(scene: Scene, rendering: Rendering) -> list[str]:
    """Return the violated consistency conditions of one rendering."""
    problems: list[str] = []
    try:
        answer = compute_answer(scene.family, scene.content, scene.target)
        if not answer or answer != answer.strip():
            problems.append("the answer is empty or carries surrounding whitespace")
    except (ValueError, IndexError, KeyError) as error:
        problems.append(f"the question has no valid target: {error}")
    _CHECKS[scene.family](scene, rendering, problems)
    width, height = scene.spec.width, scene.spec.height
    texts = [e for e in rendering.elements if e.text is not None]
    for element in rendering.elements:
        if not _inside(element.box, width, height):
            problems.append(f"{element.role} {element.ref} lies outside the canvas")
    for i, first in enumerate(texts):
        for second in texts[i + 1 :]:
            if _intersects(first.box, second.box):
                problems.append(
                    f"text {first.role} {first.ref} overlaps text {second.role} {second.ref}"
                )
    return problems
