"""Image pairs with a shared question and different program-computed answers (Section 4.1).

A pair is identified by ``(family, seed, rendering)``.  :func:`build_pair` draws the base
scene, edits one answer-determining variable, renders both scenes and applies the checks of
Appendix B.1.  It returns ``None`` for rejected candidates: equal answers, an invalid target,
an inconsistent rendering, or identical rendered images.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .checks import check_scene
from .render import render_scene
from .scenes import Scene, sample_edit, sample_scene

#: Two pixels count as different when any channel differs by more than this (0-255 scale).
DIFFERENCE_THRESHOLD = 16


@dataclass(frozen=True)
class Pair:
    """A training or test pair ``u = (I0, I1, q, y0, y1)``."""

    family: str
    seed: int
    rendering: str
    question: str
    answers: tuple[str, str]
    scenes: tuple[Scene, Scene]
    fine_edit: bool
    #: Number of pixels that differ between the two renderings at the native canvas size.
    edit_area: int
    width: int
    height: int

    @property
    def size(self) -> tuple[int, int]:
        """Native canvas size ``(width, height)``."""
        return self.width, self.height

    def images(self) -> tuple[Image.Image, Image.Image]:
        """Render both images at the native canvas size."""
        return render_scene(self.scenes[0]).image, render_scene(self.scenes[1]).image

    def record(self) -> dict[str, Any]:
        """Manifest entry.  ``family``, ``seed`` and ``rendering`` regenerate the pair; the
        other fields document it and are verified when a manifest is loaded."""
        spec = self.scenes[0].spec
        return {
            "family": self.family,
            "seed": self.seed,
            "rendering": self.rendering,
            "question": self.question,
            "answers": list(self.answers),
            "task": self.scenes[0].target["type"],
            "width": self.width,
            "height": self.height,
            "layout": spec.layout,
            "font": spec.font,
            "font_px": spec.font_px,
            "fine_edit": self.fine_edit,
            "edit_area": self.edit_area,
        }


def changed_pixels(first: Image.Image, second: Image.Image) -> int:
    """Number of pixel positions whose colour differs between two equally sized images."""
    a = np.asarray(first.convert("RGB"), dtype=np.int16)
    b = np.asarray(second.convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        raise ValueError("images of a pair must share their size")
    return int((np.abs(a - b).max(axis=-1) > DIFFERENCE_THRESHOLD).sum())


def edit_scale(edit_area: int, native_size: tuple[int, int], mode_size: tuple[int, int]) -> float:
    """Projected edit scale (Appendix A.2).

    The square root of the edited-region area, measured before rasterisation to an input
    mode and expressed in the pixel units of that mode.  ``native_size`` and ``mode_size``
    are ``(width, height)`` of the canvas and of the mode's resized image.
    """
    native_w, native_h = native_size
    mode_w, mode_h = mode_size
    return math.sqrt(edit_area * (mode_w * mode_h) / (native_w * native_h))


def build_pair_report(
    family: str, seed: int, rendering: str = "familiar"
) -> tuple[Pair | None, list[str]]:
    """Build the pair of a candidate seed and return it with the reasons for rejection."""
    base = sample_scene(family, seed, rendering)
    try:
        edited, fine = sample_edit(base)
        answers = (base.answer, edited.answer)
    except (ValueError, RuntimeError) as error:
        return None, [f"no valid edit: {error}"]
    if answers[0] == answers[1]:
        return None, ["the edit does not change the answer"]
    renderings = (render_scene(base), render_scene(edited))
    problems = [
        p
        for scene, r in zip((base, edited), renderings, strict=True)
        for p in check_scene(scene, r)
    ]
    if problems:
        return None, problems
    area = changed_pixels(renderings[0].image, renderings[1].image)
    if area == 0:
        return None, ["the two renderings are identical"]
    pair = Pair(
        family=family,
        seed=seed,
        rendering=rendering,
        question=base.question,
        answers=answers,
        scenes=(base, edited),
        fine_edit=fine,
        edit_area=area,
        width=base.spec.width,
        height=base.spec.height,
    )
    return pair, []


def build_pair(family: str, seed: int, rendering: str = "familiar") -> Pair | None:
    """Build the pair of a candidate seed, or ``None`` if the candidate is rejected."""
    return build_pair_report(family, seed, rendering)[0]
