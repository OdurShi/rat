"""Fonts of the scene renderer.

The TrueType files are shipped with the package (DejaVu and STIX, see the licence files in
``fonts/``) so that rendering does not depend on fonts installed on the machine.  The
familiar rendering family uses the two sans-serif faces; the held-out rendering family of
the paired test set uses the two serif faces (Appendix E.1).
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

from PIL import ImageFont

#: font key -> (regular file, bold file)
FONT_FILES: dict[str, tuple[str, str]] = {
    "sans": ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
    "mono": ("DejaVuSansMono.ttf", "DejaVuSans-Bold.ttf"),
    "serif": ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf"),
    "stix": ("STIXGeneral.ttf", "STIXGeneralBol.ttf"),
}

FAMILIAR_FONTS: tuple[str, ...] = ("sans", "mono")
HELDOUT_FONTS: tuple[str, ...] = ("serif", "stix")


@lru_cache(maxsize=512)
def load_font(key: str, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Return the face ``key`` at ``size`` pixels."""
    if key not in FONT_FILES:
        raise KeyError(f"unknown font {key!r}; available: {sorted(FONT_FILES)}")
    if size < 1:
        raise ValueError("font size must be positive")
    name = FONT_FILES[key][1 if bold else 0]
    path = resources.files("rat.data").joinpath("fonts", name)
    with resources.as_file(path) as file:
        return ImageFont.truetype(str(file), size=size)
