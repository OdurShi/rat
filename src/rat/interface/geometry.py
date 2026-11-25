"""Preprocessing geometry of the visual interfaces (Section 4.1, Appendix E.4).

Mode selection uses the actual visual-token counts produced by preprocessing.  These depend
only on the image size and the input mode, so they are computed here without running a
model.  Three interface families are covered:

* **Qwen2.5-VL** (variable tokens): aspect-ratio-preserving resize to multiples of 28 pixels
  with a pixel range; one token per 28 x 28 block.  ``L`` and ``H`` cap the image at 256 and
  1,024 tokens.
* **LLaVA-1.5** (fixed grid): the image is padded to a square and encoded at 336 pixels
  into 24 x 24 = 576 tokens.  ``H`` uses the padded input; ``L`` downsamples the padded
  canvas to 168 pixels and upsamples it back, so both modes cost 576 tokens.
* **SmolVLM / Idefics3** (tiles): the longest edge is resized to the mode's length, both
  sides are rounded up to multiples of the tile size, and each tile plus one global view
  gives a fixed number of tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Size = tuple[int, int]  # (width, height)


@dataclass(frozen=True)
class InputMode:
    """An input mode ``a`` of a visual interface.

    ``rank`` orders modes from least to most visual detail; ties in token cost are broken
    towards the lower rank (Eq. (4): ties favour ``L``).
    """

    name: str
    rank: int

    def __str__(self) -> str:
        return self.name


LOW = InputMode("L", 0)
HIGH = InputMode("H", 1)


@dataclass(frozen=True)
class TokenGrid:
    """Token layout of one image in one mode.

    ``parts`` lists rectangular token blocks ``(rows, cols, x0, y0, x1, y1)``: a block of
    ``rows x cols`` tokens covering the image region ``[x0, x1] x [y0, y1]`` in normalised
    image coordinates ``[-1, 1]``.  A single-grid interface has one part covering the whole
    image; a tiled interface has one part per tile and one for the global view.
    ``resized`` is the pixel size ``(width, height)`` the image content is resampled to.
    """

    parts: tuple[tuple[int, int, float, float, float, float], ...]
    resized: Size

    @property
    def count(self) -> int:
        return sum(rows * cols for rows, cols, *_ in self.parts)


# --------------------------------------------------------------------------------------
# Qwen2.5-VL
# --------------------------------------------------------------------------------------

QWEN_FACTOR = 28  # 14-pixel patches merged 2 x 2
QWEN_MIN_TOKENS = 4
QWEN_MODE_TOKENS = {"L": 256, "H": 1024}


def smart_resize(
    height: int,
    width: int,
    factor: int = QWEN_FACTOR,
    min_pixels: int = 56 * 56,
    max_pixels: int = 28 * 28 * 1280,
) -> tuple[int, int]:
    """Qwen2-VL / Qwen2.5-VL resize rule; returns ``(height, width)``.

    Both sides become multiples of ``factor``, the pixel count lies in
    ``[min_pixels, max_pixels]`` and the aspect ratio is kept as closely as possible.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def qwen_pixel_range(max_tokens: int) -> tuple[int, int]:
    """``(min_pixels, max_pixels)`` of a Qwen mode: ``4 x 28^2`` to ``max_tokens x 28^2``."""
    return QWEN_MIN_TOKENS * QWEN_FACTOR**2, max_tokens * QWEN_FACTOR**2


def qwen_grid(size: Size, max_tokens: int) -> TokenGrid:
    width, height = size
    min_pixels, max_pixels = qwen_pixel_range(max_tokens)
    h_bar, w_bar = smart_resize(height, width, QWEN_FACTOR, min_pixels, max_pixels)
    rows, cols = h_bar // QWEN_FACTOR, w_bar // QWEN_FACTOR
    return TokenGrid(((rows, cols, -1.0, -1.0, 1.0, 1.0),), (w_bar, h_bar))


# --------------------------------------------------------------------------------------
# LLaVA-1.5
# --------------------------------------------------------------------------------------

LLAVA_INPUT = 336
LLAVA_LOW_INPUT = 168
LLAVA_GRID = 24


def llava_grid(size: Size, mode: str, grid: int = LLAVA_GRID, high: int = LLAVA_INPUT) -> TokenGrid:
    """Fixed grid of ``grid x grid`` tokens over the padded square canvas.

    ``resized`` reports the size the image content has when it is last sampled below its
    final resolution: the padded square scaled to ``high`` pixels for ``H`` and to
    ``high / 2`` pixels for ``L``.
    """
    width, height = size
    side = high if mode == "H" else high // 2
    scale = side / max(width, height)
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    return TokenGrid(((grid, grid, -1.0, -1.0, 1.0, 1.0),), resized)


# --------------------------------------------------------------------------------------
# SmolVLM / Idefics3
# --------------------------------------------------------------------------------------


def _rescale_to_max_len(height: int, width: int, max_len: int) -> tuple[int, int]:
    aspect = width / height
    if width >= height:
        width = max_len
        height = int(width / aspect)
        if height % 2 != 0:
            height += 1
    else:
        height = max_len
        width = int(height * aspect)
        if width % 2 != 0:
            width += 1
    return max(height, 1), max(width, 1)


def tile_layout(size: Size, longest_edge: int, tile: int) -> tuple[int, int, Size]:
    """Rows and columns of tiles and the resized size for the SmolVLM image processor.

    Returns ``(rows, cols, (width, height))``; ``rows = cols = 0`` means a single view
    without a separate global image.
    """
    width, height = size
    height, width = _rescale_to_max_len(height, width, longest_edge)
    aspect = width / height
    if width >= height:
        width = math.ceil(width / tile) * tile
        height = math.ceil(int(width / aspect) / tile) * tile
    else:
        height = math.ceil(height / tile) * tile
        width = math.ceil(int(height * aspect) / tile) * tile
    if height > tile or width > tile:
        return height // tile, width // tile, (width, height)
    return 0, 0, (width, height)


def tile_grid(size: Size, longest_edge: int, tile: int, tokens_per_side: int) -> TokenGrid:
    """Token layout of the tiled interface: tiles in row-major order, then the global view."""
    rows, cols, resized = tile_layout(size, longest_edge, tile)
    k = tokens_per_side
    if rows == 0:
        return TokenGrid(((k, k, -1.0, -1.0, 1.0, 1.0),), resized)
    parts = []
    for r in range(rows):
        for c in range(cols):
            parts.append(
                (
                    k,
                    k,
                    -1 + 2 * c / cols,
                    -1 + 2 * r / rows,
                    -1 + 2 * (c + 1) / cols,
                    -1 + 2 * (r + 1) / rows,
                )
            )
    parts.append((k, k, -1.0, -1.0, 1.0, 1.0))
    return TokenGrid(tuple(parts), resized)


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def linear_scale(native: Size, resized: Size) -> float:
    """Geometric-mean linear scale factor from the native canvas to a resized image."""
    return math.sqrt((resized[0] * resized[1]) / (native[0] * native[1]))
