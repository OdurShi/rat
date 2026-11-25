import pytest
import torch

from rat.interface.geometry import (
    HIGH,
    LOW,
    llava_grid,
    qwen_grid,
    qwen_pixel_range,
    smart_resize,
    tile_grid,
    tile_layout,
)
from rat.reader import grid_coordinates


def test_smart_resize_respects_factor_and_pixel_range():
    for height, width in ((896, 1456), (364, 560), (1904, 1428), (40, 30), (3000, 500)):
        for cap in (256, 1024):
            low, high = qwen_pixel_range(cap)
            h, w = smart_resize(height, width, 28, low, high)
            assert h % 28 == 0 and w % 28 == 0
            assert low <= h * w <= high


def test_smart_resize_matches_transformers():
    transformers = pytest.importorskip("transformers")
    del transformers
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize as reference

    for height, width in ((896, 1456), (364, 560), (1904, 1428), (784, 784), (97, 1203)):
        for cap in (256, 1024):
            low, high = qwen_pixel_range(cap)
            assert smart_resize(height, width, 28, low, high) == reference(
                height, width, 28, low, high
            )


def test_qwen_grid_counts():
    assert qwen_grid((448, 448), 256).count == 256
    assert qwen_grid((896, 896), 1024).count == 1024
    assert qwen_grid((896, 896), 256).count == 256
    small = (280, 224)
    assert qwen_grid(small, 256).count == qwen_grid(small, 1024).count == 80
    wide = qwen_grid((1456, 896), 256)
    rows, cols = wide.parts[0][:2]
    assert rows * cols == wide.count <= 256 and wide.resized == (cols * 28, rows * 28)


def test_llava_grid_is_fixed():
    for size in ((560, 364), (1904, 1428), (336, 336)):
        assert llava_grid(size, "L").count == llava_grid(size, "H").count == 576
    assert llava_grid((672, 336), "H").resized == (336, 168)
    assert llava_grid((672, 336), "L").resized == (168, 84)


def test_tile_layout_follows_the_processor_rule():
    assert tile_layout((1456, 896), 512, 512) == (0, 0, (512, 512))
    assert tile_layout((1456, 896), 1024, 512) == (2, 2, (1024, 1024))
    assert tile_layout((1024, 300), 1024, 512) == (1, 2, (1024, 512))
    assert tile_grid((1456, 896), 512, 512, 8).count == 64
    assert tile_grid((1456, 896), 1024, 512, 8).count == 5 * 64
    assert tile_grid((1024, 300), 1024, 512, 8).count == 3 * 64


def test_tile_counts_match_transformers():
    pytest.importorskip("transformers")
    from transformers.models.smolvlm.image_processing_pil_smolvlm import SmolVLMImageProcessorPil

    processor = SmolVLMImageProcessorPil(max_image_size={"longest_edge": 512})
    for size in ((1456, 896), (560, 364), (1428, 1904), (1024, 300)):
        for edge in (512, 1024, 1536):
            rows, cols, _ = tile_layout(size, edge, 512)
            frames, n_rows, n_cols = processor.get_number_of_image_patches(
                size[1], size[0], {"size": {"longest_edge": edge}}
            )
            assert (n_rows, n_cols) == (rows, cols)
            assert frames == (rows * cols + 1 if rows else 0)


def test_coordinates_follow_equation_s4():
    grid = qwen_grid((84, 56), 1024)  # 2 rows x 3 columns
    coordinates = grid_coordinates(grid)
    assert coordinates.shape == (6, 2)
    expected = [((2 * c + 1) / 3 - 1, (2 * r + 1) / 2 - 1) for r in range(2) for c in range(3)]
    assert torch.allclose(coordinates, torch.tensor(expected), atol=1e-6)
    tiles = tile_grid((1024, 1024), 1024, 512, 2)  # 2 x 2 tiles of 2 x 2 tokens + global view
    points = grid_coordinates(tiles)
    assert points.shape == (5 * 4, 2)
    assert points[0].tolist() == pytest.approx([-0.75, -0.75])
    assert points[-1].tolist() == pytest.approx([0.5, 0.5])
    assert float(points.abs().max()) < 1


def test_modes_are_ordered():
    assert LOW.rank < HIGH.rank and str(LOW) == "L"
