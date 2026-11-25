import pytest
import torch

from rat.backbones.base import LoraSettings
from rat.backbones.tiny import TinyBackbone
from rat.data.splits import generate_pool
from rat.features import open_pool


@pytest.fixture(scope="session")
def tiny_backbone():
    backbone = TinyBackbone(seed=0)
    backbone.add_adapter(LoraSettings(rank=4, alpha=8.0, dropout=0.0), seed=17)
    return backbone


@pytest.fixture(scope="session")
def tiny_pools(tiny_backbone, tmp_path_factory):
    """Small cached pools of the tiny backbone: both modes for training-side pools."""
    cache = tmp_path_factory.mktemp("cache")
    pools = {}
    for name, size in (("reader_fit", 24), ("tune", 12), ("dev", 6), ("train", 30), ("test", 12)):
        records = generate_pool(name, size)
        modes = None if name in ("reader_fit", "tune", "train") else (tiny_backbone.eval_mode,)
        pools[name] = open_pool(tiny_backbone, records, cache, name, modes)
    return pools


@pytest.fixture(autouse=True)
def _reset_adapter(request):
    yield
    if "tiny_backbone" in request.fixturenames:
        request.getfixturevalue("tiny_backbone").reset_adapter()
        torch.manual_seed(0)
