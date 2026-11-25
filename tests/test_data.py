import numpy as np
import pytest

from rat.data.checks import check_scene
from rat.data.pairs import build_pair, build_pair_report, changed_pixels, edit_scale
from rat.data.render import bar_pixels, render_scene
from rat.data.rng import Rng, permutation
from rat.data.scenes import (
    FAMILIES,
    compute_answer,
    edit_number,
    format_number,
    nice_ceiling,
    sample_edit,
    sample_scene,
)
from rat.data.splits import (
    POOLS,
    generate_pool,
    pair_from_record,
    pool_of_seed,
    read_manifest,
    write_manifest,
)


def test_rng_is_reproducible_and_keyed():
    assert [Rng(3, 1).randint(0, 99) for _ in range(3)] == [
        Rng(3, 1).randint(0, 99) for _ in range(3)
    ]
    assert Rng(3, 1).u64() != Rng(3, 2).u64()
    assert sorted(permutation(20, 5)) == list(range(20))
    draws = [Rng(7).randint(2, 4) for _ in range(1)] + [Rng(k).randint(2, 4) for k in range(200)]
    assert set(draws) == {2, 3, 4}


def test_number_helpers():
    assert format_number(1234, 2) == "12.34"
    assert format_number(5, 1) == "0.5"
    assert format_number(70, 0) == "70"
    assert [nice_ceiling(v) for v in (7, 13, 99, 101, 1234)] == [10, 15, 100, 120, 1500]
    rng = Rng(1)
    for value in (10, 57, 999, 4321):
        fine = edit_number(rng, value, 10, 9999, fine=True)
        assert fine != value and len(str(fine)) == len(str(value))
        assert sum(a != b for a, b in zip(str(fine), str(value), strict=True)) == 1
        assert edit_number(rng, value, 10, 9999, fine=False) != value


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("rendering", ["familiar", "heldout"])
def test_pairs_share_everything_but_the_edited_variable(family, rendering):
    accepted = 0
    for seed in range(40):
        pair = build_pair(family, seed, rendering)
        if pair is None:
            continue
        accepted += 1
        base, edited = pair.scenes
        assert (
            base.spec == edited.spec
            and base.question == edited.question
            and base.target == edited.target
        )
        assert pair.answers[0] != pair.answers[1]
        assert pair.answers == (base.answer, edited.answer)
        assert base.spec.rendering == rendering
        first, second = pair.images()
        assert first.size == second.size == pair.size
        assert pair.edit_area == changed_pixels(first, second) > 0
    assert accepted >= 30


def test_generation_is_deterministic():
    first, second = build_pair("table", 11), build_pair("table", 11)
    assert first.record() == second.record()
    assert np.array_equal(np.asarray(first.images()[1]), np.asarray(second.images()[1]))
    assert build_pair("table", 11).record() != build_pair("table", 12).record()


def test_answers_are_computed_from_content():
    scene = sample_scene("chart", 3)
    edited, _fine = sample_edit(scene)
    assert compute_answer("chart", scene.content, scene.target) == scene.answer
    assert edited.content != scene.content
    table = sample_scene("table", 5)
    if table.target["type"] == "total":
        column = table.target["col"]
        total = sum(row[column] for row in table.content["cells"])
        assert table.answer == format_number(total, table.content["decimals"])


def test_checks_detect_inconsistent_renderings():
    pair = None
    seed = 0
    while pair is None or pair.scenes[0].target["type"] != "value":
        pair = build_pair("chart", seed)
        seed += 1
    scene = pair.scenes[0]
    rendering = render_scene(scene)
    assert check_scene(scene, rendering) == []
    # Marks that disagree with the content are reported.
    broken = sample_scene("chart", scene.seed)
    broken.content["values"][0] += 7
    assert any("does not match" in problem for problem in check_scene(broken, rendering))
    # An out-of-canvas element is reported.
    rendering.elements[0] = type(rendering.elements[0])("title", (-5, 0, 10, 10), "x")
    assert any("outside the canvas" in problem for problem in check_scene(scene, rendering))


def test_rejections_carry_reasons():
    reasons = [build_pair_report("chart", seed)[1] for seed in range(120)]
    assert any(reasons), "some candidates are rejected"
    assert all(isinstance(reason, str) for group in reasons for reason in group)


def test_bar_pixels_and_edit_scale():
    assert bar_pixels(50, 100, 400) == 200 and bar_pixels(1, 1000, 100) == 1
    assert edit_scale(400, (1000, 1000), (500, 500)) == pytest.approx(10.0)
    assert edit_scale(400, (1000, 1000), (1000, 1000)) == pytest.approx(20.0)


def test_pools_are_disjoint_balanced_and_reproducible(tmp_path):
    seeds = set()
    for name in POOLS:
        records = generate_pool(name, 12)
        assert [r["pair_id"] for r in records] == [f"{name}-{k:05d}" for k in range(12)]
        assert {r["family"] for r in records[:3]} == set(FAMILIES)
        assert all(pool_of_seed(r["seed"]) == name for r in records)
        pool_seeds = {r["seed"] for r in records}
        assert not seeds & pool_seeds
        seeds |= pool_seeds
    test = generate_pool("test", 12)
    assert sum(r["rendering"] == "heldout" for r in test) == 6
    assert all(r["rendering"] == "familiar" for r in generate_pool("train", 12))
    assert generate_pool("dev", 9, workers=2) == generate_pool("dev", 9)
    path = write_manifest(test, tmp_path / "test.jsonl.gz")
    assert read_manifest(path) == test and read_manifest(path, limit=5) == test[:5]
    pair = pair_from_record(test[4])
    assert list(pair.answers) == test[4]["answers"]
    tampered = dict(test[4], question="?")
    with pytest.raises(ValueError):
        pair_from_record(tampered)


def test_pool_sizes_match_the_paper():
    sizes = {name: spec.size for name, spec in POOLS.items()}
    assert sizes == {
        "reader_fit": 10_000, "tune": 5_000, "dev": 2_000, "train": 50_000, "test": 4_800,
        "prelim_train": 2_000, "prelim_test": 4_800,
    }  # fmt: skip
    assert POOLS["test"].quota(0) == 800 and len(POOLS["test"].strata) == 6
