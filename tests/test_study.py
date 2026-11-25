import json

import pytest
import torch

from rat.cli import main as cli_main
from rat.config import apply_override, deep_update, load_config
from rat.demo import run as run_demo
from rat.experiments import Study, save_result
from rat.features import FeatureStore, open_pool
from rat.routing import RoutingRecord


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("study")
    config = load_config("configs/base.yaml")
    deep_update(
        config,
        {
            "backbone": {"family": "tiny"},
            "data": {
                "reader_fit": 36,
                "tune": 18,
                "dev": 9,
                "train": 45,
                "test": 18,
                "prelim_train": 9,
                "prelim_test": 9,
            },
            "cache_dir": str(root / "cache"),
            "reader": {"epochs": 2, "batch_pairs": 12, "micro_pairs": 12},
            # The random tiny interface gives margins close to zero; the grid is scaled to them.
            "routing": {"thresholds": [1e-7, 1e-5], "trial_fraction": 0.05},
            "student": {"batch_pairs": 4, "pairs_per_forward": 2, "learning_rate": 1e-3},
            "evaluation": {"max_new_tokens": 4, "batch_examples": 8},
            "budget": {"accounting": "modelled", "devices": 1, "reference_hours": 0.02},
            "lora": {"rank": 4, "alpha": 8, "dropout": 0.0},
        },
    )
    return Study(config, root / "run", log=lambda message: None)


def test_config_overrides(tmp_path):
    config = load_config(
        "configs/backbones/qwen2.5-vl-3b.yaml", ["student.learning_rate=1e-4", "data.train=100"]
    )
    assert config["backbone"]["model_id"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert config["budget"]["reference_hours"] == 106 and config["reader"]["epochs"] == 10
    assert config["student"]["learning_rate"] == 1e-4 and config["data"]["train"] == 100
    assert config["routing"]["thresholds"] == pytest.approx(
        [0.693147, 1.386294, 2.197225], abs=1e-6
    )
    with pytest.raises(ValueError):
        apply_override(config, "no-equals-sign")
    laptop = load_config("configs/laptop/smolvlm_256m.yaml")
    assert laptop["backbone"]["family"] == "smolvlm" and laptop["lora"]["rank"] == 16


def _synthetic_margins(count: int) -> torch.Tensor:
    """Branch margins with a third of the pairs passing at L, a third only at H, a third never."""
    margins = torch.full((count, 2, 2), -1.0)
    margins[0::3, 0] = 3.0
    margins[1::3, 1] = 3.0
    return margins


def test_rat_and_matched_controls(study, tmp_path):
    # A random reader on the tiny interface never passes both branches (near-identical inputs
    # give opposite margins), so the tables of the fitted reader are replaced by known ones.
    entry = study.reader(17)
    entry["tune"], entry["train"] = _synthetic_margins(18), _synthetic_margins(45)
    rat, records = study.run_gated(17)
    assert rat.threshold in (1e-7, 1e-5) and len(rat.extra["trials"]) == 2
    assert rat.extra["trials"][0]["retained_percent"] == pytest.approx(100 * 12 / 18)
    assert rat.extra["trials"][0]["top_mode_percent"] == pytest.approx(50.0)
    assert rat.routing["retained"] == rat.train["pairs"] > 0
    assert rat.routing["excluded"] > 0 and 0 < rat.train["top_mode_share"] < 1
    assert rat.costs["total"] <= rat.costs["budget"]
    assert rat.costs["preparation"] > 0 and rat.costs["routing"] > 0 and rat.costs["training"] > 0
    assert {"pair", "single", "collapse", "heldout", "worst"} <= set(rat.metrics)
    assert len(rat.extra["predictions"]) == 18
    controls = {
        r.name: r
        for r in study.controls_on_retained(
            17, rat, records, ["filter_only", "random_alloc", "equal_random", "equal_loss"]
        )
    }
    pairs = sorted(pair for pair, _mode in rat.assignments)
    # Random allocation keeps the pair identities, the number of H assignments and the tokens.
    shuffled = controls["random_alloc"]
    assert sorted(pair for pair, _ in shuffled.assignments) == pairs
    assert shuffled.train["visual_tokens"] == rat.train["visual_tokens"]
    assert sum(mode for _, mode in shuffled.assignments) == sum(mode for _, mode in rat.assignments)
    # Filtering-only trains the same pairs at H.
    filtered = controls["filter_only"]
    assert sorted(pair for pair, _ in filtered.assignments) == pairs
    assert all(mode == 1 for _, mode in filtered.assignments)
    assert filtered.train["visual_tokens"] == rat.train["same_pair_top_tokens"]
    assert filtered.costs["routing"] >= rat.costs["routing"]
    for name in ("equal_random", "equal_loss"):
        assert controls[name].train["pairs"] == rat.train["pairs"]
        assert controls[name].train["top_mode_share"] == 1.0
    assert controls["equal_loss"].costs["routing"] > controls["equal_random"].costs["routing"]
    # Results round-trip through JSON and the routing records can be rebuilt from them.
    stored = json.loads(save_result(rat, tmp_path).read_text())
    rebuilt = [
        RoutingRecord(r["index"], r["status"], r["mode"])
        for r in stored["extra"]["routing_records"]
    ]
    assert len(rebuilt) == 45 and stored["routing"]["candidates"] == 45
    allocation = study.run_allocation_only(17, rat.threshold)
    assert allocation.train["pairs"] == 45 and allocation.routing["excluded"] == 0


def test_uniform_baselines_and_references(study):
    for mode in (0, 1):
        budgeted = study.run_random(17, mode)
        assert budgeted.costs["total"] <= budgeted.costs["budget"]
        assert (
            budgeted.extra["retention_fraction"] in (0.2, 0.5, 1.0)
            and len(budgeted.extra["trials"]) == 3
        )
        assert budgeted.train["top_mode_share"] == float(mode)
        full = study.run_random(17, mode, budgeted=False)
        assert full.train["pairs"] == 45 and full.name == f"full_{'LH'[mode]}"
    unadapted = study.run_unadapted(17)
    assert unadapted.costs["total"] == 0.0 and unadapted.train == {}


def test_score_variants_and_seeds(study):
    mean = study.run_gated(29, "mean_margin", rule="mean")[0]
    student = study.run_gated(29, "student_margin", source="student")[0]
    assert mean.seed == student.seed == 29 and len(student.extra["trials"]) == 2
    assert set(study._readers) == {17, 29}


def test_preliminary_conditions(study):
    runs = {c: study.run_preliminary(43, c) for c in ("L1", "L4", "U1", "H1")}
    assert runs["L4"].train["pairs"] == 4 * runs["L1"].train["pairs"] == 36
    assert runs["L4"].train["visual_tokens"] == 4 * runs["L1"].train["visual_tokens"]
    # The sequence-length control has the token grid of H.
    assert (
        runs["U1"].train["visual_tokens"]
        == runs["H1"].train["visual_tokens"]
        > runs["L1"].train["visual_tokens"]
    )
    assert runs["H1"].metrics["pairs"] == 9


def test_feature_store_round_trip(tiny_backbone, tmp_path):
    from rat.data.splits import generate_pool

    records = generate_pool("dev", 4)
    pool = open_pool(tiny_backbone, records, tmp_path, "dev")
    again = open_pool(tiny_backbone, records, tmp_path, "dev", build=False)
    store = pool.stores[1]
    assert isinstance(store, FeatureStore) and len(store) == 4 and store.meta["mode"] == "H"
    assert store.count(2, 0) == pool.costs(2)[1] // 2
    assert store.tokens(2, 1).shape == (store.count(2, 1), tiny_backbone.visual_dim)
    assert (again.stores[0].tokens(1, 0) == pool.stores[0].tokens(1, 0)).all()
    assert pool.extraction_seconds(0, 1) > 0 and pool.lengths(0, 1)[0] > pool.lengths(0, 0)[0] - 1
    # Different images of a pair give different tokens.
    assert not (store.tokens(0, 0) == store.tokens(0, 1)).all()
    with pytest.raises(FileNotFoundError):
        open_pool(tiny_backbone, records[:2], tmp_path, "dev", build=False)


def test_quick_start_and_cli(capsys):
    result = run_demo(seed=17, fit_pairs=120, held_pairs=120, log=lambda message: None)
    assert result["final_loss"] < result["first_loss"]
    retained = [row["retained_percent"] for row in result["rows"]]
    assert retained == sorted(retained, reverse=True) and retained[0] > 0
    cli_main(["geometry", "--width", "1456", "--height", "896"])
    output = capsys.readouterr().out
    assert "240 tokens" in output and "1000 tokens" in output and "576 tokens" in output
    with pytest.raises(SystemExit) as status:
        cli_main(["check-paper"])
    assert status.value.code == 0
