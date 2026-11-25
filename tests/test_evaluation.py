import numpy as np
import pytest

from rat.evaluation.benchmarks import (
    BENCHMARKS,
    anls,
    anls_score,
    circular_accuracy,
    levenshtein,
    normalize_vqa,
    ocrbench_total,
    parse_choice,
    pope_f1,
    relaxed_accuracy,
    relaxed_correct,
    vqa_score,
)
from rat.evaluation.stats import mcnemar_exact, paired_bootstrap
from rat.selection.baselines import (
    budget_count,
    clipscore,
    coincide,
    d2_pruning,
    dataprophet,
    dose,
    kmeans,
    pair_average,
    select_top,
    xmas,
)


def test_relaxed_accuracy():
    assert relaxed_correct("104", "100") and not relaxed_correct("106", "100")
    assert relaxed_correct("0.52", "50%") and relaxed_correct("1,000", "1000")
    assert relaxed_correct("Canada", "canada") and not relaxed_correct("Canada", "Mexico")
    assert relaxed_correct("0", "0") and not relaxed_correct("0.1", "0")
    assert relaxed_accuracy(["104", "x"], ["100", "y"]) == 50.0


def test_anls():
    assert levenshtein("kitten", "sitting") == 3
    assert anls_score("Hello World", ["hello world"]) == 1.0
    assert anls_score("hello wurld", ["hello world"]) == pytest.approx(1 - 1 / 11)
    assert anls_score("completely different", ["hello world"]) == 0.0
    assert anls_score("b", ["a", "b"]) == 1.0
    assert anls(["hello world", "x"], [["hello world"], ["yyyy"]]) == 50.0


def test_vqa_consensus_accuracy():
    assert normalize_vqa("The Two dogs.") == "2 dogs"
    answers = ["stop"] * 2 + ["go"] * 8
    assert vqa_score("stop", answers) == pytest.approx((2 * (1 / 3) + 8 * (2 / 3)) / 10)
    assert vqa_score("go", answers) == 1.0 and vqa_score("yield", answers) == 0.0


def test_ocrbench_pope_and_mmbench():
    assert ocrbench_total(["The sign says OPEN", "closed"], [["open"], ["open"]]) == 1
    assert pope_f1(
        ["Yes, there is.", "No.", "yes", "It is not present"], ["yes", "no", "no", "no"]
    ) == pytest.approx(100 * 2 / 3)
    assert (
        parse_choice("B. the red one") == "B"
        and parse_choice("(C)") == "C"
        and parse_choice("Because") is None
    )
    assert circular_accuracy(["A", "B", "C", "D"], ["A", "B", "C", "A"], [0, 0, 1, 1]) == 50.0
    assert len(BENCHMARKS) == 10 and sum(b.core for b in BENCHMARKS.values()) == 7
    assert (
        BENCHMARKS["chartqapro"].max_new_tokens == 512
        and BENCHMARKS["docvqa"].max_new_tokens == 128
    )


def test_paired_statistics():
    rng = np.random.default_rng(0)
    base = rng.random(2000) < 0.6
    better = base | (rng.random(2000) < 0.1)
    interval = paired_bootstrap([better], [base], samples=2000)
    assert interval["low"] > 0 and interval["low"] < interval["difference"] < interval["high"]
    same = paired_bootstrap([base, base], [base, base], samples=500)
    assert same["difference"] == same["low"] == same["high"] == 0.0
    test = mcnemar_exact(better, base)
    assert test["only_second"] == 0 and test["p_value"] < 1e-6
    assert mcnemar_exact(base, base)["p_value"] == 1.0
    assert (
        mcnemar_exact(np.array([1, 0, 1, 0], bool), np.array([0, 1, 1, 0], bool))["p_value"] == 1.0
    )


def test_selectors_return_valid_subsets():
    rng = np.random.default_rng(1)
    centres = rng.normal(size=(4, 16))
    features = np.concatenate([c + 0.1 * rng.normal(size=(30, 16)) for c in centres])
    count = budget_count(len(features), 0.2)
    assert count == 24
    difficulty = rng.random(len(features))
    for chosen in (
        select_top(difficulty, count),
        d2_pruning(features, difficulty, count),
        coincide(features, count, clusters=4, seed=0),
        xmas(features.reshape(len(features), 4, 4), count, clusters=4, seed=0),
        dose(difficulty, rng.random(len(features)), count, seed=0),
        select_top(dataprophet(features, features[:10], 1 + difficulty), count),
    ):
        assert len(chosen) == count == len(set(chosen.tolist()))
        assert chosen.min() >= 0 and chosen.max() < len(features) and np.all(np.diff(chosen) > 0)
    # Balanced and diversity-aware selectors cover every cluster.
    labels = np.repeat(np.arange(4), 30)
    assert set(labels[xmas(features.reshape(len(features), 4, 4), 24, clusters=4, seed=0)]) == {
        0,
        1,
        2,
        3,
    }
    assert set(labels[d2_pruning(features, np.ones(len(features)), 24)]) == {0, 1, 2, 3}
    assert len(np.unique(kmeans(features, 4, seed=0))) == 4


def test_selector_building_blocks():
    assert pair_average(np.array([[1.0, 3.0], [2.0, 2.0]])).tolist() == [2.0, 2.0]
    with pytest.raises(ValueError):
        pair_average(np.ones((3, 3)))
    image, text = np.array([[1.0, 0.0], [1.0, 0.0]]), np.array([[2.0, 0.0], [-1.0, 0.0]])
    assert clipscore(image, text).tolist() == [2.5, 0.0]
    assert select_top(np.array([0.1, 0.9, 0.9, 0.5]), 2).tolist() == [1, 2]
    with pytest.raises(ValueError):
        budget_count(10, 0.0)
