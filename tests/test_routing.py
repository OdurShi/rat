import math
import random

import numpy as np
import pytest

from rat.budget import CostLedger, CostModel, TrainingCostEstimator
from rat.routing import (
    RETAINED,
    RoutingRecord,
    lazy_route,
    mode_order,
    permute_assignments_within_strata,
    route,
    routing_summary,
)

GAMMA = math.log(4)


def test_route_picks_the_cheapest_passing_mode():
    assert route([2.0, 3.0], [100, 400], GAMMA) == 0
    assert route([0.5, 3.0], [100, 400], GAMMA) == 1
    assert route([0.5, 1.0], [100, 400], GAMMA) is None
    assert route([GAMMA, 9.0], [100, 400], GAMMA) == 0  # the threshold itself passes
    with pytest.raises(ValueError):
        route([1.0, 1.0], [1, 1], 0.0)


def test_ties_in_cost_favour_the_low_mode():
    assert mode_order([576, 576]) == [0, 1]
    assert route([3.0, 3.0], [576, 576], GAMMA) == 0
    assert route([0.0, 3.0], [576, 576], GAMMA) == 1
    assert mode_order([300, 100, 100]) == [1, 2, 0]


def test_lazy_routing_is_exact_and_stops_early():
    rng = random.Random(0)
    for _ in range(500):
        count = rng.choice([2, 3, 4])
        costs = [rng.choice([64, 128, 256, 256]) for _ in range(count)]
        margins = [rng.uniform(-3, 4) for _ in range(count)]
        calls = []

        def margin_of(mode, margins=margins, calls=calls):
            calls.append(mode)
            return margins[mode]

        decision, evaluated = lazy_route(margin_of, costs, GAMMA)
        assert decision == route(margins, costs, GAMMA)
        assert calls == mode_order(costs)[: len(calls)]
        if decision is not None:
            assert calls[-1] == decision and all(margins[m] < GAMMA for m in calls[:-1])
        assert set(evaluated) == set(calls)


def test_routing_summary_and_token_saving():
    records = [
        RoutingRecord(0, RETAINED, 0, {0: 2.0}, (100, 400)),
        RoutingRecord(1, RETAINED, 1, {0: 0.1, 1: 2.0}, (100, 400)),
        RoutingRecord(2, "excluded", None, {0: 0.1, 1: 0.2}, (100, 400)),
        RoutingRecord(3, "unprocessed", None, {}, (100, 400)),
    ]
    summary = routing_summary(records, top_mode=1)
    assert summary["retained"] == 2 and summary["excluded"] == 1 and summary["unprocessed"] == 1
    assert summary["retained_share"] == pytest.approx(2 / 3)
    assert summary["top_mode_share"] == 0.5
    assert summary["visual_tokens"] == 500 and summary["same_pair_top_tokens"] == 800
    assert summary["token_saving"] == pytest.approx(1 - 500 / 800)


def test_random_allocation_preserves_quota_and_tokens():
    rng = np.random.default_rng(0)
    strata = [(128, 384), (128, 640), (512, 512)]
    costs = [strata[k] for k in rng.integers(0, 3, size=300)]
    modes = [int(m) for m in rng.integers(0, 2, size=300)]
    shuffled = permute_assignments_within_strata(modes, costs, seed=17)
    assert shuffled != modes
    assert sum(shuffled) == sum(modes)
    assert sum(c[m] for c, m in zip(costs, shuffled, strict=True)) == sum(
        c[m] for c, m in zip(costs, modes, strict=True)
    )
    for stratum in strata:
        members = [i for i, c in enumerate(costs) if c == stratum]
        assert sorted(shuffled[i] for i in members) == sorted(modes[i] for i in members)
    assert permute_assignments_within_strata(modes, costs, seed=17) == shuffled
    constant = permute_assignments_within_strata([1, 1, 1], [(1, 2)] * 3, seed=0)
    assert constant == [1, 1, 1]


def test_ledger_implements_the_accounting_rule():
    ledger = CostLedger(budget=100.0)
    ledger.charge("preparation", 30.0, "reader")
    ledger.charge("routing", 20.0)
    assert ledger.total == 50.0 and ledger.remaining == 50.0
    reservation = 40.0
    assert ledger.affordable(10.0 + reservation) and not ledger.affordable(10.1 + reservation)
    ledger.charge("training", 50.0)
    assert ledger.summary(unit=1.0) == {
        "budget": 100.0, "preparation": 30.0, "routing": 20.0, "training": 50.0, "total": 100.0
    }  # fmt: skip
    with pytest.raises(KeyError):
        ledger.charge("other", 1.0)
    with pytest.raises(ValueError):
        ledger.charge("routing", -1.0)


def test_training_cost_estimator_matches_a_direct_computation():
    model = CostModel(train_fixed=0.3, train_per_token=0.01, update_fixed=0.7)
    estimator = TrainingCostEstimator(model, batch_pairs=8, pairs_per_forward=2)
    rng = random.Random(1)
    inserted: dict[int, tuple[int, int]] = {}
    positions = list(range(40))
    rng.shuffle(positions)
    for position in positions[:25]:
        lengths = (rng.randint(80, 400), rng.randint(80, 400))
        preview = estimator.with_pair(position, lengths)
        assert estimator.add(position, lengths) == pytest.approx(preview)
        inserted[position] = lengths
        ordered = [inserted[p] for p in sorted(inserted)]
        # Both examples of a pair are padded to the longest sequence of their chunk.
        expected = 0.0
        for start in range(0, len(ordered), 2):
            chunk = ordered[start : start + 2]
            longest = max(max(pair) for pair in chunk)
            expected += model.train_fixed + model.train_per_token * 2 * len(chunk) * longest
        expected += math.ceil(len(ordered) / 8) * model.update_fixed
        assert estimator.total == pytest.approx(expected)
    assert len(estimator) == 25
    assert estimator.pass_cost([inserted[p] for p in sorted(inserted)]) == pytest.approx(
        estimator.total
    )


def test_cost_model_estimates():
    model = CostModel(extract_fixed=0.1, extract_per_token=0.01, generate_per_token=0.5)
    assert model.readout(600) == pytest.approx(0.2 + 6.0)
    assert model.extraction([100, 300]) == pytest.approx(0.2 + 4.0)
    assert model.generation([10, 20]) == pytest.approx(15.0)
    with pytest.raises(ValueError):
        TrainingCostEstimator(model, batch_pairs=8, pairs_per_forward=3)
