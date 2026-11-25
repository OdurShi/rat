import math

import numpy as np
import pytest
import torch

from rat.backbones.base import Example
from rat.budget import CostLedger, CostModel
from rat.evaluation.paired import (
    accuracy_difference,
    branch_correct,
    collapsed,
    mean_sd,
    pair_correct,
    paired_metrics,
)
from rat.pipeline import (
    Accounting,
    evaluate,
    fixed_permutation,
    pair_scores,
    prepare_reader,
    reader_margin_table,
    reference_budget,
    route_and_train,
    route_pool,
    run_pilot,
    select_threshold,
    token_saving_percent,
    training_assignments,
)
from rat.reader import ReaderFitConfig
from rat.routing import EXCLUDED, RETAINED, UNPROCESSED, UNRETAINED
from rat.student import (
    TrainConfig,
    answer_losses,
    learning_rate_factor,
    predict,
    student_margins,
    train_pass,
)

MODEL = CostModel(extract_fixed=0.0, extract_per_token=0.01, train_fixed=0.0, train_per_token=0.001)
TRAIN = TrainConfig(learning_rate=1e-3, batch_pairs=4, pairs_per_forward=2)


def _accounting():
    return Accounting(MODEL, "modelled", 1)


def _positions(size, seed=0):
    return np.argsort(fixed_permutation(size, seed, 2)).tolist()


def test_learning_rate_schedule():
    factors = [learning_rate_factor(k, 100, 0.03) for k in range(100)]
    assert factors[:3] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert factors[3] == pytest.approx(1.0) and factors[-1] < 0.01
    assert all(a >= b for a, b in zip(factors[3:], factors[4:], strict=False))
    assert learning_rate_factor(0, 1, 0.03) == 1.0


def test_paired_metrics_by_hand():
    answers = [("16.00", "18.00"), ("3", "2"), ("57859", "57858"), ("A", "B")]
    predictions = [("16.00", " 18.00 "), ("3", "3"), ("57859", "57859"), ("B", "A")]
    assert branch_correct(predictions, answers).tolist() == [
        [True, True],
        [True, False],
        [True, False],
        [False, False],
    ]
    assert pair_correct(predictions, answers).tolist() == [True, False, False, False]
    assert collapsed(predictions).tolist() == [False, True, True, False]
    records = [
        {"family": "card", "rendering": "familiar"}, {"family": "chart", "rendering": "familiar"},
        {"family": "card", "rendering": "heldout"}, {"family": "chart", "rendering": "heldout"},
    ]  # fmt: skip
    metrics = paired_metrics(predictions, answers, records)
    assert metrics["pair"] == 25.0 and metrics["single"] == 50.0 and metrics["collapse"] == 50.0
    assert metrics["familiar"] == 50.0 and metrics["heldout"] == 0.0 and metrics["worst"] == 0.0
    assert metrics["groups"]["card/familiar"] == 100.0 and len(metrics["groups"]) == 4


def test_accuracy_difference_is_averaged_within_runs():
    first = [np.array([1, 1, 0, 0], bool), np.array([1, 1, 1, 0], bool)]
    second = [np.array([1, 0, 0, 0], bool), np.array([1, 0, 0, 0], bool)]
    assert accuracy_difference(first, second) == pytest.approx(100 * (0.25 + 0.5) / 2)
    members = np.array([True, True, False, False])
    assert accuracy_difference(first, second, members) == pytest.approx(50.0)
    assert mean_sd([1.0, 2.0, 3.0]) == pytest.approx((2.0, 1.0)) and mean_sd([4.0]) == (4.0, 0.0)


def test_answer_loss_masks_everything_but_the_answer(tiny_backbone, tiny_pools):
    pool = tiny_pools["train"]
    example = pool.example(0, 0, 1)
    nll = tiny_backbone.answer_nll([example])
    other_question = Example(example.tokens, example.grid, example.question, example.answer + "7")
    longer = tiny_backbone.answer_nll([other_question])
    assert nll.shape == (1,) and float(nll.detach()) > 0
    assert float(longer.detach()) > float(nll.detach())  # one more answer token is scored
    batch = tiny_backbone.answer_nll([pool.example(0, 0, 1), pool.example(1, 1, 0)])
    assert float(batch[0].detach()) == pytest.approx(float(nll.detach()), rel=1e-4)
    assert (
        tiny_backbone.sequence_length(example.tokens.shape[0], example.question, example.answer)
        > example.tokens.shape[0]
    )


def test_adapter_starts_at_the_base_model_and_resets(tiny_backbone, tiny_pools):
    pool = tiny_pools["train"]
    examples = [pool.example(0, 0, 1), pool.example(0, 1, 1)]
    start = tiny_backbone.answer_nll(examples).detach()
    tiny_backbone.set_adapter_enabled(False)
    assert torch.allclose(tiny_backbone.answer_nll(examples).detach(), start, atol=1e-5)
    tiny_backbone.set_adapter_enabled(True)
    log = train_pass(
        tiny_backbone,
        pool,
        [(0, 1), (1, 1), (2, 0), (3, 0)] * 6,
        TrainConfig(learning_rate=5e-3, batch_pairs=4, pairs_per_forward=2),
    )
    assert log.updates == 6 and log.pairs == 24 and log.losses[-1] < log.losses[0]
    assert not torch.allclose(tiny_backbone.answer_nll(examples).detach(), start, atol=1e-4)
    tiny_backbone.reset_adapter()
    assert torch.allclose(tiny_backbone.answer_nll(examples).detach(), start, atol=1e-5)


def test_empty_retained_set_returns_the_initial_student(tiny_backbone, tiny_pools):
    before = tiny_backbone.adapter_state()
    log = train_pass(tiny_backbone, tiny_pools["train"], [], TRAIN)
    assert log.updates == 0 and log.pairs == 0
    after = tiny_backbone.adapter_state()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_training_stops_before_the_budget_is_exceeded(tiny_backbone, tiny_pools):
    pool = tiny_pools["train"]
    assignments = [(i, 1) for i in range(12)]
    full = sum(
        MODEL.chunk([n for i, m in assignments[k : k + 2] for n in pool.lengths(i, m)])
        for k in range(0, 12, 2)
    )
    ledger = CostLedger(budget=0.55 * full)
    log = train_pass(tiny_backbone, pool, assignments, TRAIN, ledger, MODEL, "modelled")
    assert log.stopped_by_budget and 0 < log.updates < 3
    assert ledger.total <= ledger.budget and ledger.training == pytest.approx(log.charged)
    assert log.visual_tokens == sum(pool.costs(i)[1] for i, _ in log.trained)
    assert log.mode_pairs == {1: log.pairs}


def test_route_pool_follows_algorithm_s1(tiny_pools):
    pool = tiny_pools["train"]
    count = len(pool)
    scores = torch.full((count, 2), -1.0)
    scores[0::3, 0] = 5.0  # pass at L
    scores[1::3, 1] = 5.0  # fail at L, pass at H
    order = fixed_permutation(count, 0, 1)
    ledger = CostLedger(float("inf"))
    records = route_pool(
        pool, scores, math.log(4), order, _positions(count), ledger, _accounting(), TRAIN
    )
    by_index = {r.index: r for r in records}
    assert [r.index for r in records] == order
    for index in range(count):
        record = by_index[index]
        costs = pool.costs(index)
        if index % 3 == 0:
            assert (record.status, record.mode) == (RETAINED, 0) and set(record.margins) == {0}
        elif index % 3 == 1:
            # A pair whose modes cost the same is still read at L first (ties favour L).
            assert (record.status, record.mode) == (RETAINED, 1) and set(record.margins) == {0, 1}
        else:
            assert record.status == EXCLUDED and record.mode is None
        assert record.costs == costs
    expected = sum(
        MODEL.readout(pool.costs(i)[0]) + (MODEL.readout(pool.costs(i)[1]) if i % 3 else 0.0)
        for i in range(count)
    )
    assert ledger.routing == pytest.approx(expected) and ledger.training == 0.0
    assignments = training_assignments(records, _positions(count))
    assert len(assignments) == sum(i % 3 != 2 for i in range(count))
    positions = _positions(count)
    assert [positions[i] for i, _ in assignments] == sorted(positions[i] for i, _ in assignments)
    fallback = route_pool(
        pool,
        scores,
        math.log(4),
        order,
        _positions(count),
        CostLedger(float("inf")),
        _accounting(),
        TRAIN,
        fallback_mode=1,
    )
    assert all(r.status == RETAINED for r in fallback)
    assert sum(r.mode == 1 for r in fallback) == sum(i % 3 != 0 for i in range(count))


def test_route_pool_reserves_the_training_pass(tiny_pools):
    pool = tiny_pools["train"]
    count = len(pool)
    scores = torch.full((count, 2), 5.0)
    order = fixed_permutation(count, 0, 1)
    unlimited = CostLedger(float("inf"))
    route_pool(pool, scores, 1.0, order, _positions(count), unlimited, _accounting(), TRAIN)
    budget = 0.5 * (
        unlimited.routing
        + sum(MODEL.chunk(list(pool.lengths(i, 0)) * 2) for i in range(0, count, 2))
    )
    ledger = CostLedger(budget)
    records = route_pool(pool, scores, 1.0, order, _positions(count), ledger, _accounting(), TRAIN)
    statuses = [r.status for r in records]
    retained = statuses.count(RETAINED)
    assert 0 < retained < count
    assert statuses.count(UNRETAINED) + statuses.count(UNPROCESSED) == count - retained
    # Everything after the stop is unprocessed, and the reserved pass still fits.
    first_stop = next(k for k, s in enumerate(statuses) if s != RETAINED)
    assert all(s == UNPROCESSED for s in statuses[first_stop + 1 :])
    assignments = training_assignments(records, _positions(count))
    log_ledger = CostLedger(budget, routing=ledger.routing)
    from rat.budget import TrainingCostEstimator

    estimator = TrainingCostEstimator(MODEL, TRAIN.batch_pairs, TRAIN.pairs_per_forward)
    assert log_ledger.affordable(estimator.pass_cost([pool.lengths(i, m) for i, m in assignments]))


def test_full_pipeline_on_the_tiny_backbone(tiny_backbone, tiny_pools):
    fit, tune, dev, train, test = (
        tiny_pools[k] for k in ("reader_fit", "tune", "dev", "train", "test")
    )
    model, pilot_seconds = run_pilot(tiny_backbone, train, TRAIN, list(range(8)), max_new_tokens=4)
    assert model.train_per_token > 0 and model.extract_per_token >= 0 and pilot_seconds > 0
    accounting = Accounting(model, "modelled", 1)
    budget = reference_budget(train, model, TRAIN)
    assert budget > 0
    ledger = CostLedger(float("inf"))
    reader, info = prepare_reader(
        tiny_backbone,
        fit,
        ReaderFitConfig(epochs=2, batch_pairs=8, micro_pairs=8),
        17,
        ledger,
        accounting,
    )
    assert info["steps"] == 6 and ledger.preparation > 0
    tune_scores = pair_scores(reader_margin_table(reader, tune))
    train_margins = reader_margin_table(reader, train)
    assert train_margins.shape == (len(train), 2, 2)
    assert torch.equal(pair_scores(train_margins, "mean"), train_margins.mean(-1))
    # A threshold grid around the observed scores makes the gate non-trivial for a random reader.
    grid = sorted(
        {
            float(q)
            for q in torch.quantile(tune_scores.max(1).values, torch.tensor([0.3, 0.6]))
            if q > 0
        }
        | {1e-6}
    )
    threshold, trials = select_threshold(
        tiny_backbone,
        tune,
        tune_scores,
        dev,
        grid,
        10 * budget,
        accounting,
        TRAIN,
        17,
        batch_examples=8,
        max_new_tokens=4,
    )
    assert threshold in grid and len(trials) == len(grid)
    assert trials[0]["retained_percent"] >= trials[-1]["retained_percent"]
    run_ledger = CostLedger(10 * budget)
    result, records = route_and_train(
        tiny_backbone,
        train,
        pair_scores(train_margins),
        threshold,
        run_ledger,
        accounting,
        TRAIN,
        17,
    )
    assert result.routing["candidates"] == len(train)
    assert result.train["pairs"] == result.routing["retained"] == len(result.assignments)
    assert result.costs["total"] == pytest.approx(run_ledger.total)
    assert run_ledger.total <= run_ledger.budget
    assert 0.0 <= token_saving_percent(result.assignments, train) <= 100.0
    metrics, predictions, _seconds = evaluate(
        tiny_backbone, test, batch_examples=8, max_new_tokens=4
    )
    assert len(predictions) == len(test) and {
        "pair",
        "single",
        "collapse",
        "heldout",
        "worst",
    } <= set(metrics)


def test_student_scores(tiny_backbone, tiny_pools):
    pool = tiny_pools["train"]
    losses = answer_losses(tiny_backbone, pool, 1, [0, 1, 2], batch_examples=4)
    assert losses.shape == (3,) and bool((losses > 0).all())
    margins, seconds = student_margins(tiny_backbone, pool, 0, [0, 1, 2], batch_examples=4)
    assert margins.shape == (3, 2) and seconds.shape == (3,)
    example = pool.example(0, 0, 0)
    swapped = Example(example.tokens, example.grid, example.question, pool.answers[0][1])
    expected = tiny_backbone.answer_nll([swapped]) - tiny_backbone.answer_nll([example])
    assert float(margins[0, 0]) == pytest.approx(float(expected.detach()), abs=1e-4)
    answers, _ = predict(tiny_backbone, pool, 1, [0, 1], batch_examples=3, max_new_tokens=3)
    assert len(answers) == 2 and all(len(a) == 2 for a in answers)


def test_restricted_candidates_are_the_only_pairs_read(tiny_backbone, tiny_pools):
    pool = tiny_pools["train"]
    scores = torch.full((len(pool), 2), float("-inf"))
    scores[:, 1] = float("inf")
    subset = fixed_permutation(len(pool), 17, 1)[:10]
    ledger = CostLedger(float("inf"))
    result, records = route_and_train(
        tiny_backbone,
        pool,
        scores,
        1.0,
        ledger,
        _accounting(),
        TRAIN,
        17,
        candidates=subset,
        allowed_modes=[1],
    )
    assert [r.index for r in records] == subset and result.routing["candidates"] == 10
    assert result.train["pairs"] == 10 and result.train["top_mode_share"] == 1.0
    assert ledger.routing == pytest.approx(sum(MODEL.readout(pool.costs(i)[1]) for i in subset))
