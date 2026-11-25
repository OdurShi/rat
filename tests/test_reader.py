import math

import pytest
import torch

from rat.interface.geometry import TokenGrid, qwen_grid
from rat.margins import (
    PAPER_THRESHOLDS,
    branch_margins,
    decompose,
    mean_margin,
    min_margin,
    passes,
    probability_from_threshold,
    threshold_from_probability,
)
from rat.reader import (
    CandidateReader,
    ReaderFitConfig,
    fit_reader,
    pad_inputs,
    reader_input,
    reader_loss,
    score_pairs,
)


def _grid(tokens: int) -> TokenGrid:
    return TokenGrid(((1, tokens, -1.0, -1.0, 1.0, 1.0),), (28 * tokens, 28))


def test_equation_1_by_hand():
    torch.manual_seed(0)
    reader = CandidateReader(visual_dim=3, text_dim=4, gain=1.0, seed=1)
    z = torch.randn(5, 5)
    t = torch.randn(4)
    alpha = torch.softmax(z @ reader.attention @ t / math.sqrt(5), dim=0)
    expected = alpha @ (z @ reader.readout @ t) / math.sqrt(5)
    score = reader(z[None], torch.ones(1, 5, dtype=torch.bool), t[None, None])
    assert score.shape == (1, 1)
    assert float(score.detach()) == pytest.approx(float(expected.detach()), rel=1e-5)
    weights = reader.attention_weights(z[None], torch.ones(1, 5, dtype=torch.bool), t[None, None])
    assert float(weights.detach().sum()) == pytest.approx(1.0)


def test_padding_does_not_change_scores():
    reader = CandidateReader(4, 6, gain=1.0, seed=3)
    short, long = torch.randn(3, 6), torch.randn(7, 6)
    candidates = torch.randn(2, 2, 6)
    batch, mask = pad_inputs([short, long])
    together = reader(batch, mask, candidates)
    alone = reader(short[None], torch.ones(1, 3, dtype=torch.bool), candidates[:1])
    assert torch.allclose(together[0], alone[0], atol=1e-5)


def test_initial_scores_are_small_and_parameters_shared():
    reader = CandidateReader(16, 16, seed=0)
    assert {name for name, _ in reader.named_parameters()} == {"attention", "readout"}
    assert reader.attention.shape == (18, 16)
    assert float(reader.attention.detach().abs().max()) < 0.01
    again = CandidateReader(16, 16, seed=0)
    assert torch.equal(reader.readout, again.readout)


def test_reader_input_appends_coordinates():
    grid = qwen_grid((84, 56), 1024)
    z = reader_input(torch.ones(6, 5), grid)
    assert z.shape == (6, 7) and torch.all(z[:, :5] == 1)
    assert z[0, 5:].tolist() == pytest.approx([-2 / 3, -0.5])
    with pytest.raises(ValueError):
        reader_input(torch.ones(5, 5), grid)


def test_loss_is_binary_cross_entropy_of_the_correct_candidate():
    scores = torch.tensor([[[2.0, 0.5], [-1.0, 3.0]]])
    expected = (math.log1p(math.exp(0.5 - 2.0)) + math.log1p(math.exp(-1.0 - 3.0))) / 2
    assert float(reader_loss(scores)) == pytest.approx(expected)
    assert float(reader_loss(torch.zeros(4, 2, 2, 2))) == pytest.approx(math.log(2))


def test_margins_and_decomposition():
    scores = torch.tensor([[2.0, 0.5], [-1.0, 3.0]])
    assert branch_margins(scores).tolist() == [1.5, 4.0]
    assert float(min_margin(scores)) == 1.5 and float(mean_margin(scores)) == 2.75
    delta, beta, minimum = decompose(scores)
    assert float(delta) == 2.75 and float(beta) == -1.25
    assert float(minimum) == pytest.approx(float(delta) - abs(float(beta))) == 1.5
    # Margins (10, -1) have mean 4.5 and fail because their minimum is -1 (Appendix C.1).
    unbalanced = torch.tensor([[10.0, 0.0], [1.0, 0.0]])
    assert branch_margins(unbalanced).tolist() == [10.0, -1.0]
    assert float(mean_margin(unbalanced)) == 4.5 and not bool(
        passes(unbalanced, PAPER_THRESHOLDS[0])
    )
    with pytest.raises(ValueError):
        passes(scores, 0.0)


def test_margin_invariances():
    torch.manual_seed(1)
    scores = torch.randn(7, 2, 2)
    shifted = scores + torch.randn(7, 2, 1)  # a constant added to either row
    assert torch.allclose(branch_margins(shifted), branch_margins(scores), atol=1e-6)
    swapped = scores.flip(-2).flip(-1)  # exchange branches together with their labels
    assert torch.allclose(min_margin(swapped), min_margin(scores))
    assert torch.allclose(branch_margins(swapped), branch_margins(scores).flip(-1))


def test_identical_inputs_fail_every_positive_threshold():
    reader = CandidateReader(4, 5, gain=1.0, seed=2)
    tokens = torch.randn(6, 4)
    scores = reader.score_matrix([tokens, tokens], [_grid(6), _grid(6)], torch.randn(2, 5))
    margins = branch_margins(scores)
    assert float(margins[0]) == pytest.approx(-float(margins[1]), abs=1e-5)
    assert float(min_margin(scores)) <= 0


def test_candidate_reordering_permutes_columns():
    reader = CandidateReader(4, 5, gain=1.0, seed=2)
    tokens = [torch.randn(6, 4), torch.randn(6, 4)]
    grids = [_grid(6), _grid(6)]
    candidates = torch.randn(2, 5)
    scores = reader.score_matrix(tokens, grids, candidates)
    reordered = reader.score_matrix(tokens, grids, candidates.flip(0))
    assert torch.allclose(reordered, scores.flip(-1), atol=1e-5)


def test_thresholds_are_probabilities_of_the_weaker_branch():
    assert [round(probability_from_threshold(g), 6) for g in PAPER_THRESHOLDS] == [
        pytest.approx(2 / 3, abs=1e-6), pytest.approx(0.8, abs=1e-6), pytest.approx(0.9, abs=1e-6),
    ]  # fmt: skip
    assert threshold_from_probability(0.8) == pytest.approx(math.log(4))
    with pytest.raises(ValueError):
        threshold_from_probability(0.5)


def _toy_problem(pairs: int, noise: float, seed: int):
    """Tokens that carry the answer direction of the correct candidate in one position."""
    generator = torch.Generator().manual_seed(seed)
    dim = 8
    candidates = torch.randn(pairs, 2, dim, generator=generator)
    tokens = {}
    for index in range(pairs):
        for mode, strength in enumerate((0.3, 1.5)):
            both = []
            for branch in range(2):
                block = noise * torch.randn(5, dim, generator=generator)
                block[int(torch.randint(5, (1,), generator=generator))] += (
                    strength * candidates[index, branch]
                )
                both.append(block)
            tokens[index, mode] = both

    def fetch(index, mode):
        return tokens[index, mode], [_grid(5), _grid(5)]

    return fetch, candidates


def test_fit_reader_learns_and_generalises_on_a_toy_interface():
    fetch, candidates = _toy_problem(96, noise=0.3, seed=0)
    reader = CandidateReader(8, 8, seed=0)
    config = ReaderFitConfig(epochs=30, batch_pairs=16, micro_pairs=8, learning_rate=3e-3)
    log = fit_reader(reader, fetch, candidates, 2, config, seed=0)
    assert log.steps == 30 * 6 and log.losses[-1] < 0.8 * log.losses[0]
    assert not any(p.requires_grad is False for p in reader.parameters())
    held, held_candidates = _toy_problem(64, noise=0.3, seed=1)
    low = min_margin(score_pairs(reader, held, held_candidates, range(64), 0))
    high = min_margin(score_pairs(reader, held, held_candidates, range(64), 1))
    # More evidence in the high mode gives larger margins and more passes.
    assert float((high > 0).float().mean()) > 0.9
    assert float(high.median()) > float(low.median())


def test_standardisation_is_identity_by_default_and_rescales_when_set():
    reader = CandidateReader(4, 5, gain=1.0, seed=2)
    z, mask, t = torch.randn(2, 6, 6), torch.ones(2, 6, dtype=torch.bool), torch.randn(2, 2, 5)
    plain = reader(z, mask, t)
    reader.set_standardisation(torch.zeros(4), torch.ones(4), torch.zeros(5), torch.ones(5))
    assert torch.allclose(reader(z, mask, t), plain)
    reader.set_standardisation(torch.zeros(4), 2 * torch.ones(4), torch.zeros(5), torch.ones(5))
    scaled = z.clone()
    scaled[..., :4] *= 2
    assert torch.allclose(reader(scaled, mask, t), plain, atol=1e-5)
