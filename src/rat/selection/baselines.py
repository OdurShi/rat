"""Data selectors compared with RAT (Section 5.1, Appendix E.2).

Every selector works on complete pairs: branch-level scores, features or trajectories are
combined by their arithmetic mean before ranking, graph construction, clustering or
sampling (:func:`pair_average`), and the selected images are then processed by the shared
student interface at ``H``.  The functions take arrays, so the proxy models that produce
them (CLIP, TinyLLaVA-2B, Vicuna-7B, InternVL3-2B in the paper) are interchangeable.

* :func:`clipscore` / :func:`select_top`: CLIPScore ranking (Hessel et al., 2021).
* :func:`d2_pruning`: message passing between difficulty scores on a nearest-neighbour
  graph (Maharana et al., 2024).
* :func:`coincide`: cluster sampling by transferability and density (Lee et al., 2024).
* :func:`xmas`: balanced sampling over clusters of alignment trajectories (Naharas et al., 2026).
* :func:`dose`: sampling from the joint distribution of text-quality and alignment scores
  (Wu et al., 2026).
* :func:`dataprophet`: similarity, perplexity and diversity towards a target pool (Qi et al., 2026).
"""

from __future__ import annotations

import numpy as np

RETENTION_FRACTIONS: tuple[float, ...] = (0.2, 0.5, 1.0)


def pair_average(branch_values: np.ndarray) -> np.ndarray:
    """Combine the two branches of every pair (axis 1) by their arithmetic mean."""
    values = np.asarray(branch_values, dtype=float)
    if values.ndim < 2 or values.shape[1] != 2:
        raise ValueError("expected an array of shape (pairs, 2, ...)")
    return values.mean(axis=1)


def budget_count(pairs: int, fraction: float) -> int:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("the retention fraction must lie in (0, 1]")
    return max(1, int(round(fraction * pairs)))


def select_top(scores: np.ndarray, count: int) -> np.ndarray:
    """Indices of the ``count`` highest scores; ties follow the candidate order."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    return np.sort(order[:count])


def _unit(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    return features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-12)


def clipscore(
    image_features: np.ndarray, text_features: np.ndarray, weight: float = 2.5
) -> np.ndarray:
    """CLIPScore ``w * max(cos(image, text), 0)`` for aligned rows."""
    cosine = (_unit(image_features) * _unit(text_features)).sum(axis=-1)
    return weight * np.maximum(cosine, 0.0)


def _nearest_neighbours(features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    unit = _unit(features)
    distance = np.maximum(2.0 - 2.0 * unit @ unit.T, 0.0)
    np.fill_diagonal(distance, np.inf)
    k = min(k, len(features) - 1)
    index = np.argsort(distance, axis=1)[:, :k]
    return index, np.take_along_axis(distance, index, axis=1)


def d2_pruning(
    features: np.ndarray,
    difficulty: np.ndarray,
    count: int,
    neighbours: int = 10,
    forward_gamma: float = 1.0,
    reverse_gamma: float = 1.0,
) -> np.ndarray:
    """D2 Pruning: forward message passing adds the difficulty of close neighbours, then
    samples are taken greedily while reverse messages lower the scores around each pick."""
    index, distance = _nearest_neighbours(features, neighbours)
    scores = np.asarray(difficulty, dtype=float).copy()
    scores = scores + (np.exp(-forward_gamma * distance) * scores[index]).sum(axis=1)
    available = np.ones(len(scores), dtype=bool)
    chosen = []
    for _ in range(min(count, len(scores))):
        pick = int(np.argmax(np.where(available, scores, -np.inf)))
        chosen.append(pick)
        available[pick] = False
        scores[index[pick]] -= np.exp(-reverse_gamma * distance[pick]) * scores[pick]
    return np.sort(np.asarray(chosen))


def kmeans(features: np.ndarray, clusters: int, seed: int, iterations: int = 25) -> np.ndarray:
    """Spherical k-means labels (deterministic for a seed)."""
    unit = _unit(features)
    clusters = min(clusters, len(unit))
    generator = np.random.Generator(np.random.PCG64(seed))
    centres = unit[generator.choice(len(unit), size=clusters, replace=False)]
    labels = np.zeros(len(unit), dtype=int)
    for _ in range(iterations):
        labels = np.argmax(unit @ centres.T, axis=1)
        for cluster in range(clusters):
            members = unit[labels == cluster]
            if len(members):
                centres[cluster] = members.mean(axis=0)
        centres = _unit(centres)
    return labels


def _allocate(weights: np.ndarray, sizes: np.ndarray, count: int) -> np.ndarray:
    """Split ``count`` over clusters in proportion to ``weights`` without exceeding sizes."""
    quota = np.zeros(len(sizes), dtype=int)
    remaining = count
    active = sizes > 0
    while remaining > 0 and active.any():
        share = weights * active
        share = share / share.sum() if share.sum() > 0 else active / active.sum()
        step = np.minimum(np.floor(share * remaining).astype(int), sizes - quota)
        if step.sum() == 0:
            order = np.argsort(-(share * (sizes - quota > 0)))
            step[order[0]] = 1
        quota += step
        remaining -= int(step.sum())
        active = sizes - quota > 0
    return quota


def _herding(features: np.ndarray, count: int) -> list[int]:
    """Greedy selection whose mean feature tracks the cluster mean (reduces the MMD)."""
    mean = features.mean(axis=0)
    chosen: list[int] = []
    total = np.zeros_like(mean)
    available = np.ones(len(features), dtype=bool)
    for step in range(1, count + 1):
        gap = np.linalg.norm((total + features) / step - mean, axis=1)
        pick = int(np.argmin(np.where(available, gap, np.inf)))
        chosen.append(pick)
        available[pick] = False
        total += features[pick]
    return chosen


def coincide(
    features: np.ndarray, count: int, clusters: int = 100, temperature: float = 0.1, seed: int = 0
) -> np.ndarray:
    """COINCIDE: clusters of concept-skill activations are sampled with probability
    proportional to ``exp(S / (tau * D))``, where ``S`` is the mean cosine similarity of a
    cluster centre to the other centres (transferability) and ``D`` the mean distance of
    its members to the centre (density); members are picked to match the cluster mean."""
    unit = _unit(features)
    labels = kmeans(unit, clusters, seed)
    ids = np.unique(labels)
    centres = _unit(np.stack([unit[labels == c].mean(axis=0) for c in ids]))
    similarity = centres @ centres.T
    transfer = (similarity.sum(axis=1) - 1.0) / max(1, len(ids) - 1)
    density = np.array(
        [np.linalg.norm(unit[labels == c] - centres[k], axis=1).mean() for k, c in enumerate(ids)]
    )
    logits = transfer / (temperature * np.maximum(density, 1e-6))
    weights = np.exp(logits - logits.max())
    sizes = np.array([(labels == c).sum() for c in ids])
    quota = _allocate(weights, sizes, min(count, len(unit)))
    chosen: list[int] = []
    for k, cluster in enumerate(ids):
        members = np.flatnonzero(labels == cluster)
        chosen.extend(members[_herding(unit[members], int(quota[k]))].tolist())
    return np.sort(np.asarray(chosen))


def xmas(trajectories: np.ndarray, count: int, clusters: int = 1000, seed: int = 0) -> np.ndarray:
    """XMAS: examples are clustered by their cross-modal alignment trajectories over proxy
    checkpoints (``pairs x checkpoints x values``) and a balanced subset is drawn, taking
    the members closest to each cluster centre first."""
    flat = np.asarray(trajectories, dtype=float).reshape(len(trajectories), -1)
    clusters = min(clusters, len(flat))
    labels = kmeans(flat, clusters, seed)
    ids = np.unique(labels)
    unit = _unit(flat)
    queues = []
    for cluster in ids:
        members = np.flatnonzero(labels == cluster)
        centre = _unit(unit[members].mean(axis=0, keepdims=True))[0]
        queues.append(members[np.argsort(-(unit[members] @ centre), kind="stable")].tolist())
    chosen: list[int] = []
    while len(chosen) < min(count, len(flat)):
        for queue in queues:
            if queue and len(chosen) < count:
                chosen.append(queue.pop(0))
    return np.sort(np.asarray(chosen))


def dose(
    text_quality: np.ndarray, alignment: np.ndarray, count: int, bins: int = 10, seed: int = 0
) -> np.ndarray:
    """DOSE: the joint distribution of the text-quality and image-text alignment scores is
    binned on a grid, and samples are drawn across the grid with weights that grow with
    both scores, so that the subset keeps the spread of the pool."""
    quality, aligned = np.asarray(text_quality, dtype=float), np.asarray(alignment, dtype=float)
    generator = np.random.Generator(np.random.PCG64(seed))

    def rank(values: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(values, kind="stable"), kind="stable") / max(
            1, len(values) - 1
        )

    q_rank, a_rank = rank(quality), rank(aligned)
    cell = np.minimum((q_rank * bins).astype(int), bins - 1) * bins + np.minimum(
        (a_rank * bins).astype(int), bins - 1
    )
    weight = (q_rank + a_rank) / 2.0 + 1e-3
    ids = np.unique(cell)
    sizes = np.array([(cell == c).sum() for c in ids])
    cell_weight = np.array([weight[cell == c].sum() for c in ids])
    quota = _allocate(cell_weight, sizes, min(count, len(quality)))
    chosen: list[int] = []
    for k, c in enumerate(ids):
        members = np.flatnonzero(cell == c)
        if quota[k]:
            probabilities = weight[members] / weight[members].sum()
            chosen.extend(
                generator.choice(
                    members, size=int(quota[k]), replace=False, p=probabilities
                ).tolist()
            )
    return np.sort(np.asarray(chosen))


def dataprophet(
    features: np.ndarray, target_features: np.ndarray, perplexity: np.ndarray, neighbours: int = 10
) -> np.ndarray:
    """DataProphet-style score of every candidate against a target reference pool:
    similarity to the target features times the candidate's perplexity times its
    diversity (mean distance to its nearest candidates).  Higher is better."""
    unit, target = _unit(features), _unit(target_features)
    similarity = np.maximum(unit @ target.T, 0.0).mean(axis=1)
    _, distance = _nearest_neighbours(unit, neighbours)
    diversity = np.sqrt(distance).mean(axis=1)
    return similarity * np.asarray(perplexity, dtype=float) * diversity
