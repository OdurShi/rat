"""Margins, the gate and routing on hand-written score matrices.

    python examples/margins_and_routing.py

Shows Eq. (3), the decomposition of Eq. (S6) and Eq. (4) for three pairs, then the same rule
on a ladder of three resolutions.
"""

from __future__ import annotations

import math

import torch

from rat.margins import branch_margins, decompose, probability_from_threshold
from rat.routing import lazy_route, route

GAMMA = math.log(4)

#: Score matrices S[b, j] (image branch x candidate) in the modes L and H.
PAIRS = {
    "clear at low resolution": ([[3.1, 0.4], [0.2, 2.9]], [[3.4, 0.1], [0.0, 3.3]]),
    "needs high resolution": ([[1.2, 1.0], [1.1, 1.3]], [[2.8, 0.3], [0.5, 2.6]]),
    "one branch never resolves": ([[5.0, 0.0], [1.0, 0.2]], [[6.0, 0.0], [1.5, 0.5]]),
}
COSTS = (468, 1944)  # visual tokens of both images at L and H


def main() -> None:
    print(
        f"threshold gamma = log 4: the weaker branch needs probability {probability_from_threshold(GAMMA):.2f}\n"
    )
    for name, matrices in PAIRS.items():
        print(name)
        margins = []
        for mode, matrix in zip("LH", matrices, strict=True):
            scores = torch.tensor(matrix)
            delta, beta, minimum = (float(v) for v in decompose(scores))
            l0, l1 = branch_margins(scores).tolist()
            margins.append(minimum)
            print(
                f"  {mode}: margins ({l0:+.1f}, {l1:+.1f})  mean contrast {delta:+.2f}  "
                f"imbalance {beta:+.2f}  minimum {minimum:+.2f}  {'pass' if minimum >= GAMMA else 'fail'}"
            )
        decision = route(margins, COSTS, GAMMA)
        print(f"  -> {'skip the pair' if decision is None else 'train at ' + 'LH'[decision]}\n")

    print("Three-rung ladder (256 / 576 / 1,024 tokens per image), evaluated lazily:")
    ladder_costs = (512, 1152, 2048)
    for margins in ([2.0, 2.5, 3.0], [0.3, 1.6, 2.4], [0.1, 0.4, 1.9], [0.1, 0.2, 0.3]):
        decision, evaluated = lazy_route(lambda mode, m=margins: m[mode], ladder_costs, GAMMA)
        chosen = (
            "skip" if decision is None else f"rung {decision} ({ladder_costs[decision]} tokens)"
        )
        print(f"  margins {margins} -> {chosen}; read {len(evaluated)} of 3 modes")


if __name__ == "__main__":
    main()
