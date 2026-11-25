"""Natural benchmarks and their scoring conventions (Section 5.1, Table S2).

The metric functions follow the official definitions: relaxed accuracy with a 5% numerical
tolerance for ChartQA, average normalised Levenshtein similarity for DocVQA and
InfographicVQA, VQA consensus accuracy for TextVQA, substring matching for OCRBench, F1 of
the "yes" class for POPE and CircularEval accuracy for MMBench.  ``BENCHMARKS`` lists the
evaluation splits of Table S2.
"""

from __future__ import annotations

import re
import string
from collections.abc import Sequence
from dataclasses import dataclass

# --------------------------------------------------------------------------------------
# ChartQA: relaxed accuracy
# --------------------------------------------------------------------------------------


def _to_float(text: str) -> float | None:
    text = text.strip()
    try:
        if text.endswith("%"):
            return float(text.rstrip("%")) / 100.0
        return float(text.replace(",", ""))
    except ValueError:
        return None


def relaxed_correct(prediction: str, target: str, tolerance: float = 0.05) -> bool:
    """ChartQA relaxed correctness: numbers within ``tolerance`` relative error, text by
    case-insensitive exact match."""
    predicted, expected = _to_float(prediction), _to_float(target)
    if predicted is not None and expected is not None:
        if expected == 0.0:
            return predicted == 0.0
        return abs(predicted - expected) / abs(expected) <= tolerance
    return prediction.strip().lower() == target.strip().lower()


def relaxed_accuracy(predictions: Sequence[str], targets: Sequence[str]) -> float:
    return (
        100.0
        * sum(relaxed_correct(p, t) for p, t in zip(predictions, targets, strict=True))
        / len(targets)
    )


# --------------------------------------------------------------------------------------
# DocVQA / InfographicVQA: ANLS
# --------------------------------------------------------------------------------------


def levenshtein(first: str, second: str) -> int:
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, start=1):
        current = [i]
        for j, b in enumerate(second, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def anls_score(prediction: str, references: Sequence[str], threshold: float = 0.5) -> float:
    """Normalised Levenshtein similarity of one answer: the best reference counts, and
    similarities below ``1 - threshold`` are set to zero."""
    best = 0.0
    answer = " ".join(prediction.strip().lower().split())
    for reference in references:
        target = " ".join(reference.strip().lower().split())
        longest = max(len(answer), len(target))
        distance = levenshtein(answer, target) / longest if longest else 0.0
        best = max(best, 1.0 - distance if distance < threshold else 0.0)
    return best


def anls(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    """100 x ANLS over a dataset."""
    scores = [anls_score(p, r) for p, r in zip(predictions, references, strict=True)]
    return 100.0 * sum(scores) / len(scores)


# --------------------------------------------------------------------------------------
# TextVQA: VQA consensus accuracy
# --------------------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}  # fmt: skip
_PUNCTUATION = re.compile(f"[{re.escape(string.punctuation)}]")


def normalize_vqa(text: str) -> str:
    """Answer normalisation of the VQA evaluator: case, punctuation, articles, digits."""
    text = text.lower().replace("\n", " ").replace("\t", " ").strip()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = _PUNCTUATION.sub(lambda m: "" if m.group() in "'" else " ", text)
    words = [_NUMBER_WORDS.get(word, word) for word in text.split() if word not in _ARTICLES]
    return " ".join(words)


def vqa_score(prediction: str, answers: Sequence[str]) -> float:
    """Consensus accuracy of one answer against the ten human answers:
    the mean over leave-one-out subsets of ``min(1, matches / 3)``."""
    answer = normalize_vqa(prediction)
    humans = [normalize_vqa(a) for a in answers]
    scores = []
    for left_out in range(len(humans)):
        others = humans[:left_out] + humans[left_out + 1 :]
        scores.append(min(1.0, sum(h == answer for h in others) / 3.0))
    return sum(scores) / len(scores)


def vqa_accuracy(predictions: Sequence[str], answers: Sequence[Sequence[str]]) -> float:
    scores = [vqa_score(p, a) for p, a in zip(predictions, answers, strict=True)]
    return 100.0 * sum(scores) / len(scores)


# --------------------------------------------------------------------------------------
# OCRBench, POPE, MMBench
# --------------------------------------------------------------------------------------


def ocrbench_correct(
    prediction: str, references: Sequence[str], strip_spaces: bool = False
) -> bool:
    """OCRBench v1: an answer is correct if a reference occurs in it (case-insensitive;
    handwritten-expression items compare with all whitespace removed)."""
    answer = prediction.lower().strip().replace("\n", " ")
    if strip_spaces:
        answer = "".join(answer.split())
    for reference in references:
        target = reference.lower().strip().replace("\n", " ")
        if strip_spaces:
            target = "".join(target.split())
        if target in answer:
            return True
    return False


def ocrbench_total(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> int:
    """Number of correct answers; the full benchmark has 1,000 items."""
    return sum(ocrbench_correct(p, r) for p, r in zip(predictions, references, strict=True))


def parse_yes_no(prediction: str) -> str:
    """POPE answer parsing: the first of "yes"/"no" among the leading words; "no" otherwise."""
    words = _PUNCTUATION.sub(" ", prediction.lower()).split()
    for word in words[:8]:
        if word in ("yes", "no"):
            return word
        if word == "not":
            return "no"
    return "no"


def pope_f1(predictions: Sequence[str], labels: Sequence[str]) -> float:
    """F1 of the "yes" class in percent."""
    parsed = [parse_yes_no(p) for p in predictions]
    truth = [label.strip().lower() for label in labels]
    true_positive = sum(p == "yes" and t == "yes" for p, t in zip(parsed, truth, strict=True))
    predicted = sum(p == "yes" for p in parsed)
    actual = sum(t == "yes" for t in truth)
    if not true_positive:
        return 0.0
    precision, recall = true_positive / predicted, true_positive / actual
    return 100.0 * 2 * precision * recall / (precision + recall)


def parse_choice(prediction: str, options: str = "ABCD") -> str | None:
    """Exact-choice parsing: a leading option letter, optionally in brackets or followed by
    punctuation."""
    match = re.match(rf"\s*\(?([{options}])\)?(?:[\s.:,)]|$)", prediction.strip())
    return match.group(1) if match else None


def circular_accuracy(
    predictions: Sequence[str], answers: Sequence[str], groups: Sequence[int]
) -> float:
    """MMBench CircularEval: a question is correct only if every circular shift of its
    options is answered correctly.  ``groups`` gives the question index of every item."""
    correct: dict[int, bool] = {}
    for prediction, answer, group in zip(predictions, answers, groups, strict=True):
        correct[group] = correct.get(group, True) and parse_choice(prediction) == answer
    return 100.0 * sum(correct.values()) / len(correct)


# --------------------------------------------------------------------------------------
# Registry (Table S2)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Benchmark:
    name: str
    split: str
    metric: str
    #: Maximum generation length (Appendix E.1): 128 for short answers, 512 otherwise.
    max_new_tokens: int = 128
    #: Part of the seven-benchmark core transfer suite (chart, document and OCR).
    core: bool = True


BENCHMARKS: dict[str, Benchmark] = {
    b.name: b
    for b in (
        Benchmark(
            "chartqa",
            "test; human and augmented separately",
            "relaxed accuracy (5% numerical tolerance)",
        ),
        Benchmark("chartqapro", "full released benchmark", "official overall score", 512),
        Benchmark("textvqa", "v0.5.1 validation", "VQA consensus accuracy"),
        Benchmark("docvqa", "single-page validation", "ANLS"),
        Benchmark("infographicvqa", "validation", "ANLS"),
        Benchmark("ocrbench", "v1, 1,000 examples", "total score out of 1,000"),
        Benchmark(
            "ocrbench_v2", "public 10,000-example release", "official English/Chinese scores", 512
        ),
        Benchmark("mathvista", "testmini", "answer accuracy", 512, core=False),
        Benchmark(
            "mmbench", "English development, 1,164 questions", "CircularEval accuracy", core=False
        ),
        Benchmark("pope", "MS-COCO random/popular/adversarial", "F1 per set", core=False),
    )
}
