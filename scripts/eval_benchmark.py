"""Evaluate a student on natural benchmarks at the fixed mode ``H`` (Section 5.3, Table S2).

Usage::

    python scripts/eval_benchmark.py --config configs/backbones/qwen2.5-vl-3b.yaml \
        --benchmark chartqa [--adapter runs/main/seed17/rat.adapter.pt] [--limit 200]

Covers the benchmarks that are available as public Hugging Face datasets with a single
image and a short answer: ChartQA (human and augmented questions reported separately),
DocVQA, InfographicVQA and TextVQA.  All students receive the image and the question
without OCR transcripts and use the same prompt, processor and greedy decoder.  Requires
``pip install "rat-vlm[hf,benchmarks]"``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rat.backbones.base import Example, LoraSettings
from rat.config import build_backbone, load_config
from rat.evaluation.benchmarks import BENCHMARKS, anls, relaxed_accuracy, vqa_accuracy

#: benchmark -> (dataset id, configuration, split, question field, answer field)
SOURCES = {
    "chartqa": ("HuggingFaceM4/ChartQA", None, "test", "query", "label"),
    "docvqa": ("lmms-lab/DocVQA", "DocVQA", "validation", "question", "answers"),
    "infographicvqa": ("lmms-lab/DocVQA", "InfographicVQA", "validation", "question", "answers"),
    "textvqa": ("lmms-lab/textvqa", None, "validation", "question", "answers"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", required=True, choices=sorted(SOURCES))
    parser.add_argument(
        "--adapter", type=Path, help="adapter state saved by run_study.py --save-adapters"
    )
    parser.add_argument("--limit", type=int, help="evaluate only the first examples")
    parser.add_argument("--dataset", help="override the Hugging Face dataset id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()
    from datasets import load_dataset

    config = load_config(args.config, args.overrides)
    backbone = build_backbone(config)
    if args.adapter:
        lora = dict(config["lora"])
        lora["targets"] = tuple(lora["targets"])
        backbone.add_adapter(LoraSettings(**lora), seed=0)
        backbone.load_adapter_state(torch.load(args.adapter))
    dataset_id, name, split, question_field, answer_field = SOURCES[args.benchmark]
    dataset = load_dataset(args.dataset or dataset_id, name, split=split, streaming=True)
    spec = BENCHMARKS[args.benchmark]
    predictions, references, subsets = [], [], []
    batch: list[Example] = []
    mode = backbone.eval_mode

    def flush() -> None:
        if batch:
            predictions.extend(backbone.generate(batch, spec.max_new_tokens))
            batch.clear()

    for index, row in enumerate(dataset):
        if args.limit is not None and index >= args.limit:
            break
        image = row["image"].convert("RGB")
        tokens = backbone.encode_images([image], mode)[0]
        batch.append(Example(tokens, backbone.token_grid(image.size, mode), row[question_field]))
        answers = row[answer_field]
        references.append(list(answers) if isinstance(answers, (list, tuple)) else [str(answers)])
        subsets.append("augmented" if row.get("human_or_machine") else "human")
        if len(batch) == config["evaluation"]["batch_examples"]:
            flush()
    flush()
    if args.benchmark == "chartqa":
        scores = {}
        for subset in ("human", "augmented"):
            member = [k for k, s in enumerate(subsets) if s == subset]
            if member:
                scores[f"relaxed_accuracy_{subset}"] = relaxed_accuracy(
                    [predictions[k] for k in member], [references[k][0] for k in member]
                )
    elif args.benchmark == "textvqa":
        scores = {"vqa_accuracy": vqa_accuracy(predictions, references)}
    else:
        scores = {"anls": anls(predictions, references)}
    result = {
        "benchmark": args.benchmark,
        "split": spec.split,
        "examples": len(predictions),
        "backbone": backbone.name,
        "adapter": str(args.adapter) if args.adapter else None,
        **scores,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({**result, "predictions": predictions}, indent=1))


if __name__ == "__main__":
    main()
