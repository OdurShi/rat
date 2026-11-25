"""Experiment configuration.

A configuration is a nested dictionary loaded from YAML.  ``base`` names another file whose
values are overridden by the current one; ``key.sub=value`` strings override single entries
from the command line.  The defaults in ``configs/base.yaml`` are the settings of the paper.
"""

from __future__ import annotations

import contextlib
import copy
from pathlib import Path
from typing import Any

import yaml


def deep_update(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``source`` into ``target`` (in place) and return it."""
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML configuration, resolving ``base`` files relative to the file itself."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    config: dict[str, Any] = {}
    base = raw.pop("base", None)
    if base is not None:
        config = load_config(path.parent / base)
    deep_update(config, raw)
    for override in overrides or []:
        apply_override(config, override)
    return config


def apply_override(config: dict[str, Any], override: str) -> None:
    """Apply one ``dotted.key=value`` override; the value is parsed as YAML."""
    if "=" not in override:
        raise ValueError(f"override {override!r} must have the form key=value")
    dotted, text = override.split("=", 1)
    node = config
    keys = dotted.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    value = yaml.safe_load(text)
    if isinstance(value, str):
        # YAML 1.1 reads "1e-4" as a string; accept such numbers as floats.
        with contextlib.suppress(ValueError):
            value = float(value)
    node[keys[-1]] = value


def build_backbone(config: dict[str, Any]):
    """Instantiate the backbone described by ``config['backbone']``."""
    spec = dict(config["backbone"])
    family = spec.pop("family")
    if family == "tiny":
        from .backbones.tiny import TinyBackbone

        return TinyBackbone(**spec)
    from .backbones.hf import HF_BACKBONES

    if family not in HF_BACKBONES:
        raise KeyError(
            f"unknown backbone family {family!r}; expected one of {sorted(HF_BACKBONES)} or 'tiny'"
        )
    return HF_BACKBONES[family](**spec)
