"""A tiny self-contained backbone for tests, CI and the quick start.

It has the same three parts as the real backbones, built and randomly initialised in code:
a frozen visual interface (a fixed random projection of 28-pixel patches of the resized
grayscale image), a small causal character-level language model, and language-side LoRA.
Nothing is downloaded and everything runs on a CPU in seconds.  The model is far too small to
answer questions well; it exercises every code path of the pipeline with real tensors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn

from ..interface.geometry import HIGH, LOW, InputMode, Size, TokenGrid, qwen_grid
from ..lora import LoRALinear
from .base import Backbone, Example, LoraSettings

PATCH = 28
VOCABULARY = [chr(code) for code in range(32, 127)] + ["\n"]
PAD, BOS, IMG, END = range(len(VOCABULARY), len(VOCABULARY) + 4)


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(dim, dim) for _ in range(4))
        self.up_proj, self.down_proj = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        h = self.norm1(x)

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, length, self.heads, dim // self.heads).transpose(1, 2)

        attention = nn.functional.scaled_dot_product_attention(
            split(self.q_proj(h)), split(self.k_proj(h)), split(self.v_proj(h)), attn_mask=mask
        )
        x = x + self.o_proj(attention.transpose(1, 2).reshape(batch, length, dim))
        return x + self.down_proj(nn.functional.gelu(self.up_proj(self.norm2(x))))


class _LanguageModel(nn.Module):
    def __init__(self, dim: int, layers: int, heads: int, max_length: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(len(VOCABULARY) + 4, dim)
        self.position = nn.Embedding(max_length, dim)
        self.blocks = nn.ModuleList(_Block(dim, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, len(VOCABULARY) + 4)

    def forward(self, embeddings: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Final-layer states for ``B x T x d`` inputs; ``valid`` marks non-padding positions."""
        length = embeddings.shape[1]
        x = embeddings + self.position(torch.arange(length, device=embeddings.device))
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        mask = causal[None, None] & valid[:, None, None, :]
        # A padded query row attends to itself so that its softmax stays finite.
        mask = mask | torch.eye(length, dtype=torch.bool, device=x.device)[None, None]
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


def _encode(text: str) -> list[int]:
    return [VOCABULARY.index(ch) if ch in VOCABULARY else VOCABULARY.index("?") for ch in text]


class TinyBackbone(Backbone):
    """Randomly initialised stand-in; ``L`` and ``H`` allow at most 16 and 64 visual tokens."""

    modes = (LOW, HIGH)

    def __init__(
        self,
        dim: int = 48,
        layers: int = 2,
        heads: int = 4,
        low_tokens: int = 16,
        high_tokens: int = 64,
        seed: int = 0,
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        self.name = "tiny"
        self.device = torch.device(device)
        self.visual_dim = self.text_dim = dim
        self.max_tokens = {"L": low_tokens, "H": high_tokens}
        torch.manual_seed(seed)
        self.model = _LanguageModel(dim, layers, heads, max_length).to(self.device).eval()
        self.projection = (torch.randn(PATCH * PATCH, dim) / PATCH).to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._initial: dict[str, torch.Tensor] | None = None

    # ---- geometry and frozen interface ----------------------------------------------

    def token_grid(self, size: Size, mode: InputMode) -> TokenGrid:
        return qwen_grid(size, self.max_tokens[mode.name])

    @torch.no_grad()
    def encode_images(
        self, images: Sequence[Image.Image], mode: InputMode, detail: InputMode | None = None
    ) -> list[torch.Tensor]:
        out = []
        for image in images:
            grid = self.token_grid(image.size, mode)
            gray = image.convert("L")
            if detail is not None and detail != mode:
                gray = gray.resize(
                    self.token_grid(image.size, detail).resized, Image.Resampling.BICUBIC
                )
            gray = gray.resize(grid.resized, Image.Resampling.BICUBIC)
            pixels = torch.from_numpy(np.asarray(gray, dtype=np.float32) / 255.0 - 0.5)
            rows, cols = grid.parts[0][:2]
            patches = (
                pixels.view(rows, PATCH, cols, PATCH).permute(0, 2, 1, 3).reshape(rows * cols, -1)
            )
            out.append((patches.to(self.device) @ self.projection).cpu())
        return out

    def _states(
        self, sequences: Sequence[list[int]], visual: Sequence[torch.Tensor | None]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        longest = max(len(s) for s in sequences)
        ids = torch.full((len(sequences), longest), PAD, dtype=torch.long)
        for row, sequence in enumerate(sequences):
            ids[row, : len(sequence)] = torch.tensor(sequence)
        ids = ids.to(self.device)
        embeddings = self.model.embed(ids)
        for row, tokens in enumerate(visual):
            if tokens is not None:
                embeddings = embeddings.clone() if row == 0 else embeddings
                embeddings[row, (ids[row] == IMG).nonzero().squeeze(-1)] = tokens.to(self.device)
        return self.model(embeddings, ids != PAD), ids

    @torch.no_grad()
    def candidate_embeddings(
        self, questions: Sequence[str], answers: Sequence[str]
    ) -> torch.Tensor:
        self.set_training(False)
        rows = []
        for question, answer in zip(questions, answers, strict=True):
            prefix = _encode(f"Question: {question}\nAnswer: ")
            states, _ = self._states([prefix + _encode(answer)], [None])
            rows.append(states[0, len(prefix) :].mean(dim=0))
        return torch.stack(rows).cpu()

    # ---- student ----------------------------------------------------------------------

    def _prompt(self, tokens: int, question: str) -> list[int]:
        return [BOS] + [IMG] * tokens + _encode(f"Q: {question} A: ")

    def sequence_length(self, tokens: int, question: str, answer: str) -> int:
        return len(self._prompt(tokens, question)) + len(answer) + 1

    def answer_nll(self, examples: Sequence[Example]) -> torch.Tensor:
        prompts = [self._prompt(e.tokens.shape[0], e.question) for e in examples]
        answers = [_encode(e.answer) + [END] for e in examples]
        states, _ = self._states(
            [p + a for p, a in zip(prompts, answers, strict=True)], [e.tokens for e in examples]
        )
        losses = []
        for row, (prompt, answer) in enumerate(zip(prompts, answers, strict=True)):
            logits = self.model.head(states[row, len(prompt) - 1 : len(prompt) + len(answer) - 1])
            losses.append(
                nn.functional.cross_entropy(
                    logits, torch.tensor(answer, device=self.device), reduction="sum"
                )
            )
        return torch.stack(losses)

    @torch.no_grad()
    def generate(self, examples: Sequence[Example], max_new_tokens: int = 128) -> list[str]:
        self.set_training(False)
        outputs = []
        for example in examples:
            sequence = self._prompt(example.tokens.shape[0], example.question)
            produced: list[int] = []
            for _ in range(max_new_tokens):
                states, _ = self._states([sequence + produced], [example.tokens])
                token = int(self.model.head(states[0, -1]).argmax())
                if token == END:
                    break
                produced.append(token)
            outputs.append("".join(VOCABULARY[t] for t in produced if t < len(VOCABULARY)).strip())
        return outputs

    # ---- adaptation -------------------------------------------------------------------

    def _lora_layers(self) -> list[LoRALinear]:
        return [module for module in self.model.modules() if isinstance(module, LoRALinear)]

    def add_adapter(self, settings: LoraSettings, seed: int) -> None:
        if self._initial is not None:
            raise RuntimeError("the adapter is already attached; use reset_adapter()")
        generator = torch.Generator().manual_seed(seed)
        for block in self.model.blocks:
            for name in settings.targets:
                if hasattr(block, name):
                    layer = LoRALinear(
                        getattr(block, name),
                        settings.rank,
                        settings.alpha,
                        settings.dropout,
                        generator,
                    )
                    setattr(block, name, layer.to(self.device))
        self.remember_initial_adapter()

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for layer in self._lora_layers() for p in (layer.lora_A, layer.lora_B)]

    def adapter_state(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if "lora_" in name
        }

    def load_adapter_state(self, state: dict[str, torch.Tensor]) -> None:
        parameters = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, value in state.items():
                parameters[name].copy_(value)

    def remember_initial_adapter(self) -> None:
        self._initial = self.adapter_state()

    def reset_adapter(self) -> None:
        if self._initial is None:
            raise RuntimeError("no adapter attached")
        self.load_adapter_state(self._initial)

    def set_adapter_enabled(self, enabled: bool) -> None:
        for layer in self._lora_layers():
            layer.enabled = enabled

    def set_training(self, training: bool) -> None:
        self.model.train(training)


assert math.isqrt(PATCH * PATCH) == PATCH
