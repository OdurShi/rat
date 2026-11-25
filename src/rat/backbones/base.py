"""Backbone interface used by the RAT pipeline.

A backbone bundles the three roles the paper assigns to the vision-language model:

* the frozen visual interface ``phi_a`` (preprocessing, visual encoder, connector) that maps
  an image in input mode ``a`` to projected visual tokens (Section 4.1);
* the frozen initial language backbone ``psi_0`` that encodes a question and a candidate
  answer into one embedding (Eq. (S5));
* the student ``p_theta(y | X, q)`` whose language-side adaptation parameters are trained
  with answer-only cross-entropy (Eq. (5)) and that answers by greedy decoding.

Visual tokens are plain tensors, so they can be cached once and reused for reader fitting,
routing and student updates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from PIL import Image

from ..interface.geometry import InputMode, Size, TokenGrid


@dataclass(frozen=True)
class LoraSettings:
    """Language-side LoRA of Appendix E.4."""

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    #: Leaf names of the adapted projections: attention query/key/value/output and the MLP
    #: input/output projections.
    targets: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )  # fmt: skip


@dataclass(frozen=True)
class Example:
    """One single-image example: cached visual interface, question and (for training) answer."""

    tokens: torch.Tensor
    grid: TokenGrid
    question: str
    answer: str | None = None
    meta: dict = field(default_factory=dict, compare=False)


class Backbone(ABC):
    """Abstract vision-language backbone with input modes ordered from ``L`` to ``H``."""

    name: str
    modes: tuple[InputMode, ...]
    visual_dim: int
    text_dim: int
    device: torch.device

    @property
    def eval_mode(self) -> InputMode:
        """The fixed evaluation mode ``H`` shared by all compared methods."""
        return self.modes[-1]

    # ---- geometry -----------------------------------------------------------------

    @abstractmethod
    def token_grid(self, size: Size, mode: InputMode) -> TokenGrid:
        """Token layout of an image of ``size = (width, height)`` in ``mode``."""

    def token_count(self, size: Size, mode: InputMode) -> int:
        """``K_{i,a}``: number of visual tokens, from preprocessing geometry alone."""
        return self.token_grid(size, mode).count

    def pair_cost(self, size: Size, mode: InputMode) -> int:
        """``c_i(a) = K^0_{i,a} + K^1_{i,a}`` for two images of the same canvas size."""
        return 2 * self.token_count(size, mode)

    # ---- frozen interface ---------------------------------------------------------

    @abstractmethod
    def encode_images(
        self, images: Sequence[Image.Image], mode: InputMode, detail: InputMode | None = None
    ) -> list[torch.Tensor]:
        """Projected visual tokens ``K x d_v`` of every image (on the CPU, no gradients).

        With ``detail`` the image is first reduced to the image detail of that mode and then
        encoded on the token grid of ``mode``; this gives the sequence-length control
        ``U`` of the preliminary study (``mode = H``, ``detail = L``).
        """

    @abstractmethod
    def candidate_embeddings(
        self, questions: Sequence[str], answers: Sequence[str]
    ) -> torch.Tensor:
        """Eq. (S5): mean final-layer state over the answer positions of
        ``"Question: {q}\\nAnswer: {y}"``, one row per (question, answer)."""

    # ---- student ------------------------------------------------------------------

    @abstractmethod
    def sequence_length(self, tokens: int, question: str, answer: str) -> int:
        """Length of the teacher-forced sequence of an example with ``tokens`` visual tokens."""

    @abstractmethod
    def answer_nll(self, examples: Sequence[Example]) -> torch.Tensor:
        """Per-example negative log-likelihood of the answer, summed over the answer tokens
        including the end token (the inner sums of Eq. (5)); differentiable."""

    @abstractmethod
    def generate(self, examples: Sequence[Example], max_new_tokens: int = 128) -> list[str]:
        """Greedy answers, stripped of leading and trailing whitespace."""

    @abstractmethod
    def add_adapter(self, settings: LoraSettings, seed: int) -> None:
        """Attach language-side LoRA and remember its initial state ``theta_0``."""

    @abstractmethod
    def reset_adapter(self) -> None:
        """Return the adaptation parameters to ``theta_0``."""

    @abstractmethod
    def remember_initial_adapter(self) -> None:
        """Declare the current adaptation parameters to be ``theta_0``."""

    @abstractmethod
    def trainable_parameters(self) -> list[torch.nn.Parameter]: ...

    @abstractmethod
    def set_training(self, training: bool) -> None: ...

    def set_adapter_enabled(self, enabled: bool) -> None:  # noqa: B027
        """Switch the adapter on or off; with it off the backbone is the unadapted model."""

    @abstractmethod
    def adapter_state(self) -> dict[str, torch.Tensor]: ...

    @abstractmethod
    def load_adapter_state(self, state: dict[str, torch.Tensor]) -> None: ...

    # ---- derived quantities -------------------------------------------------------

    @torch.no_grad()
    def candidate_margins(
        self, examples: Sequence[Example], alternatives: Sequence[str]
    ) -> torch.Tensor:
        """Two-candidate log-likelihood margin of the student (Appendix E.3).

        For every example the log probability of its answer minus that of the alternative,
        each summed over answer tokens including the end token.
        """
        swapped = [
            Example(e.tokens, e.grid, e.question, other)
            for e, other in zip(examples, alternatives, strict=True)
        ]
        return self.answer_nll(swapped) - self.answer_nll(examples)
