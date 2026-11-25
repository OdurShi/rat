"""Backbones built on Hugging Face Transformers.

* :class:`QwenVLBackbone`: Qwen2.5-VL-3B/7B-Instruct, the variable-token backbones of the
  paper.  ``L`` and ``H`` cap the image at 256 and 1,024 visual tokens.
* :class:`LlavaBackbone`: LLaVA-1.5-7B, the fixed-grid backbone of the paper (576 tokens in
  both modes; ``L`` removes image detail by a 168-pixel round trip).
* :class:`SmolVLMBackbone`: SmolVLM-256M/500M, a tiled backbone that makes the complete
  pipeline runnable on a laptop.  ``L`` is the single 512-pixel view, ``H`` adds 512-pixel
  tiles of the image resized to a longest edge of 1,024 pixels.

All three keep the visual encoder and connector frozen, adapt the language model with LoRA
and consume cached visual tokens through the ``mm_encoder_outputs`` argument of the model.
Requires ``transformers >= 5.20`` and ``peft >= 0.21`` (``pip install "rat-vlm[hf]"``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image

from ..interface.geometry import (
    HIGH,
    LLAVA_GRID,
    LOW,
    InputMode,
    Size,
    TokenGrid,
    llava_grid,
    qwen_grid,
    qwen_pixel_range,
    tile_grid,
    tile_layout,
)
from .base import Backbone, Example, LoraSettings

#: Dataset prompt of the generated pairs (the student sees the image and this text).
DEFAULT_PROMPT = "{question}\nAnswer the question using a single word or phrase."


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resample(image: Image.Image, size: Size) -> Image.Image:
    return (
        image if image.size == tuple(size) else image.resize(tuple(size), Image.Resampling.BICUBIC)
    )


class HFBackbone(Backbone):
    """Shared implementation; subclasses supply the model-specific pieces."""

    modes = (LOW, HIGH)
    end_token: str
    answer_prefix = ""
    add_special_tokens = False

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        dtype: str | None = None,
        prompt_template: str = DEFAULT_PROMPT,
        image_batch: int = 8,
        vision_dtype: str | None = None,
    ) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.name = model_id
        self.device = pick_device(device)
        if dtype is None:
            dtype = "bfloat16" if self.device.type == "cuda" else "float32"
        self.dtype = getattr(torch, dtype)
        self.prompt_template = prompt_template
        self.image_batch = image_batch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=self.dtype)
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.end_token_id = self.tokenizer.convert_tokens_to_ids(self.end_token)
        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.end_token_id
        )
        embedding = self.model.get_input_embeddings()
        self.text_dim = self.visual_dim = embedding.embedding_dim
        self._initial_adapter: dict[str, torch.Tensor] | None = None
        self._prompt_cache: dict[tuple, list[int]] = {}
        # The frozen interface may run in a different precision than the student (for
        # example half precision on Apple GPUs); it then uses its own copy of the weights.
        self.vision_dtype = getattr(torch, vision_dtype) if vision_dtype else self.dtype
        self.vision_model = self.model.model
        if self.vision_dtype != self.dtype:
            import copy

            self.vision_model = copy.deepcopy(self.model.model).to(self.vision_dtype).eval()

    # ---- hooks --------------------------------------------------------------------

    def _language_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def _image_placeholder(self, grid: TokenGrid) -> str:
        """Text that reserves ``grid.count`` image-token positions."""
        raise NotImplementedError

    def _chat_prompt(self, image_text: str, text: str) -> str:
        """Chat-formatted prompt up to the start of the assistant's answer."""
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}
        ]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return self._insert_image(prompt, image_text)

    def _insert_image(self, prompt: str, image_text: str) -> str:
        raise NotImplementedError

    def _pixel_features(self, images: Sequence[Image.Image], mode: InputMode) -> list[torch.Tensor]:
        raise NotImplementedError

    def _forward_extras(
        self, input_ids: torch.Tensor, grids: Sequence[TokenGrid]
    ) -> dict[str, Any]:
        return {}

    def _step_extras(self, attention: torch.Tensor) -> dict[str, Any]:
        """Extra arguments of an incremental decoding step (after the prompt was encoded)."""
        return {}

    def _mode_image(
        self, image: Image.Image, mode: InputMode, detail: InputMode | None
    ) -> Image.Image:
        """Image handed to the processor; reduces the detail to ``detail`` when requested."""
        image = image.convert("RGB")
        if detail is None or detail == mode:
            return image
        low = self.token_grid(image.size, detail).resized
        high = self.token_grid(image.size, mode).resized
        return _resample(_resample(image, low), high)

    # ---- frozen interface ---------------------------------------------------------

    @torch.no_grad()
    def encode_images(
        self, images: Sequence[Image.Image], mode: InputMode, detail: InputMode | None = None
    ) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        prepared = [self._mode_image(image, mode, detail) for image in images]
        for start in range(0, len(prepared), self.image_batch):
            chunk = prepared[start : start + self.image_batch]
            for image, tokens in zip(chunk, self._pixel_features(chunk, mode), strict=True):
                expected = self.token_count(image.size, mode)
                if tokens.shape[0] != expected:
                    raise RuntimeError(
                        f"{self.name}: geometry predicts {expected} tokens, "
                        f"the model gave {tokens.shape[0]}"
                    )
                out.append(tokens.float().cpu())
        return out

    @torch.no_grad()
    def candidate_embeddings(
        self, questions: Sequence[str], answers: Sequence[str]
    ) -> torch.Tensor:
        language_model = self._language_model()
        self.set_training(False)
        rows = []
        for start in range(0, len(questions), 64):
            texts, spans = [], []
            for q, y in zip(
                questions[start : start + 64], answers[start : start + 64], strict=True
            ):
                text = f"Question: {q}\nAnswer: {y}"
                texts.append(text)
                spans.append(len(text) - len(y))
            encoded = self.tokenizer(
                texts,
                return_offsets_mapping=True,
                add_special_tokens=False,
                padding=True,
                padding_side="right",
                return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping")
            ids = encoded["input_ids"].to(self.device)
            attention = encoded["attention_mask"].to(self.device)
            states = language_model(
                input_ids=ids, attention_mask=attention
            ).last_hidden_state.float()
            # Positions whose characters overlap the candidate answer; template, question and
            # padding are excluded, and no end token is appended.
            answer_start = torch.tensor(spans).unsqueeze(1)
            member = (offsets[..., 1] > answer_start) & (offsets[..., 1] > offsets[..., 0])
            member = (member & encoded["attention_mask"].bool()).to(self.device)
            rows.append((states * member.unsqueeze(-1)).sum(1) / member.sum(1, keepdim=True))
        return torch.cat(rows).cpu()

    # ---- student ------------------------------------------------------------------

    def _prompt_ids(self, grid: TokenGrid, question: str) -> list[int]:
        key = (grid.parts, question)
        if key not in self._prompt_cache:
            if len(self._prompt_cache) > 200_000:
                self._prompt_cache.clear()
            text = self._chat_prompt(
                self._image_placeholder(grid), self.prompt_template.format(question=question)
            )
            self._prompt_cache[key] = self.tokenizer(
                text, add_special_tokens=self.add_special_tokens
            ).input_ids
        return self._prompt_cache[key]

    def _answer_ids(self, answer: str) -> list[int]:
        ids = self.tokenizer(self.answer_prefix + answer, add_special_tokens=False).input_ids
        return [*ids, self.end_token_id]

    def sequence_length(self, tokens: int, question: str, answer: str) -> int:
        probe = TokenGrid(((1, 1, -1.0, -1.0, 1.0, 1.0),), (1, 1))
        text = len(self._prompt_ids(probe, question)) - 1
        return tokens + text + len(self._answer_ids(answer))

    def _batch(
        self, examples: Sequence[Example], sequences: Sequence[list[int]], left_pad: bool
    ) -> dict[str, Any]:
        longest = max(len(s) for s in sequences)
        ids = torch.full((len(sequences), longest), self.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(sequences), longest), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            span = slice(longest - len(sequence), longest) if left_pad else slice(0, len(sequence))
            ids[row, span] = torch.tensor(sequence)
            attention[row, span] = 1
        ids, attention = ids.to(self.device), attention.to(self.device)
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        features = [e.tokens.to(self.device, self.dtype) for e in examples]
        inputs = {
            "input_ids": ids,
            "attention_mask": attention,
            "mm_encoder_outputs": {"image": BaseModelOutputWithPooling(pooler_output=features)},
        }
        inputs.update(self._forward_extras(ids, [e.grid for e in examples]))
        return inputs

    def answer_nll(self, examples: Sequence[Example]) -> torch.Tensor:
        prompts = [self._prompt_ids(e.grid, e.question) for e in examples]
        answers = [self._answer_ids(e.answer) for e in examples]
        sequences = [p + a for p, a in zip(prompts, answers, strict=True)]
        inputs = self._batch(examples, sequences, left_pad=False)
        hidden = self.model.model(**inputs).last_hidden_state
        # Logits only where an answer token (or the end token) is predicted.
        rows, positions, targets = [], [], []
        for row, (prompt, answer) in enumerate(zip(prompts, answers, strict=True)):
            for k, token in enumerate(answer):
                rows.append(row)
                positions.append(len(prompt) + k - 1)
                targets.append(token)
        index = torch.tensor(rows, device=self.device)
        logits = self.model.lm_head(
            hidden[index, torch.tensor(positions, device=self.device)]
        ).float()
        losses = torch.nn.functional.cross_entropy(
            logits, torch.tensor(targets, device=self.device), reduction="none"
        )
        return torch.zeros(len(examples), device=self.device).index_add(0, index, losses)

    @torch.no_grad()
    def generate(self, examples: Sequence[Example], max_new_tokens: int = 128) -> list[str]:
        self.set_training(False)
        sequences = [self._prompt_ids(e.grid, e.question) for e in examples]
        inputs = self._batch(examples, sequences, left_pad=True)
        attention = inputs["attention_mask"]
        outputs = self.model.model(**inputs, use_cache=True)
        finished = torch.zeros(len(examples), dtype=torch.bool, device=self.device)
        produced: list[torch.Tensor] = []
        for _step in range(max_new_tokens):
            logits = self.model.lm_head(outputs.last_hidden_state[:, -1]).float()
            token = logits.argmax(dim=-1)
            token = torch.where(finished, torch.full_like(token, self.end_token_id), token)
            produced.append(token)
            finished = finished | (token == self.end_token_id)
            if bool(finished.all()):
                break
            attention = torch.cat([attention, torch.ones_like(attention[:, :1])], dim=1)
            outputs = self.model.model(
                input_ids=token.unsqueeze(1),
                attention_mask=attention,
                past_key_values=outputs.past_key_values,
                use_cache=True,
                **self._step_extras(attention),
            )
        texts = []
        for row in torch.stack(produced, dim=1).tolist():
            if self.end_token_id in row:
                row = row[: row.index(self.end_token_id)]
            texts.append(self.tokenizer.decode(row, skip_special_tokens=True).strip())
        return texts

    # ---- adaptation ---------------------------------------------------------------

    def add_adapter(self, settings: LoraSettings, seed: int) -> None:
        from peft import LoraConfig, inject_adapter_in_model

        if self._initial_adapter is not None:
            raise RuntimeError("the adapter is already attached; use reset_adapter()")
        language_model = self._language_model()
        prefix = next(
            name for name, module in self.model.named_modules() if module is language_model
        )
        targets = [
            f"{prefix}.{name}"
            for name, module in language_model.named_modules()
            if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in settings.targets
        ]
        if not targets:
            raise RuntimeError(f"no LoRA target among {settings.targets} in the language model")
        config = LoraConfig(
            r=settings.rank,
            lora_alpha=settings.alpha,
            lora_dropout=settings.dropout,
            target_modules=targets,
            bias="none",
        )
        torch.manual_seed(seed)
        inject_adapter_in_model(config, self.model)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_("lora_" in name)
        self.model.to(self.device)
        self._initial_adapter = self.adapter_state()

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

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
                parameters[name].copy_(value.to(parameters[name].device, parameters[name].dtype))

    def remember_initial_adapter(self) -> None:
        self._initial_adapter = self.adapter_state()

    def reset_adapter(self) -> None:
        if self._initial_adapter is None:
            raise RuntimeError("no adapter attached")
        self.load_adapter_state(self._initial_adapter)

    def set_adapter_enabled(self, enabled: bool) -> None:
        from peft.tuners.tuners_utils import BaseTunerLayer

        for module in self.model.modules():
            if isinstance(module, BaseTunerLayer):
                module.enable_adapters(enabled)

    def set_training(self, training: bool) -> None:
        self.model.train(training)


class SmolVLMBackbone(HFBackbone):
    """SmolVLM (Idefics3 architecture): 512-pixel tiles, 64 tokens per tile for the 256M model."""

    end_token = "<end_of_utterance>"
    answer_prefix = " "

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
        low_edge: int | None = None,
        high_edge: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id, **kwargs)
        self.tile = self.processor.image_processor.max_image_size["longest_edge"]
        self.tokens_per_side = round(self.processor.image_seq_len**0.5)
        self.edges = {"L": low_edge or self.tile, "H": high_edge or 2 * self.tile}

    def _language_model(self) -> torch.nn.Module:
        return self.model.model.text_model

    def token_grid(self, size: Size, mode: InputMode) -> TokenGrid:
        return tile_grid(size, self.edges[mode.name], self.tile, self.tokens_per_side)

    def _image_placeholder(self, grid: TokenGrid) -> str:
        from transformers.models.smolvlm.processing_smolvlm import get_image_prompt_string

        rows = cols = 0
        if len(grid.parts) > 1:
            _, _, x0, y0, x1, y1 = grid.parts[0]
            rows, cols = round(2 / (y1 - y0)), round(2 / (x1 - x0))
        processor = self.processor
        return get_image_prompt_string(
            rows,
            cols,
            processor.image_seq_len,
            fake_token_around_image=processor.fake_image_token,
            image_token=processor.image_token,
            global_image_token=self.tokenizer.convert_ids_to_tokens(
                processor.global_image_token_id
            ),
        )

    def _insert_image(self, prompt: str, image_text: str) -> str:
        return prompt.replace(self.processor.image_token, image_text, 1)

    def _pixel_features(self, images: Sequence[Image.Image], mode: InputMode) -> list[torch.Tensor]:
        edge = self.edges[mode.name]
        out: list[torch.Tensor | None] = [None] * len(images)
        # Images with the same tile layout are stacked without padding frames.
        groups: dict[tuple, list[int]] = {}
        for index, image in enumerate(images):
            groups.setdefault(tile_layout(image.size, edge, self.tile), []).append(index)
        for members in groups.values():
            pixels = self.processor.image_processor(
                [[images[i]] for i in members],
                size={"longest_edge": edge},
                do_image_splitting=True,
                return_tensors="pt",
            )
            features = self.vision_model.get_image_features(
                pixels["pixel_values"].to(self.device, self.vision_dtype),
                pixels["pixel_attention_mask"].to(self.device),
            ).pooler_output
            for index, tokens in zip(members, features, strict=True):
                out[index] = tokens
        return out  # type: ignore[return-value]


class QwenVLBackbone(HFBackbone):
    """Qwen2.5-VL-Instruct with the native aspect-ratio-preserving processor (Appendix E.4)."""

    end_token = "<|im_end|>"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        low_tokens: int = 256,
        high_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id, **kwargs)
        self.max_tokens = {"L": low_tokens, "H": high_tokens}
        self.image_token = self.processor.image_token
        self.image_token_id = self.model.config.image_token_id

    def _language_model(self) -> torch.nn.Module:
        return self.model.model.language_model

    def token_grid(self, size: Size, mode: InputMode) -> TokenGrid:
        return qwen_grid(size, self.max_tokens[mode.name])

    def _image_placeholder(self, grid: TokenGrid) -> str:
        return self.image_token * grid.count

    def _insert_image(self, prompt: str, image_text: str) -> str:
        return prompt.replace(self.image_token, image_text, 1)

    def _pixel_features(self, images: Sequence[Image.Image], mode: InputMode) -> list[torch.Tensor]:
        min_pixels, max_pixels = qwen_pixel_range(self.max_tokens[mode.name])
        pixels = self.processor.image_processor(
            images=list(images), min_pixels=min_pixels, max_pixels=max_pixels, return_tensors="pt"
        )
        features = self.vision_model.get_image_features(
            pixels["pixel_values"].to(self.device, self.vision_dtype),
            pixels["image_grid_thw"].to(self.device),
        ).pooler_output
        return list(features)

    def _forward_extras(
        self, input_ids: torch.Tensor, grids: Sequence[TokenGrid]
    ) -> dict[str, Any]:
        # Grid sizes in 14-pixel patches (before the 2 x 2 merge) for the 3-D rotary positions.
        thw = [[1, 2 * grid.parts[0][0], 2 * grid.parts[0][1]] for grid in grids]
        return {
            "image_grid_thw": torch.tensor(thw, device=self.device),
            "mm_token_type_ids": (input_ids == self.image_token_id).long(),
        }

    def _step_extras(self, attention: torch.Tensor) -> dict[str, Any]:
        # Position of the new token on all three rotary axes: its text index plus the offset
        # the image introduced (the model stores it when the prompt is encoded).
        last = attention.long().sum(dim=-1, keepdim=True) - 1
        deltas = self.model.model.rope_deltas.to(last.device)
        return {"position_ids": (last + deltas).unsqueeze(0).expand(3, -1, -1)}


class LlavaBackbone(HFBackbone):
    """LLaVA-1.5 with its CLIP ViT-L/14-336 encoder and fixed 576-token grid (Appendix E.4).

    ``H`` pads the image to a square and encodes it at the native input size; ``L``
    downsamples the padded canvas to half that size and upsamples it back before encoding.
    Both modes produce the same number of tokens.
    """

    end_token = "</s>"
    add_special_tokens = True

    def __init__(self, model_id: str = "llava-hf/llava-1.5-7b-hf", **kwargs: Any) -> None:
        super().__init__(model_id, **kwargs)
        vision = self.model.config.vision_config
        self.input_size = vision.image_size
        self.grid_side = vision.image_size // vision.patch_size
        self.image_token = self.processor.image_token
        mean = self.processor.image_processor.image_mean
        self.pad_color = tuple(int(round(255 * channel)) for channel in mean)

    def _language_model(self) -> torch.nn.Module:
        return self.model.model.language_model

    def token_grid(self, size: Size, mode: InputMode) -> TokenGrid:
        return llava_grid(size, mode.name, self.grid_side, self.input_size)

    def _image_placeholder(self, grid: TokenGrid) -> str:
        return self.image_token * grid.count

    def _insert_image(self, prompt: str, image_text: str) -> str:
        return prompt.replace(self.image_token, image_text, 1)

    def _padded(self, image: Image.Image) -> Image.Image:
        side = max(image.size)
        canvas = Image.new("RGB", (side, side), self.pad_color)
        canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        return canvas

    def _mode_image(
        self, image: Image.Image, mode: InputMode, detail: InputMode | None
    ) -> Image.Image:
        square = self._padded(image.convert("RGB"))
        full = (self.input_size, self.input_size)
        if (detail or mode).name == "L":
            half = (self.input_size // 2, self.input_size // 2)
            return _resample(_resample(square, half), full)
        return _resample(square, full)

    def token_count(self, size: Size, mode: InputMode) -> int:
        return self.grid_side * self.grid_side

    def _pixel_features(self, images: Sequence[Image.Image], mode: InputMode) -> list[torch.Tensor]:
        pixels = self.processor.image_processor(
            images=list(images), do_resize=False, do_center_crop=False, return_tensors="pt"
        )
        features = self.vision_model.get_image_features(
            pixels["pixel_values"].to(self.device, self.vision_dtype),
            vision_feature_layer=self.model.config.vision_feature_layer,
            vision_feature_select_strategy=self.model.config.vision_feature_select_strategy,
        ).pooler_output
        return list(features)

    @torch.no_grad()
    def encode_images(
        self, images: Sequence[Image.Image], mode: InputMode, detail: InputMode | None = None
    ) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        prepared = [self._mode_image(image, mode, detail) for image in images]
        for start in range(0, len(prepared), self.image_batch):
            chunk = prepared[start : start + self.image_batch]
            out.extend(tokens.float().cpu() for tokens in self._pixel_features(chunk, mode))
        return out


assert LLAVA_GRID == 24  # the default grid of LLaVA-1.5 used by the geometry helpers

HF_BACKBONES = {
    "qwen2.5-vl": QwenVLBackbone,
    "llava-1.5": LlavaBackbone,
    "smolvlm": SmolVLMBackbone,
}
