"""Any chat VLM in transformers, read through option letters (`s1ght.runtimes.letters`).

Every question is one prompt; the answer is the next-token logits of the option letters at
the end of the prompt, so one forward pass per prompt and no generation. Prompts are padded
on the left and run in batches; only the last position's logits are computed.

The prompt reads the whole state and images again for every question. Reusing the encoded
prefix across questions (Qwen's KV cache cut at `<|vision_end|>`) is a later optimization.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from s1ght.runtimes import letters
from s1ght.runtimes.base import score_to_answer
from s1ght.runtimes.native.families.base import DTYPES, Item, release, torch_dtype
from s1ght.schema.questions import question_from_wire


class SemifGenericFamily:
    """Option-letter readout for chat VLMs loaded with `AutoModelForImageTextToText`.

    Args:
        batch: Prompts per forward pass.
        flip: Option-order debiasing: "auto", "always" or "never" (see `letters`).
        margin: Gap in nat under which "auto" asks again in other option orders.
        max_side: Longer image side in pixels after S1GHT's own resize.
    """

    name = "semif_generic"
    caps = letters.CAPS
    template_kwargs: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        *,
        batch: int = 4,
        flip: letters.Flip = "auto",
        margin: float = letters.FLIP_MARGIN,
        max_side: int = letters.MAX_SIDE,
    ) -> None:
        if batch < 1:
            raise ValueError("batch must be >= 1")
        if flip not in ("auto", "always", "never"):
            raise ValueError(f"flip must be 'auto', 'always' or 'never', got {flip!r}")
        self.batch, self.flip, self.margin, self.max_side = batch, flip, margin, max_side
        self.model: Any = None
        self.processor: Any = None
        self.device = "cpu"
        self.letter_ids: dict[str, list[int]] = {}
        self.chat_kwargs: dict[str, Any] = {}

    def load(self, path: str, device: str, dtype: str) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")
        self.processor = AutoProcessor.from_pretrained(path)
        tokenizer = self.processor.tokenizer
        tokenizer.padding_side = "left"
        model: Any = AutoModelForImageTextToText.from_pretrained(path, dtype=torch_dtype(dtype))
        self.model = model.to(device).eval()
        self.device = device
        self.letter_ids = letter_token_ids(tokenizer)
        # pass only the switches this template reads (Qwen3-VL-Instruct has no thinking mode)
        template = str(getattr(self.processor, "chat_template", "") or "")
        self.chat_kwargs = {k: v for k, v in self.template_kwargs.items() if k in template}

    def messages(self, prompt: letters.Prompt) -> list[dict[str, Any]]:
        """The prompt in the processor's chat format (image parts without data)."""
        out = []
        for m in prompt.messages:
            content = m["content"]
            if isinstance(content, list):
                content = [{"type": "image"} if part["type"] == "image" else part for part in content]
            out.append({"role": m["role"], "content": content})
        return out

    def run(self, items: list[Item]) -> list[tuple[dict[str, dict[str, Any]], int]]:
        if self.model is None:
            raise RuntimeError("model is not loaded")
        images = [[letters.fit_image(img, self.max_side) for img in imgs] for _, _, imgs in items]
        tokens = [0] * len(items)

        def score(prompts: list[letters.Prompt]) -> list[np.ndarray]:
            out = []
            for start in range(0, len(prompts), self.batch):
                chunk = prompts[start : start + self.batch]
                logits, lengths = self.forward(chunk, images)
                for p, row, n in zip(chunk, logits, lengths, strict=True):
                    tokens[p.item] += n
                    scores = {c: float(np.logaddexp.reduce(row[self.letter_ids[c]])) for c in p.letters}
                    out.append(letters.letter_logprobs(scores, p.letters))
            return out

        reads = letters.read_all([(s, qs, len(imgs)) for s, qs, imgs in items], score, self.flip, self.margin)
        return [
            ({qid: score_to_answer(question_from_wire(qs[qid]), v).to_wire() for qid, v in r.items()}, tokens[i])
            for i, ((_, qs, _), r) in enumerate(zip(items, reads, strict=True))
        ]

    def forward(self, prompts: list[letters.Prompt], images: list[list[Any]]) -> tuple[np.ndarray, list[int]]:
        """Last-position logits (float32, one row per prompt) and each prompt's token count."""
        import torch

        texts = [
            self.processor.apply_chat_template(
                self.messages(p), tokenize=False, add_generation_prompt=True, **self.chat_kwargs
            )
            for p in prompts
        ]
        pics = [
            images[p.item][part["index"]]
            for p in prompts
            for m in p.messages
            if isinstance(m["content"], list)
            for part in m["content"]
            if part["type"] == "image"
        ]
        inputs = self.processor(text=texts, images=pics or None, padding=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs, logits_to_keep=1).logits[:, -1].float().cpu().numpy()
        return logits, [int(n) for n in inputs["attention_mask"].sum(dim=1)]

    def close(self) -> None:
        self.model = self.processor = None
        release()


def letter_token_ids(tokenizer: Any) -> dict[str, list[int]]:
    """Token ids that spell each option letter on their own ("A", and " A" when that is one token)."""
    ids: dict[str, list[int]] = {}
    for c in letters.LETTERS:
        found = []
        for text in (c, f" {c}"):
            enc = tokenizer.encode(text, add_special_tokens=False)
            if len(enc) == 1 and enc[0] not in found:
                found.append(enc[0])
        if not found:
            raise ValueError(f"the tokenizer has no single token for {c!r}; option letters cannot be read")
        ids[c] = found
    return ids
