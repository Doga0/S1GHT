"""Prefix KV cache for option-letter families: encode a request's images once, branch its prompts off them.

Every prompt of a request starts with the same system message and images (`letters.build`),
and differs only in the question text after the last image. When a request has two or more
prompts in one pass, `PrefixCache` runs that shared prefix once (with the vision encoder),
copies its KV cache for every prompt and runs only the question suffixes, right-padded in
batches. The prefix is kept for the second pass of `letters.read_all` (the extra option orders
of close questions) and freed when that pass does not need it. Requests with one prompt, or
without images, go to the family's plain path.

Nothing here is model-specific. A family opts in with a `PrefixLayout`: the special token that
closes an image block and the position scheme. Because that token is special, the tokenizer
splits there, so a prompt's ids are the prefix ids followed by its own suffix ids and the
images are preprocessed once per request.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from s1ght.runtimes import letters

Logits = list[tuple[np.ndarray, int]]
"""Per prompt: last-position logits (float32) and the prompt's token count."""


@dataclass(frozen=True)
class PrefixLayout:
    """How a model family marks the end of the shared prefix and numbers positions.

    Attributes:
        end_token: Special token closing an image block; the prefix ends after its last one
            ("<|vision_end|>" for Qwen-VL).
        mrope: Qwen-VL multimodal RoPE: the prefix gets 3-D positions from the model's
            `get_rope_index`, and the text after it continues from the largest of them. Without
            it, positions are 0, 1, 2, ... and the model numbers the suffix itself.
    """

    end_token: str
    mrope: bool = False


@dataclass
class Prefix:
    """One request's encoded prefix."""

    ids: list[int]
    past: Any  # transformers Cache after the prefix
    next_position: int  # position of the first suffix token


class PrefixCache:
    """Runs prompts off shared prefixes for one loaded model.

    Args:
        model: A transformers image-text-to-text model.
        processor: Its processor (tokenizer and image processor).
        layout: The family's `PrefixLayout`.
        device: Where `model` lives.
        batch: Prompts per suffix forward pass (each holds its own copy of the prefix cache).
    """

    def __init__(self, model: Any, processor: Any, layout: PrefixLayout, device: str, batch: int) -> None:
        self.model, self.processor, self.layout = model, processor, layout
        self.device, self.batch = device, batch
        self.end_id = processor.tokenizer.convert_tokens_to_ids(layout.end_token)

    def logits(
        self,
        prompts: list[letters.Prompt],
        texts: list[str],
        images: list[list[Any]],
        store: dict[int, Prefix],
        plain: Callable[[list[int]], Logits],
    ) -> Logits:
        """Logits of `prompts` (templated as `texts`); `plain(indices)` scores the rest.

        `store` holds prefixes for one `run` of the family; read_all scores twice, and a prefix
        the second pass does not need is freed when it starts.
        """
        groups: dict[int, list[int]] = {}
        for k, p in enumerate(prompts):
            groups.setdefault(p.item, []).append(k)
        for item in [i for i in store if i not in groups]:
            del store[item]
        out: list[tuple[np.ndarray, int] | None] = [None] * len(prompts)
        rest: list[int] = []
        for item, ks in groups.items():
            shared = None
            if images[item] and (len(ks) > 1 or item in store):
                shared = self.branch([texts[k] for k in ks], images[item], store, item)
            if shared is None:
                rest += ks
            else:
                for k, r in zip(ks, shared, strict=True):
                    out[k] = r
        for k, r in zip(rest, plain(rest) if rest else [], strict=True):
            out[k] = r
        return out  # type: ignore[return-value]

    def split(self, text: str) -> tuple[str, str] | None:
        """`text` cut after its last end token; None when it has none."""
        at = text.rfind(self.layout.end_token)
        if at < 0:
            return None
        at += len(self.layout.end_token)
        return text[:at], text[at:]

    def branch(self, texts: list[str], images: list[Any], store: dict[int, Prefix], item: int) -> Logits | None:
        """One request's prompts off its prefix; None when they do not share one."""
        import torch

        parts = [self.split(t) for t in texts]
        if any(p is None or p[0] != parts[0][0] for p in parts):  # type: ignore[index]
            return None
        tokenizer = self.processor.tokenizer
        suffixes = [list(row) for row in tokenizer([p[1] for p in parts], add_special_tokens=False)["input_ids"]]  # type: ignore[index]
        if not all(suffixes):
            return None
        prefix = store.get(item)
        if prefix is None:
            prefix = self.encode(texts[0], images, suffixes[0])
            if prefix is None:
                return None
            store[item] = prefix
        cut = len(prefix.ids)

        out: Logits = []
        pad = tokenizer.pad_token_id or 0
        for start in range(0, len(suffixes), self.batch):
            rows = suffixes[start : start + self.batch]
            m, width = len(rows), max(len(r) for r in rows)
            ids = torch.full((m, width), pad, dtype=torch.long)
            mask = torch.zeros((m, cut + width), dtype=torch.long)
            mask[:, :cut] = 1
            for r, row in enumerate(rows):
                ids[r, : len(row)] = torch.tensor(row)
                mask[r, cut : cut + len(row)] = 1
            extra: dict[str, Any] = {}
            if self.layout.mrope:
                positions = torch.arange(width) + prefix.next_position
                extra["position_ids"] = positions.view(1, 1, -1).expand(3, m, -1).to(self.device)
            last = [len(r) - 1 for r in rows]
            keep = sorted(set(last))
            past = copy.deepcopy(prefix.past)
            past.batch_repeat_interleave(m)
            with torch.inference_mode():
                logits = self.model(
                    input_ids=ids.to(self.device),
                    attention_mask=mask.to(self.device),
                    past_key_values=past,
                    use_cache=True,
                    logits_to_keep=torch.tensor(keep, device=self.device),
                    **extra,
                ).logits
            pick = logits[torch.arange(m), torch.tensor([keep.index(n) for n in last])]
            out += [(row, cut + len(r)) for row, r in zip(pick.float().cpu().numpy(), rows, strict=True)]
        return out

    def encode(self, text: str, images: list[Any], suffix: list[int]) -> Prefix | None:
        """Run the prefix of `text` once, with its images, and keep the KV cache.

        None when the processor's ids do not split into prefix + `suffix` at the end token.
        """
        import torch

        enc = self.processor(text=[text], images=images, return_tensors="pt")
        ids = enc["input_ids"][0].tolist()
        cut = len(ids) - len(suffix)
        if cut <= 0 or ids[cut:] != suffix or ids[cut - 1] != self.end_id:
            return None
        inputs: dict[str, Any] = {}
        for key, value in enc.items():
            if hasattr(value, "shape") and value.ndim == 2 and value.shape == enc["input_ids"].shape:
                value = value[:, :cut]  # per-token inputs: ids, mask, token types
            inputs[key] = value.to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        next_position = cut
        if self.layout.mrope:
            positions, _ = self.model.model.get_rope_index(
                input_ids=inputs["input_ids"],
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
                image_grid_thw=inputs.get("image_grid_thw"),
            )
            inputs["position_ids"] = positions
            next_position = int(positions.max()) + 1
        with torch.inference_mode():
            out = self.model(**inputs, use_cache=True, logits_to_keep=1)
        return Prefix(ids[:cut], out.past_key_values, next_position)
