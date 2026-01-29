from __future__ import annotations

import random
import re
from functools import cache
from typing import Dict, Iterable, List, Sequence

import datasets
from transformers import AutoTokenizer

_MAGIC_NUMBER_RE = re.compile(r"\b\d{7}\b")

_HAYSTACK_SENTENCE = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
)
_NEEDLE_TEMPLATE = "One of the special magic numbers for {key} is: {value}."
_QUESTION_TEMPLATE = (
    "Question: What is the special magic number for {key} mentioned in the provided text?\nAnswer:"
)
_GEN_PREFIX_TEMPLATE = "The special magic number for {key} mentioned in the provided text is"


def _ordered_unique(xs: Iterable[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(int(x))
    return out


@cache
def _get_tokenizer(tokenizer: str | None = None, pretrained: str | None = None, **kwargs):
    name = tokenizer or pretrained
    if not name:
        raise ValueError("ruler_heatmap_utils requires `tokenizer` or `pretrained` in metadata.")
    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def _random_key(rng: random.Random) -> str:
    # Use a stable, tokenizer-friendly key format.
    return f"item-{rng.randint(0, 999999):06d}"


def _random_magic_number(rng: random.Random) -> str:
    return str(rng.randint(1_000_000, 9_999_999))


def _build_context(
    *,
    tokenizer,
    key: str,
    value: str,
    max_length: int,
    depth_percent: int,
    tokens_to_generate: int,
) -> str:
    needle = _NEEDLE_TEMPLATE.format(key=key, value=value)
    question = _QUESTION_TEMPLATE.format(key=key)

    # Estimate how many haystack sentences we can fit.
    hay_len = len(tokenizer(_HAYSTACK_SENTENCE).input_ids)
    needle_len = len(tokenizer(needle).input_ids)
    question_len = len(tokenizer(question).input_ids)

    # Conservative estimate; we'll shrink until it fits.
    budget = max_length - tokens_to_generate - needle_len - question_len
    n_hay = max(1, budget // max(1, hay_len))

    def build_with_n(n: int) -> str:
        n = max(1, n)
        insert_at = int(round((depth_percent / 100.0) * n))
        insert_at = max(0, min(n, insert_at))
        sents = [_HAYSTACK_SENTENCE] * n
        sents.insert(insert_at, needle)
        return "\n".join(sents)

    context = build_with_n(n_hay)
    while True:
        total_len = len(tokenizer(context + "\n\n" + question).input_ids) + tokens_to_generate
        if total_len <= max_length or n_hay <= 1:
            break
        n_hay -= 1
        context = build_with_n(n_hay)
    return context


def niah_heatmap_numbers(
    *,
    tokenizer: str | None = None,
    pretrained: str | None = None,
    max_seq_lengths: Sequence[int] = (2048, 4096, 8192, 16384, 32768),
    depths: Sequence[int] = tuple(range(0, 101, 10)),
    num_samples_per_cell: int = 1,
    tokens_to_generate: int = 64,
    seed: int = 42,
    **kwargs,
) -> Dict[str, datasets.Dataset]:
    """
    Generate a vanilla NIAH-style dataset with explicit (depth_percent × max_length) grid.

    This is intended for base-model evaluation + heatmap visualization, not for matching
    NVIDIA's exact RULER synthetic generation.
    """
    tok = _get_tokenizer(tokenizer=tokenizer, pretrained=pretrained, **kwargs)
    seqs = _ordered_unique(int(x) for x in max_seq_lengths)
    depth_list = _ordered_unique(int(x) for x in depths)

    rng = random.Random(seed)
    rows: List[dict] = []
    for max_len in seqs:
        for depth in depth_list:
            for _ in range(max(1, int(num_samples_per_cell))):
                key = _random_key(rng)
                value = _random_magic_number(rng)
                context = _build_context(
                    tokenizer=tok,
                    key=key,
                    value=value,
                    max_length=max_len,
                    depth_percent=depth,
                    tokens_to_generate=tokens_to_generate,
                )
                question = _QUESTION_TEMPLATE.format(key=key)
                rows.append(
                    {
                        "input": f"{context}\n\n{question}",
                        "outputs": [value],
                        "max_length": max_len,
                        "depth_percent": depth,
                        "needle_type": "numbers",
                        "key": key,
                        "gen_prefix": _GEN_PREFIX_TEMPLATE.format(key=key),
                    }
                )

    return {"test": datasets.Dataset.from_list(rows, split=datasets.Split.TEST)}


def process_results_niah_numbers(doc: dict, results: List[str]) -> Dict[str, float]:
    pred = results[0] if results else ""
    gold = [str(x) for x in (doc.get("outputs") or []) if x is not None]
    gold_set = set(gold)

    pred_numbers = _MAGIC_NUMBER_RE.findall(str(pred))
    for n in pred_numbers:
        if n in gold_set:
            return {"score": 1.0}

    # Fallback: normalized substring match.
    pred_norm = " ".join(str(pred).strip().split())
    for g in gold:
        if g and g in pred_norm:
            return {"score": 1.0}

    return {"score": 0.0}

