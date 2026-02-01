from __future__ import annotations

import random
import re
from functools import cache
from typing import Dict, Iterable, List, Sequence

import datasets
from transformers import AutoTokenizer

_MAGIC_NUMBER_RE = re.compile(r"\b\d+\b")

_HAYSTACK_SENTENCE = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
)

_QA_TEMPLATE = "Question: {question}\nAnswer:"


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
    # lm-eval passes CLI `--model_args` and `--metadata` into TaskManager.metadata,
    # which then gets merged into each task config's `metadata`.
    # Our native model uses `tokenizer_path`, so accept it as a fallback.
    name = tokenizer or pretrained or kwargs.get("tokenizer_path")
    if not name:
        raise ValueError(
            "ruler_extra_utils requires `tokenizer`/`pretrained` (or `tokenizer_path`) in metadata."
        )
    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def _random_key(rng: random.Random) -> str:
    return f"item-{rng.randint(0, 999999):06d}"


def _random_word(rng: random.Random) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    return "".join(rng.choice(letters) for _ in range(8))


def _build_context_with_insert(
    *,
    tokenizer,
    inserts: Sequence[str],
    max_length: int,
    depth_percent: int,
    tokens_to_generate: int,
) -> str:
    """
    Build a long context comprised of repeated haystack sentences, with `inserts`
    injected around a given depth percent.

    This is intentionally simple (base-model friendly) and only approximates
    "depth" semantics for long-context stress testing.
    """
    hay_len = len(tokenizer(_HAYSTACK_SENTENCE).input_ids)
    insert_lens = sum(len(tokenizer(x).input_ids) for x in inserts)
    budget = max_length - tokens_to_generate - insert_lens
    n_hay = max(1, budget // max(1, hay_len))

    def build_with_n(n: int) -> str:
        n = max(1, n)
        insert_at = int(round((depth_percent / 100.0) * n))
        insert_at = max(0, min(n, insert_at))
        sents = [_HAYSTACK_SENTENCE] * n
        for offset, insert in enumerate(inserts):
            sents.insert(insert_at + offset, insert)
        return "\n".join(sents)

    context = build_with_n(n_hay)
    while True:
        total_len = len(tokenizer(context).input_ids) + tokens_to_generate
        if total_len <= max_length or n_hay <= 1:
            break
        n_hay -= 1
        context = build_with_n(n_hay)
    return context


def ruler_custom_qa_simple_dataset(
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
    tok = _get_tokenizer(tokenizer=tokenizer, pretrained=pretrained, **kwargs)
    seqs = _ordered_unique(int(x) for x in max_seq_lengths)
    depth_list = _ordered_unique(int(x) for x in depths)

    rng = random.Random(seed)
    rows: List[dict] = []
    for max_len in seqs:
        for depth in depth_list:
            for _ in range(max(1, int(num_samples_per_cell))):
                key = _random_key(rng)
                value = _random_word(rng)
                inserts = [f"Fact: The code for {key} is {value}."]
                context = _build_context_with_insert(
                    tokenizer=tok,
                    inserts=inserts,
                    max_length=max_len,
                    depth_percent=depth,
                    tokens_to_generate=tokens_to_generate,
                )
                question = _QA_TEMPLATE.format(question=f"What is the code for {key}?")
                rows.append(
                    {
                        "input": f"{context}\n\n{question}",
                        "outputs": [value],
                        # `_QA_TEMPLATE` already ends with "Answer:", so keep `gen_prefix`
                        # empty to avoid "Answer: Answer:" duplication.
                        "gen_prefix": "",
                        "max_length": max_len,
                        "depth_percent": depth,
                        "task": "qa",
                    }
                )
    return {"test": datasets.Dataset.from_list(rows, split=datasets.Split.TEST)}


def ruler_custom_multihop_dataset(
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
    tok = _get_tokenizer(tokenizer=tokenizer, pretrained=pretrained, **kwargs)
    seqs = _ordered_unique(int(x) for x in max_seq_lengths)
    depth_list = _ordered_unique(int(x) for x in depths)

    rng = random.Random(seed)
    rows: List[dict] = []
    for max_len in seqs:
        for depth in depth_list:
            for _ in range(max(1, int(num_samples_per_cell))):
                person = _random_key(rng)
                city = _random_key(rng)
                country = _random_key(rng)
                inserts = [
                    f"Fact: {person} is located in {city}.",
                    f"Fact: {city} is located in {country}.",
                ]
                context = _build_context_with_insert(
                    tokenizer=tok,
                    inserts=inserts,
                    max_length=max_len,
                    depth_percent=depth,
                    tokens_to_generate=tokens_to_generate,
                )
                question = _QA_TEMPLATE.format(question=f"In which country is {person}?")
                rows.append(
                    {
                        "input": f"{context}\n\n{question}",
                        "outputs": [country],
                        # `_QA_TEMPLATE` already ends with "Answer:", so keep `gen_prefix`
                        # empty to avoid "Answer: Answer:" duplication.
                        "gen_prefix": "",
                        "max_length": max_len,
                        "depth_percent": depth,
                        "task": "multihop",
                    }
                )
    return {"test": datasets.Dataset.from_list(rows, split=datasets.Split.TEST)}


def ruler_custom_aggregation_sum_dataset(
    *,
    tokenizer: str | None = None,
    pretrained: str | None = None,
    max_seq_lengths: Sequence[int] = (2048, 4096, 8192, 16384, 32768),
    depths: Sequence[int] = tuple(range(0, 101, 10)),
    num_samples_per_cell: int = 1,
    tokens_to_generate: int = 64,
    seed: int = 42,
    num_numbers: int = 8,
    max_number: int = 20,
    **kwargs,
) -> Dict[str, datasets.Dataset]:
    tok = _get_tokenizer(tokenizer=tokenizer, pretrained=pretrained, **kwargs)
    seqs = _ordered_unique(int(x) for x in max_seq_lengths)
    depth_list = _ordered_unique(int(x) for x in depths)

    rng = random.Random(seed)
    rows: List[dict] = []
    for max_len in seqs:
        for depth in depth_list:
            for _ in range(max(1, int(num_samples_per_cell))):
                key = _random_key(rng)
                numbers = [rng.randint(0, int(max_number)) for _ in range(max(1, int(num_numbers)))]
                total = str(sum(numbers))
                inserts = [
                    "Facts:\n" + "\n".join([f"- {key}: {n}" for n in numbers]) + "\nEnd of facts."
                ]
                context = _build_context_with_insert(
                    tokenizer=tok,
                    inserts=inserts,
                    max_length=max_len,
                    depth_percent=depth,
                    tokens_to_generate=tokens_to_generate,
                )
                question = _QA_TEMPLATE.format(question=f"What is the sum of all numbers for {key}?")
                rows.append(
                    {
                        "input": f"{context}\n\n{question}",
                        "outputs": [total],
                        # `_QA_TEMPLATE` already ends with "Answer:", so keep `gen_prefix`
                        # empty to avoid "Answer: Answer:" duplication.
                        "gen_prefix": "",
                        "max_length": max_len,
                        "depth_percent": depth,
                        "task": "aggregation_sum",
                    }
                )
    return {"test": datasets.Dataset.from_list(rows, split=datasets.Split.TEST)}


def process_results_exact_match(doc: dict, results: List[str]) -> Dict[str, float]:
    pred = results[0] if results else ""
    gold = [str(x) for x in (doc.get("outputs") or []) if x is not None]
    if not gold:
        return {"score": 0.0}

    pred_numbers = _MAGIC_NUMBER_RE.findall(str(pred))
    gold_set = set(gold)
    for n in pred_numbers:
        if n in gold_set:
            return {"score": 1.0}

    pred_norm = " ".join(str(pred).strip().split())
    for g in gold:
        if g and g in pred_norm:
            return {"score": 1.0}
    return {"score": 0.0}
