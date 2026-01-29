from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List

import datasets

try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover
    hf_hub_download = None


_REPO_ID = "xinrongzhang2022/InfiniteBench"
_DEFAULT_SPLIT = "test"


@lru_cache(maxsize=None)
def _resolve_jsonl_path(filename: str) -> str:
    if hf_hub_download is None:
        raise RuntimeError(
            "huggingface_hub is required to auto-download InfiniteBench JSONL files. "
            "Install huggingface_hub or pass a local path via dataset_kwargs."
        )
    return hf_hub_download(repo_id=_REPO_ID, repo_type="dataset", filename=filename)


def _load_jsonl(filename: str, *, split_name: str = _DEFAULT_SPLIT, **kwargs) -> Dict[str, datasets.Dataset]:
    # Allow callers to override the file path (useful for offline runs).
    data_path = kwargs.get("data_path") or kwargs.get("data_file")
    if not data_path:
        data_path = _resolve_jsonl_path(filename)
    ds = datasets.load_dataset("json", data_files=data_path, split="train")
    return {split_name: ds}


def load_passkey(**kwargs) -> Dict[str, datasets.Dataset]:
    return _load_jsonl("passkey.jsonl", **kwargs)


def load_number_string(**kwargs) -> Dict[str, datasets.Dataset]:
    return _load_jsonl("number_string.jsonl", **kwargs)


def load_longbook_choice_eng(**kwargs) -> Dict[str, datasets.Dataset]:
    return _load_jsonl("longbook_choice_eng.jsonl", **kwargs)


_DIGITS_RE = re.compile(r"\d+")


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _extract_candidate_numbers(text: str) -> List[str]:
    if not text:
        return []
    return _DIGITS_RE.findall(text)


def _get_gold_answers(doc: dict) -> List[str]:
    # InfiniteBench stores answers as a list of strings, e.g. ["71432"].
    ans = doc.get("answer")
    if ans is None:
        return []
    if isinstance(ans, list):
        return [str(x) for x in ans if x is not None]
    return [str(ans)]


def process_results_exact_match(doc: dict, results: List[str]) -> Dict[str, float]:
    """
    Exact match for retrieval-style tasks (PassKey / Number).

    The model may produce extra text; we treat it as correct if any extracted digit
    sequence matches any reference answer exactly.
    """
    pred = results[0] if results else ""
    gold = _get_gold_answers(doc)
    gold_norm = {_normalize_text(x) for x in gold if x is not None}

    pred_norm = _normalize_text(pred)
    if pred_norm in gold_norm:
        return {"exact_match": 1.0}

    for c in _extract_candidate_numbers(pred_norm):
        if c in gold_norm:
            return {"exact_match": 1.0}

    # Fallback: substring match (helps when answer formatting differs slightly).
    for g in gold_norm:
        if g and g in pred_norm:
            return {"exact_match": 1.0}

    return {"exact_match": 0.0}

