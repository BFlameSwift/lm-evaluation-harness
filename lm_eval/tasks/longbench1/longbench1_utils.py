import re
from functools import cache
from typing import Any, Dict, Iterable, List


@cache
def _load_longbench_split(dataset_name: str, split: str):
    # Note: THUDM/LongBench requires `trust_remote_code=True`.
    from datasets import load_dataset

    ds = load_dataset("THUDM/LongBench", dataset_name, split=split, trust_remote_code=True)
    return ds


def load_passage_retrieval_en(**kwargs):
    # lm-eval expects a mapping of split->rows for custom datasets.
    # Accept **kwargs to tolerate lm-eval passing dataset_kwargs such as `version`.
    _ = kwargs
    return {"test": _load_longbench_split("passage_retrieval_en", "test")}


def _extract_paragraph_numbers(text: str) -> List[int]:
    if not text:
        return []
    lowered = text.lower()

    # Prefer explicit "paragraph N" mentions.
    nums = [int(n) for n in re.findall(r"paragraph\s*(\d+)", lowered)]
    if nums:
        return nums

    # Fallback: first integer anywhere.
    m = re.search(r"(\d+)", lowered)
    return [int(m.group(1))] if m else []


def process_results_paragraph_retrieval(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    pred = results[0] if results else ""
    pred_nums = set(_extract_paragraph_numbers(pred))

    gold_answers: Iterable[str] = doc.get("answers") or []
    gold_nums = set()
    for ans in gold_answers:
        gold_nums.update(_extract_paragraph_numbers(ans))

    correct = 1.0 if (pred_nums and gold_nums and (pred_nums & gold_nums)) else 0.0
    return {"acc": correct}
