import logging
import re
from functools import cache
from typing import TYPE_CHECKING, Union

from transformers import AutoTokenizer


if TYPE_CHECKING:
    import transformers


eval_logger = logging.getLogger(__name__)

DEFAULT_SEQ_LENGTHS = [
    4096,
]


@cache
def get_tokenizer(
    tokenizer=None, pretrained=None, **kwargs
) -> Union["transformers.PreTrainedTokenizer", "transformers.PreTrainedTokenizerFast"]:
    pretrained = tokenizer or pretrained
    assert pretrained, "No tokenizer or pretrained provided."
    eval_logger.info(f"Using tokenizer {pretrained} for synthetic tasks.")
    return AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)


def postprocess_pred(prediction: list[str]) -> list[str]:
    res = []
    for predict_str in prediction:
        predict_str = predict_str.strip()

        # Remove all non-printable characters
        np_pattern = re.compile(r"[\x00-\x1f]")
        predict_str = np_pattern.sub("\n", predict_str).strip()
        res.append(predict_str)

    return res


_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}\b"
)
_MAGIC_NUMBER_RE = re.compile(r"\b\d{7}\b")


def _infer_needle_type(outputs: list[str]) -> str:
    """Infer NIAH needle type from references."""
    if not outputs:
        return ""
    sample = str(outputs[0])
    if _UUID_RE.search(sample):
        return "uuids"
    if _MAGIC_NUMBER_RE.search(sample):
        return "numbers"
    return ""


def _extract_needles(text: str, needle_type: str) -> list[str]:
    if not text:
        return []
    if needle_type == "uuids":
        return _UUID_RE.findall(text)
    if needle_type == "numbers":
        return _MAGIC_NUMBER_RE.findall(text)
    return []


def postprocess_pred_for_niah(prediction: list[str], outputs: list[str]) -> list[str]:
    """Robust NIAH postprocess: extract 7-digit numbers / UUIDs when possible."""
    pred = postprocess_pred(prediction)
    needle_type = _infer_needle_type(outputs)
    if not needle_type:
        return pred
    out: list[str] = []
    for p in pred:
        needles = _extract_needles(p, needle_type)
        # If nothing extracted, keep raw to avoid breaking other tasks.
        out.append(" ".join(needles) if needles else p)
    return out


def string_match_all(preds: list[str], refs: list[list[str]]) -> float:
    score = sum(
        [
            sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
            for pred, ref in zip(preds, refs)
        ]
    ) / len(preds)
    return score


def string_match_part(preds: list[str], refs: list[list[str]]) -> float:
    score = max(
        [
            sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
            for pred, ref in zip(preds, refs)
        ]
    ) / len(preds)
    return score


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    # hacky: set all other lengths to -1
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}
    input_len = doc["max_length"]
    pred = postprocess_pred_for_niah(results, doc.get("outputs", []))
    score = string_match_all(pred, [doc["outputs"]])
    metrics[str(input_len)] = score
    return metrics


def process_results_part(doc: dict, results: list[str]) -> dict[str, float]:
    # hacky: set all other lengths to -1
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}
    input_len = doc["max_length"]
    pred = postprocess_pred_for_niah(results, doc.get("outputs", []))
    score = string_match_part(pred, [doc["outputs"]])
    metrics[str(input_len)] = score
    return metrics


def aggregate_metrics(metrics: list[float]) -> float:
    res = [x for x in metrics if x != -1]
    if not res:
        # we don't have any samples with this length
        return -1
    return sum(res) / len(res)
