import logging
import re
from functools import cache
from typing import TYPE_CHECKING, Union

import datasets
from transformers import AutoTokenizer


if TYPE_CHECKING:
    import transformers


eval_logger = logging.getLogger(__name__)


@cache
def get_tokenizer(
    tokenizer=None, pretrained=None, **kwargs
) -> Union["transformers.PreTrainedTokenizer", "transformers.PreTrainedTokenizerFast"]:
    pretrained = tokenizer or pretrained
    assert pretrained, "No tokenizer or pretrained provided."
    eval_logger.info(f"Using tokenizer {pretrained} for babilong tasks.")
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


def load_dataset(**kwargs):
    config_name = kwargs.get("max_seq_lengths", "0k")

    # Get specific qa split
    qa_split = kwargs.get("qa_split")

    eval_logger.info(
        f"Loading babilong dataset: max_seq_lengths={config_name}, split={qa_split}"
    )
    dataset = datasets.load_dataset(
        "RMT-team/babilong-1k-samples", name=config_name, split=qa_split
    )
    return {qa_split: dataset}


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    pred = postprocess_pred(results)
    target = doc.get("target", "").strip()

    # Base-model friendly scoring:
    # - Prefer exact/substring match to the canonical target string.
    # - Fall back to matching the *key answer token* (usually the final location word).
    #
    # This avoids zeroing scores when the model answers correctly but doesn't follow
    # an exact templated sentence format.
    pred0 = pred[0] if pred else ""
    pred_low = pred0.lower()
    target_low = target.lower()

    score = 0.0
    if target_low and target_low in pred_low:
        score = 1.0
    else:
        # Extract final "answer token" (strip punctuation/quotes).
        # Examples:
        # - "The most recent location of Charlie is balcony." -> "balcony"
        # - "The bottle is in the balcony." -> "balcony"
        # - "kitchen" -> "kitchen"
        ans_token = re.sub(r"^[\"'`]+|[\"'`\\.,;:!?]+$", "", target.split()[-1]) if target else ""
        ans_token_low = ans_token.lower()
        if ans_token_low and ans_token_low in pred_low:
            score = 1.0

    return {"acc": score}
