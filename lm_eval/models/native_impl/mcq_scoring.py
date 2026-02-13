"""Multiple-choice (MCQ) scoring helpers for the `native` model.

This module centralizes:
- parsing of MCQ/verifier-related model args
- verifier score computation (yes_only / yes_minus_no / yes_prob)
- prompt construction for the "choice verifier" path used in some MCQ setups

The intent is to keep these pure/stateless so likelihood code can import them
without dragging in the heavy `NativeCausalLM` implementation.
"""

from __future__ import annotations

import math
import re
from typing import List, Optional


def _parse_mcq_score_mode(name: Optional[str]) -> str:
    if name is None:
        return "ll"
    mode = str(name).strip().lower()
    if mode not in {"ll", "verifier", "yes_only", "yes_minus_no", "yes_prob"}:
        raise ValueError(f"Unsupported mcq_score_mode: {name}")
    return mode


def _map_choice_score_mode_to_mcq_mode(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    mode = str(name).strip().lower()
    if not mode:
        return None
    if mode in {"ll", "loglikelihood"}:
        return "ll"
    if mode in {"verifier", "yes_only", "yes_minus_no", "yes_prob"}:
        return mode
    # Backward-compat with eval_longbench modes; harness MCQ uses ll/verifier-style scores.
    if mode in {"label", "option", "both"}:
        return "ll"
    raise ValueError(f"Unsupported choice_score_mode: {name}")


def _parse_verifier_score_mode(name: Optional[str]) -> str:
    if name is None:
        return "yes_prob"
    mode = str(name).strip().lower()
    if mode not in {"yes_only", "yes_minus_no", "yes_prob"}:
        raise ValueError(f"Unsupported verifier_score_mode: {name}")
    return mode


def _parse_mcq_verifier_prompt_style(name: Optional[str]) -> str:
    if name is None:
        return "minimal"
    mode = str(name).strip().lower()
    if mode not in {"minimal", "explicit_yesno"}:
        raise ValueError(f"Unsupported mcq_verifier_prompt_style: {name}")
    return mode


def _parse_mcq_verifier_candidate_style(name: Optional[str]) -> str:
    if name is None:
        return "auto"
    mode = str(name).strip().lower()
    if mode not in {"auto", "with_label", "text_only"}:
        raise ValueError(f"Unsupported mcq_verifier_candidate_style: {name}")
    return mode


def _parse_mcq_verifier_tie_break(name: Optional[str]) -> str:
    if name is None:
        return "none"
    mode = str(name).strip().lower()
    if mode not in {"none", "ll_yes", "ll_margin", "choice_idx"}:
        raise ValueError(f"Unsupported mcq_verifier_tie_break: {name}")
    return mode


def _is_mcq_verifier_mode(mode: Optional[str]) -> bool:
    mode_s = str(mode or "").strip().lower()
    return mode_s in {"verifier", "yes_only", "yes_minus_no", "yes_prob"}


def _resolve_verifier_score_mode(mcq_score_mode: Optional[str], verifier_score_mode: str) -> str:
    mode_s = str(mcq_score_mode or "").strip().lower()
    if mode_s in {"yes_only", "yes_minus_no", "yes_prob"}:
        return mode_s
    return verifier_score_mode


def _parse_verifier_apply_norm(name: Optional[str]) -> str:
    if name is None:
        return "none"
    mode = str(name).strip().lower()
    if mode not in {"none", "sum", "length"}:
        raise ValueError(f"Unsupported verifier_apply_norm: {name}")
    return mode


def _map_choice_score_norm_to_verifier_norm(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    mode = str(name).strip().lower()
    if not mode:
        return None
    if mode in {"none", "sum", "length"}:
        return mode
    if mode in {"avg", "mean"}:
        return "length"
    raise ValueError(f"Unsupported choice_score_norm: {name}")


def _sigmoid_stable(x: float) -> float:
    if math.isnan(x):
        return 0.5
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _verifier_score_from_ll(ll_yes: float, ll_no: float, mode: str) -> float:
    mode_s = str(mode or "").strip().lower()
    if mode_s == "yes_only":
        return float(ll_yes)
    diff = float(ll_yes) - float(ll_no)
    if mode_s == "yes_minus_no":
        if math.isnan(diff):
            return float("-inf")
        return float(diff)
    if mode_s == "yes_prob":
        return float(_sigmoid_stable(diff))
    raise ValueError(f"Invalid verifier_score_mode: {mode}")


def _apply_mcq_verifier_tie_break(
    score: float,
    *,
    ll_yes: float,
    ll_no: float,
    choice_idx: Optional[int],
    mode: str,
    eps: float = 1e-8,
) -> float:
    mode_s = str(mode or "").strip().lower()
    if mode_s == "none":
        return float(score)

    signal = 0.0
    if mode_s == "ll_yes":
        signal = float(ll_yes) if math.isfinite(float(ll_yes)) else -1.0e9
    elif mode_s == "ll_margin":
        diff = float(ll_yes) - float(ll_no)
        signal = float(diff) if math.isfinite(diff) else -1.0e9
    elif mode_s == "choice_idx":
        idx = int(choice_idx) if choice_idx is not None else 0
        signal = -float(idx)
    else:
        raise ValueError(f"Invalid mcq_verifier_tie_break: {mode}")
    return float(score) + float(eps) * float(signal)


def _apply_verifier_score_norm(score: float, candidate_tokens: int, mode: str) -> float:
    mode_s = str(mode or "").strip().lower()
    if mode_s == "none":
        return float(score)
    denom = max(1, int(candidate_tokens))
    if mode_s == "length":
        return float(score) / float(denom)
    if mode_s == "sum":
        return float(score) * float(denom)
    raise ValueError(f"Invalid verifier_apply_norm: {mode}")


def _build_verifier_variant_texts(seed: Optional[str], fallback: str) -> List[str]:
    base = str(seed) if seed is not None else ""
    if not base.strip():
        base = fallback
    stem = base.strip()
    cands = [
        base,
        stem,
        f" {stem}",
        stem.lower(),
        f" {stem.lower()}",
        stem.capitalize(),
        f" {stem.capitalize()}",
    ]
    out: List[str] = []
    seen = set()
    for c in cands:
        if not c:
            continue
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _normalize_verifier_question_context(text: Optional[str], max_chars: int = 1200) -> str:
    ctx = str(text or "").strip()
    if not ctx:
        return ""
    # Keep spacing readable while preserving line structure when available.
    ctx = re.sub(r"[ \t]+\n", "\n", ctx)
    ctx = re.sub(r"\n{3,}", "\n\n", ctx)
    if len(ctx) > max_chars:
        ctx = ctx[-max_chars:]
    return ctx


def _build_choice_verifier_prompt_text(
    label: str,
    opt_text: str,
    prompt_style: str = "minimal",
    question_context: str = "",
    options_context: str = "",
) -> str:
    label_s = str(label or "").strip()
    opt_s = str(opt_text or "").strip()
    if label_s and opt_s:
        candidate = f"({label_s}) {opt_s}"
    elif label_s:
        candidate = f"({label_s})"
    else:
        candidate = opt_s
    style = str(prompt_style or "minimal").strip().lower()
    qctx = _normalize_verifier_question_context(question_context)
    octx = _normalize_verifier_question_context(options_context)
    if style == "explicit_yesno":
        if qctx:
            if octx:
                return (
                    "Judge whether the following candidate answer is correct.\n"
                    "Question Context:\n"
                    f"{qctx}\n"
                    "Options:\n"
                    f"{octx}\n"
                    f"Candidate: {candidate}\n"
                    "Question: Is this candidate answer correct for the question context above?\n"
                    "Reply with Yes or No.\n"
                    "Answer:"
                )
            return (
                "Judge whether the following candidate answer is correct.\n"
                "Question Context:\n"
                f"{qctx}\n"
                f"Candidate: {candidate}\n"
                "Question: Is this candidate answer correct for the question context above?\n"
                "Reply with Yes or No.\n"
                "Answer:"
            )
        return (
            "Judge whether the following candidate answer is correct.\n"
            + (f"Options:\n{octx}\n" if octx else "")
            + f"Candidate: {candidate}\n"
            + "Question: Is this candidate answer correct?\n"
            + "Reply with Yes or No.\n"
            "Answer:"
        )
    if qctx:
        if octx:
            return (
                f"Question Context:\n{qctx}\n"
                f"Options:\n{octx}\n"
                f"Candidate: {candidate}\n"
                "Is this candidate answer correct? Answer:"
            )
        return (
            f"Question Context:\n{qctx}\n"
            f"Candidate: {candidate}\n"
            "Is this candidate answer correct? Answer:"
        )
    if octx:
        return (
            f"Options:\n{octx}\n"
            f"Candidate: {candidate}\n"
            "Is this candidate answer correct? Answer:"
        )
    return f"Candidate: {candidate}\nIs this candidate answer correct? Answer:"


def _extract_options_block_from_prompt_text(prompt_text: Optional[str], max_lines: int = 12) -> str:
    text = str(prompt_text or "")
    if not text:
        return ""
    lines = text.splitlines()
    out: List[str] = []
    seen = set()
    for line in lines:
        s = str(line).strip()
        if not s:
            continue
        m = re.match(r"^\\(?\\s*([A-Z])\\s*\\)?\\s*[\\.\\):\\-]\\s*(.+)$", s)
        if not m:
            continue
        label = m.group(1).upper().strip()
        body = re.sub(r"\\s+", " ", m.group(2)).strip()
        if not body:
            continue
        norm = (label, body.lower())
        if norm in seen:
            continue
        seen.add(norm)
        out.append(f"({label}) {body}")
        if len(out) >= int(max_lines):
            break
    return "\n".join(out)
