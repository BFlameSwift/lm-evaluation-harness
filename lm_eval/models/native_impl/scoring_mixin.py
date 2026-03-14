"""Scoring / verifier helpers for the native model.

This mixin contains multi-choice verifier utilities and low-level log-prob
scoring routines that are used by both the likelihood and generation paths.

Keeping these methods outside `model.py` makes the main adapter easier to read
and reduces the chance of accidentally coupling unrelated evaluation modes.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

from .mcq_scoring import (
    _build_choice_verifier_prompt_text,
    _extract_options_block_from_prompt_text,
    _is_mcq_verifier_mode,
    _normalize_verifier_question_context,
    _resolve_verifier_score_mode,
    _verifier_score_from_ll,
)
from .utils import token_embed as _token_embed


class ScoringMixin:
    """Mixin that provides verifier + logprob scoring helpers."""

    def _get_likelihood_prefix_tokens(self, mode: str) -> List[int]:
        """Return cached prefix tokens inserted before likelihood scoring.

        Some configs want to prepend a short marker before scoring (e.g. "Answer:")
        to match few-shot formatting. Prefixes are cached per mode to avoid
        repeated tokenization.
        """
        if mode == "reconstruct_first":
            cached = getattr(self, "_likelihood_prefix_tokens_reconstruct", None)
            if cached is None:
                text = getattr(self, "_likelihood_prefix_reconstruct", "")
                cached = self._tokenizer.encode(text, bos=False, eos=False) if text else []
                setattr(self, "_likelihood_prefix_tokens_reconstruct", cached)
            return cached
        if mode == "compress_answer":
            cached = getattr(self, "_likelihood_prefix_tokens_compress_answer", None)
            if cached is None:
                text = getattr(self, "_likelihood_prefix_compress_answer", "")
                cached = self._tokenizer.encode(text, bos=False, eos=False) if text else []
                setattr(self, "_likelihood_prefix_tokens_compress_answer", cached)
            return cached
        return []

    def _get_verifier_prompt_suffix_tokens(self) -> List[int]:
        """Return cached token ids appended to MCQ verifier prompts."""
        cached = getattr(self, "_verifier_prompt_suffix_tokens", None)
        if cached is None:
            text = getattr(self, "_verifier_prompt_suffix", "")
            cached = self._tokenizer.encode(text, bos=False, eos=False) if text else []
            self._verifier_prompt_suffix_tokens = cached
        return cached

    def _use_mcq_verifier(self) -> bool:
        """Whether the current run uses the yes/no verifier scoring path."""
        return _is_mcq_verifier_mode(getattr(self, "_mcq_score_mode", "ll"))

    def _active_verifier_score_mode(self) -> str:
        """Resolve the effective yes/no conversion mode (`yes_*`)."""
        return _resolve_verifier_score_mode(
            getattr(self, "_mcq_score_mode", "ll"),
            getattr(self, "_verifier_score_mode", "yes_prob"),
        )

    def _resolve_choice_text_from_doc(self, doc: Optional[dict], choice_idx: Optional[int]) -> Optional[str]:
        """Best-effort lookup of option text for a given MCQ choice index.

        Supported doc formats (common across harness tasks):
        - MMLU: `{"choices": [str, ...]}`
        - ARC: `{"choices": {"text": [...], "label": [...]}}`
        - HellaSwag: `{"endings": [str, ...]}` (after `process_docs`)
        """
        if not isinstance(doc, dict):
            return None
        if choice_idx is None:
            return None
        idx = int(choice_idx)
        if idx < 0:
            return None
        choices = doc.get("choices")
        # mmlu-style: {"choices": [str, ...]}
        if isinstance(choices, list):
            if idx < len(choices):
                value = choices[idx]
                return str(value) if value is not None else None
            return None
        # arc-style: {"choices": {"text":[...], "label":[...]}}
        if isinstance(choices, dict):
            text_list = choices.get("text")
            if isinstance(text_list, list) and idx < len(text_list):
                value = text_list[idx]
                return str(value) if value is not None else None
            return None
        # hellaswag-style after process_docs.
        endings = doc.get("endings")
        if isinstance(endings, list) and idx < len(endings):
            value = endings[idx]
            return str(value) if value is not None else None
        return None

    @staticmethod
    def _choice_label_from_index(idx: int) -> str:
        """Convert `0 -> A`, `1 -> B`, ... for verifier-friendly labeling."""
        if idx < 26:
            return chr(ord("A") + idx)
        return str(idx + 1)

    def _resolve_choice_options_from_doc(self, doc: Optional[dict]) -> str:
        """Build a short `(A) ...` options block from the doc for verifier prompts.

        This is intentionally truncated to keep verifier prompts bounded.
        """
        if not isinstance(doc, dict):
            return ""

        lines: List[str] = []

        def _push(label: str, text: Any) -> None:
            s = str(text) if text is not None else ""
            s = re.sub(r"\s+", " ", s).strip()
            if not s:
                return
            if len(s) > 240:
                s = s[:240]
            lines.append(f"({label}) {s}")

        choices = doc.get("choices")
        if isinstance(choices, list):
            for i, v in enumerate(choices[:12]):
                _push(self._choice_label_from_index(i), v)
        elif isinstance(choices, dict):
            text_list = choices.get("text")
            label_list = choices.get("label")
            if isinstance(text_list, list):
                for i, v in enumerate(text_list[:12]):
                    label = None
                    if isinstance(label_list, list) and i < len(label_list):
                        label = str(label_list[i]).strip()
                    if not label:
                        label = self._choice_label_from_index(i)
                    _push(label, v)
        else:
            endings = doc.get("endings")
            if isinstance(endings, list):
                for i, v in enumerate(endings[:12]):
                    _push(self._choice_label_from_index(i), v)

        return "\n".join(lines)

    def _build_verifier_candidate_tokens(
        self,
        *,
        continuation_tokens: List[int],
        continuation_str: str,
        doc: Optional[dict],
        choice_idx: Optional[int],
        context_tokens: Optional[List[int]] = None,
        context_text: Optional[str] = None,
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Build the *verifier prompt* tokens appended to a base LL prompt.

        In MCQ verifier modes (`mcq_score_mode in {verifier, yes_*}`), we do not
        directly compare option continuations via plain loglikelihood.

        Instead, for each MCQ option we:
        1) build a *single* base prompt (usually the task's LL prompt),
        2) append a structured "Candidate + Question/Options" block,
        3) score the loglikelihood of "Yes" vs "No" continuations.

        This helper returns the tokens for step (2) plus metadata for debugging.

        Candidate selection preference order:
        - If `doc` contains explicit option text for `choice_idx`, use that.
        - If the raw continuation is a label like "A"/"B"/"C"/"D", try mapping
          label -> doc option text.
        - Otherwise, fall back to the original continuation tokens.

        Notes on bias:
        - When the continuation is only a label, including the label again in
          the verifier prompt can introduce a strong positional prior. The
          default `candidate_style=auto` attempts to reduce this by using
          text-only candidates when possible.
        """
        choice_text = self._resolve_choice_text_from_doc(doc, choice_idx)
        label_text = str(continuation_str or "").strip()
        candidate_style = str(getattr(self, "_mcq_verifier_candidate_style", "auto") or "auto").strip().lower()
        meta: Dict[str, Any] = {
            "choice_idx": int(choice_idx) if choice_idx is not None else None,
            "source": "continuation",
            "candidate_style": candidate_style,
        }
        label_for_prompt = ""
        if label_text:
            label_clean = label_text.strip()
            if label_clean.startswith("(") and label_clean.endswith(")") and len(label_clean) >= 3:
                label_clean = label_clean[1:-1].strip()
            if len(label_clean) <= 8:
                label_for_prompt = label_clean

        if (not choice_text) and isinstance(doc, dict):
            # Some harness tasks return label-only continuations (e.g. "(C)" or "C").
            # If we can map that label back to a choice text from the doc, do so.
            label_u = str(label_for_prompt or "").strip().upper()
            if len(label_u) == 1 and ("A" <= label_u <= "Z"):
                idx_from_label = ord(label_u) - ord("A")
                choice_from_label = self._resolve_choice_text_from_doc(doc, idx_from_label)
                if choice_from_label:
                    choice_text = choice_from_label
                    meta["choice_idx_from_label"] = int(idx_from_label)

        if choice_text:
            candidate_opt = str(choice_text).strip()
            if label_for_prompt:
                candidate_opt = re.sub(
                    rf"^\(?\s*{re.escape(label_for_prompt)}\s*\)?\s*[:.)-]?\s*",
                    "",
                    candidate_opt,
                    flags=re.IGNORECASE,
                )
            # For MCQ letter continuations (A/B/C/D), including the label in verifier prompts
            # can introduce a strong positional prior. `auto` defaults to text-only in this case.
            if candidate_style == "text_only":
                label_for_prompt = ""
            elif candidate_style == "auto":
                if label_for_prompt and len(label_for_prompt) == 1 and label_for_prompt.isalpha():
                    label_for_prompt = ""
            meta["source"] = "doc_choice"
        else:
            candidate_opt = label_text
            label_for_prompt = ""

        if not candidate_opt:
            meta["source"] = "continuation_tokens"
            return list(continuation_tokens), meta

        # ------------------------------
        # Build question/options context
        # ------------------------------
        # The verifier prompt needs *some* grounding context. We try to extract:
        # - question text (preferred): doc["question"]/["query"]/["prompt"]/["input"]
        # - options block (preferred): doc["choices"]/["endings"] rendered as "(A) ..."
        # - and keep an excerpt from the original LL prompt tail as a formatting hint.
        question_context = ""
        if isinstance(doc, dict):
            for key in ("question", "query", "prompt", "input"):
                value = doc.get(key)
                if isinstance(value, str) and value.strip():
                    question_context = value.strip()
                    break
            if not question_context:
                value = doc.get("context")
                if isinstance(value, str) and value.strip():
                    question_context = value.strip()
        if not question_context and context_text:
            question_context = str(context_text).strip()
        if not question_context and context_tokens:
            try:
                tail_tokens = list(context_tokens)[-512:]
                question_context = self.tok_decode_w_special_tokens(tail_tokens)
            except Exception:
                try:
                    question_context = self._tokenizer.decode(list(context_tokens)[-512:])
                except Exception:
                    question_context = ""

        # For MCQ verifier scoring, keep an excerpt from the original prompt so
        # the verifier can inherit few-shot/task formatting cues from the
        # baseline LL prompt, not only the structured question field.
        prompt_tail = ""
        if context_tokens:
            try:
                tail_tokens = list(context_tokens)[-512:]
                prompt_tail = self.tok_decode_w_special_tokens(tail_tokens)
            except Exception:
                try:
                    prompt_tail = self._tokenizer.decode(list(context_tokens)[-512:])
                except Exception:
                    prompt_tail = ""
        prompt_tail = _normalize_verifier_question_context(prompt_tail, max_chars=480)
        if question_context:
            if prompt_tail and prompt_tail not in question_context:
                question_context = _normalize_verifier_question_context(
                    f"{question_context}\n\nPrompt Excerpt:\n{prompt_tail}",
                    max_chars=1200,
                )
        else:
            question_context = prompt_tail
        question_context = _normalize_verifier_question_context(question_context)
        options_context = self._resolve_choice_options_from_doc(doc)
        if not options_context and context_text:
            options_context = _extract_options_block_from_prompt_text(context_text)

        # Safety fallback: when both question/options context are unavailable, a verifier
        # prompt built from only "(A)/(B)/(C)/(D) option text" is often degenerate and can
        # push yes/no scoring below random. Fall back to continuation tokens in this case.
        if not question_context and not options_context:
            meta["source"] = "continuation_tokens_no_context"
            return list(continuation_tokens), meta

        prompt_text = "\n" + _build_choice_verifier_prompt_text(
            label_for_prompt,
            candidate_opt,
            prompt_style=getattr(self, "_mcq_verifier_prompt_style", "minimal"),
            question_context=question_context,
            options_context=options_context,
        )
        cand_tokens = self._tokenizer.encode(prompt_text, bos=False, eos=False)
        if not cand_tokens:
            meta["source"] = "continuation_tokens"
            return list(continuation_tokens), meta
        preview = prompt_text[:200]
        meta["candidate_preview"] = preview
        if question_context:
            meta["question_context_len"] = int(len(question_context))
        return cand_tokens, meta

    def _score_verifier_yes_no_from_base(
        self,
        *,
        base_embeds: torch.Tensor,
        base_comp_mask: torch.Tensor,
        decoder_budget: int,
        rows_per_chunk: int,
    ) -> Dict[str, Any]:
        """Score the verifier's Yes/No continuations given a *fixed* base prefix.

        Why variants exist:
        - Tokenizers can treat `"Yes"` vs `" Yes"` very differently.
        - Some tasks/chat templates may bias toward specific casing/spacing.

        We therefore score multiple Yes variants and multiple No variants, pick
        the best loglikelihood within each group, and then convert the two best
        scores into a scalar according to `yes_only|yes_minus_no|yes_prob`.

        Returns a dict with the winning variants and per-variant diagnostics.
        """
        yes_scores: List[Dict[str, Any]] = []
        for variant, toks in self._verifier_yes_variants:
            out = self._score_continuation_fixed_base(
                base_embeds=base_embeds,
                cont_tokens=list(toks),
                base_comp_mask=base_comp_mask,
                decoder_budget=decoder_budget,
                rows_per_chunk=rows_per_chunk,
            )
            yes_scores.append({"variant": variant, **out})
        no_scores: List[Dict[str, Any]] = []
        for variant, toks in self._verifier_no_variants:
            out = self._score_continuation_fixed_base(
                base_embeds=base_embeds,
                cont_tokens=list(toks),
                base_comp_mask=base_comp_mask,
                decoder_budget=decoder_budget,
                rows_per_chunk=rows_per_chunk,
            )
            no_scores.append({"variant": variant, **out})

        best_yes = max(yes_scores, key=lambda x: float(x.get("ll", float("-inf"))))
        best_no = max(no_scores, key=lambda x: float(x.get("ll", float("-inf"))))
        ll_yes = float(best_yes.get("ll", float("-inf")))
        ll_no = float(best_no.get("ll", float("-inf")))
        score = float(_verifier_score_from_ll(ll_yes, ll_no, self._active_verifier_score_mode()))
        greedy = bool(best_yes.get("greedy", False))
        return {
            "ll_yes": ll_yes,
            "ll_no": ll_no,
            "score": score,
            "greedy": greedy,
            "best_yes_variant": best_yes.get("variant"),
            "best_no_variant": best_no.get("variant"),
            "yes_variants": yes_scores,
            "no_variants": no_scores,
        }

    @torch.no_grad()
    def _score_continuation_on_tokens(
        self,
        *,
        base_tokens: List[int],
        continuation_tokens: List[int],
        decoder_budget: int,
    ) -> Dict[str, Any]:
        """Score a continuation on top of `base_tokens` using token IDs only.

        This is a *fallback* scorer used when we can't use the native fused
        decoder stack (`model.layers/norm/output`) and have to rely on generic
        HF-style `self._model_call`.

        To ensure we don't exceed `decoder_budget`, we tail-truncate the base
        (keep the suffix) so that `len(base)+len(cont)` fits.
        """
        cont = list(continuation_tokens or [])
        if not cont:
            return {"ll": 0.0, "greedy": True}

        keep_base = max(1, int(decoder_budget) - len(cont))
        base = list(base_tokens or [])
        if len(base) > keep_base:
            base = base[-keep_base:]
        seq = base + cont
        if len(seq) <= 1:
            return {"ll": float("-inf"), "greedy": False}

        seq_tokens = torch.tensor(seq, device=self.device, dtype=torch.long)
        cont_targets = torch.tensor(cont, device=self.device, dtype=torch.long)
        prefix_len = int(len(base))
        if prefix_len <= 0:
            return {"ll": float("-inf"), "greedy": False}

        # Native compressor-backed models expose fused layer stacks + norm and can
        # use our chunked continuation scorer directly.
        if hasattr(self.model, "layers") and hasattr(self.model, "norm"):
            with torch.autocast(device_type="cuda", dtype=self._dtype):
                seq_embeds = _token_embed(self.model, seq_tokens).to(dtype=self._dtype)
            seq_comp_mask = torch.zeros(int(seq_tokens.numel()), device=self.device, dtype=torch.bool)

            rows_per_chunk = int(getattr(getattr(self.model, "args", None), "cross_entropy_chunk", 8)) * 16
            rows_per_chunk = max(8, min(int(rows_per_chunk), 512))
            out = self._forward_score_continuations(
                seq_embeds=[seq_embeds],
                cont_targets=[cont_targets],
                prefix_lens=[prefix_len],
                comp_mask_list=[seq_comp_mask],
                rows_per_chunk=rows_per_chunk,
            )
            ps = (out.get("per_sample") or [{}])[0]
            return {
                "ll": float(ps.get("ll", float("-inf"))),
                "greedy": bool(ps.get("greedy", False)),
            }

        # Generic HF-style fallback (e.g., Qwen3ForCausalLM):
        # score continuation token-by-token using only the final-step logits to avoid
        # full-sequence float32 log_softmax materialization.
        total_ll = 0.0
        greedy_ok = True
        ctx = list(base)
        for tok in cont:
            if not ctx:
                return {"ll": float("-inf"), "greedy": False}
            inp = torch.tensor(ctx, device=self.device, dtype=torch.long).unsqueeze(0)
            logits = self._model_call(inp)
            if self._model_parallel_group is not None:
                from distributed.tensor_parallel import gather_from_model_parallel_region

                logits = gather_from_model_parallel_region(logits, self._model_parallel_group)
            last = logits[0, -1, :].float()
            tok_i = int(tok)
            if tok_i < 0 or tok_i >= int(last.shape[-1]):
                total_ll = float("-inf")
                greedy_ok = False
            elif math.isfinite(total_ll):
                lse = torch.logsumexp(last, dim=-1)
                total_ll += float((last[tok_i] - lse).item())
                if greedy_ok:
                    greedy_ok = int(last.argmax(dim=-1).item()) == tok_i
            ctx.append(tok_i)
            del logits, last

        return {"ll": float(total_ll), "greedy": bool(greedy_ok)}

    @torch.no_grad()
    def _score_verifier_yes_no_from_tokens(
        self,
        *,
        base_tokens: List[int],
        decoder_budget: int,
    ) -> Dict[str, Any]:
        """Token-based verifier scoring (no prompt_embeds / no fixed-base embeds).

        This path is slower than `_score_verifier_yes_no_from_base` because it
        re-runs the forward pass for each yes/no variant using token IDs.
        It's used as a compatibility fallback for HF models.
        """
        yes_scores: List[Dict[str, Any]] = []
        for variant, toks in self._verifier_yes_variants:
            out = self._score_continuation_on_tokens(
                base_tokens=base_tokens,
                continuation_tokens=list(toks),
                decoder_budget=decoder_budget,
            )
            yes_scores.append({"variant": variant, **out})

        no_scores: List[Dict[str, Any]] = []
        for variant, toks in self._verifier_no_variants:
            out = self._score_continuation_on_tokens(
                base_tokens=base_tokens,
                continuation_tokens=list(toks),
                decoder_budget=decoder_budget,
            )
            no_scores.append({"variant": variant, **out})

        best_yes = max(yes_scores, key=lambda x: float(x.get("ll", float("-inf"))))
        best_no = max(no_scores, key=lambda x: float(x.get("ll", float("-inf"))))
        ll_yes = float(best_yes.get("ll", float("-inf")))
        ll_no = float(best_no.get("ll", float("-inf")))
        score = float(_verifier_score_from_ll(ll_yes, ll_no, self._active_verifier_score_mode()))
        greedy = bool(best_yes.get("greedy", False))
        return {
            "ll_yes": ll_yes,
            "ll_no": ll_no,
            "score": score,
            "greedy": greedy,
            "best_yes_variant": best_yes.get("variant"),
            "best_no_variant": best_no.get("variant"),
            "yes_variants": yes_scores,
            "no_variants": no_scores,
        }

    @torch.no_grad()
    def _chunked_logprob_and_greedy(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        *,
        chunk_size: int = 32,
    ) -> Tuple[float, bool]:
        """
        Compute sum log-prob and greedy-match for token targets, using chunked full-vocab
        projection to avoid OOM.

        hidden: [N, d_model] (already normalized)
        targets: [N] (token ids)
        """
        n = int(targets.numel())
        if n == 0:
            return 0.0, True
        chunk_size = max(1, int(chunk_size))

        # When the full [N, vocab] projection is small, doing it in one shot:
        # - is faster (fewer kernel launches)
        # - avoids tiny numeric drift from per-chunk reductions
        #
        # When it's large, the [N, vocab] matrix can OOM (especially with fp32 logits),
        # so we fall back to chunking over rows.
        try:
            vocab = int(self.model.output.weight.shape[0])  # type: ignore[attr-defined]
        except Exception:
            vocab = 0
        if self._model_parallel_group is None and vocab > 0:
            # float32 logits would be ~4 bytes * N * vocab
            max_full_elems = 2_000_000
            if n * vocab <= max_full_elems:
                logits = self.model.output(hidden).float()
                logprobs = F.log_softmax(logits, dim=-1)
                invalid = (targets < 0) | (targets >= int(logprobs.shape[-1]))
                if bool(invalid.any().item()):
                    safe_tgt = targets.clone()
                    safe_tgt[invalid] = 0
                    gathered = logprobs.gather(-1, safe_tgt.unsqueeze(-1)).squeeze(-1)
                    gathered[invalid] = float("-inf")
                    total_lp = float(gathered.sum().item())
                    greedy_ok = bool((logprobs.argmax(dim=-1) == safe_tgt).all().item()) and not bool(invalid.any().item())
                else:
                    total_lp = float(logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum().item())
                    greedy_ok = bool((logprobs.argmax(dim=-1) == targets).all().item())
                return total_lp, greedy_ok

        greedy_ok = True
        total_lp = 0.0
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            h = hidden[s:e]
            t = targets[s:e]
            logits = self.model.output(h).float()
            if self._model_parallel_group is not None:
                # Tensor-parallel: each rank holds a shard of the vocab projection.
                # Gather to get a full-vocab logits matrix before applying softmax.
                from distributed.tensor_parallel import gather_from_model_parallel_region

                logits = gather_from_model_parallel_region(logits, self._model_parallel_group)
            logprobs = F.log_softmax(logits, dim=-1)
            invalid = (t < 0) | (t >= int(logprobs.shape[-1]))
            if bool(invalid.any().item()):
                safe_t = t.clone()
                safe_t[invalid] = 0
                gathered = logprobs.gather(-1, safe_t.unsqueeze(-1)).squeeze(-1)
                gathered[invalid] = float("-inf")
                total_lp += float(gathered.sum().item())
                greedy_ok = False
            else:
                total_lp += float(logprobs.gather(-1, t.unsqueeze(-1)).squeeze(-1).sum().item())
            if greedy_ok:
                greedy_ok = bool((logprobs.argmax(dim=-1) == t).all().item())
        return total_lp, greedy_ok

    @torch.no_grad()
    def _forward_score_token_ranges(
        self,
        *,
        seq_embeds: List[torch.Tensor],
        seq_token_ids: List[torch.Tensor],
        score_token_ranges: List[Tuple[int, int]],
        normalize_lengths: Optional[List[float]] = None,
        comp_mask_list: Optional[List[torch.Tensor]] = None,
        rows_per_chunk: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Forward a ragged batch (provided as per-sample decoder embeddings) and compute
        log-likelihood over *specified target token ranges*.

        This is intended as a reusable building block for:
        - multi-choice scoring where you reuse a shared prefix,
        - scoring only parts of a long continuation,
        - cases where you want to inject extra query text before likelihood.

        Alignment rule:
          Given `token_ids` aligned with `embeds`, scoring target tokens
          `token_ids[t0:t1]` uses logits at positions `[t0-1 : t1-1]`
          (standard next-token prediction).

        Args:
          seq_embeds: list of [L_i, d_model] tensors on CUDA (or will be moved)
          seq_token_ids: list of [L_i] token-id tensors aligned with seq_embeds
          score_token_ranges: list of (t0, t1) ranges in token indices, per sample
            - scores tokens token_ids[t0:t1]
            - requires t0 >= 1
          normalize_lengths: optional per-sample denominators for length-normalized ll
            (if None, uses #scored tokens)
          comp_mask_list: optional per-sample boolean masks for placeholder positions
            (not used by attention layers, kept for compatibility/debug)
          rows_per_chunk: chunk size for projecting hidden -> vocab (controls peak memory)

        Returns:
          {
            "per_sample": [
              {"ll": float, "ll_norm": float, "loss": float, "ppl": float|None,
               "tokens": int, "greedy": bool},
              ...
            ],
            "total": {"ll": float, "tokens": int, "loss": float|None, "ppl": float|None}
          }
        """
        n = len(seq_embeds)
        if len(seq_token_ids) != n or len(score_token_ranges) != n:
            raise ValueError(
                f"seq_embeds/seq_token_ids/score_token_ranges must have same length; "
                f"got {n}/{len(seq_token_ids)}/{len(score_token_ranges)}"
            )
        if normalize_lengths is not None and len(normalize_lengths) != n:
            raise ValueError(f"normalize_lengths must have length {n}, got {len(normalize_lengths)}")
        if comp_mask_list is not None and len(comp_mask_list) != n:
            raise ValueError(f"comp_mask_list must have length {n}, got {len(comp_mask_list)}")

        dec_lens: List[int] = []
        for i in range(n):
            e = seq_embeds[i]
            t = seq_token_ids[i]
            if e.ndim != 2:
                raise ValueError(f"seq_embeds[{i}] must be 2D [L,d]; got shape {tuple(e.shape)}")
            if t.ndim != 1:
                raise ValueError(f"seq_token_ids[{i}] must be 1D [L]; got shape {tuple(t.shape)}")
            if int(e.shape[0]) != int(t.numel()):
                raise ValueError(
                    f"seq_embeds[{i}] length {int(e.shape[0])} != seq_token_ids[{i}] length {int(t.numel())}"
                )
            dec_lens.append(int(e.shape[0]))

        if not dec_lens or sum(dec_lens) == 0:
            return {
                "per_sample": [
                    {"ll": 0.0, "ll_norm": 0.0, "loss": 0.0, "ppl": None, "tokens": 0, "greedy": True}
                    for _ in range(n)
                ],
                "total": {"ll": 0.0, "tokens": 0, "loss": None, "ppl": None},
            }

        # Build "ragged batch" metadata for our attention blocks.
        #
        # We flatten all sequences into one long token stream and represent boundaries
        # using cu_seqlens (prefix sums). Our `arch/` attention expects:
        # - cu_seqlens_{q,k}: [B+1] prefix sums
        # - positions: [sum(L_i)] positions per token (0..L_i-1 per sample)
        dec_cu = torch.tensor(
            [0] + list(torch.tensor(dec_lens).cumsum(0).tolist()),
            device=self.device,
            dtype=torch.int32,
        )
        max_dec = max(dec_lens)
        dec_positions = torch.cat([torch.arange(l, device=self.device, dtype=torch.int32) for l in dec_lens], dim=0)

        if comp_mask_list is not None:
            # `compression_token_mask` flags placeholder/memory tokens. Most attention layers
            # ignore it, but some compression-aware layers may use it (and we also keep it
            # for debug parity).
            comp_mask_flat = torch.cat(comp_mask_list, dim=0)
        else:
            comp_mask_flat = torch.zeros(sum(dec_lens), device=self.device, dtype=torch.bool)

        dec_ctx = {
            "cu_seqlens_q": dec_cu,
            "cu_seqlens_k": dec_cu,
            "max_seqlen_q": max_dec,
            "max_seqlen_k": max_dec,
            "positions": dec_positions,
            "compression_token_mask": comp_mask_flat,
        }

        # Forward pass over the flattened embeddings.
        #
        # Note: seq_embeds are already token embeddings / prompt_embeds. We do not run the
        # input embedding layer here; the caller owns embeddings construction.
        embeds_flat = torch.cat(seq_embeds, dim=0)
        with torch.autocast(device_type="cuda", dtype=self._dtype):
            h = embeds_flat
            for layer in self.model.layers:
                h = layer(h, context=dec_ctx)
            h = self.model.norm(h)

        # Build flattened score positions and targets.
        #
        # To score tokens token_ids[t0:t1], we need logits at positions (t0-1 .. t1-1).
        # We materialize those indices in the flattened token stream, then gather the
        # matching hidden states and project them to vocab.
        score_pos_chunks: List[torch.Tensor] = []
        score_tgt_chunks: List[torch.Tensor] = []
        score_ranges_flat: List[Tuple[int, int]] = []
        running = 0
        for i, (t0, t1) in enumerate(score_token_ranges):
            t0 = int(t0)
            t1 = int(t1)
            if t1 <= t0:
                score_ranges_flat.append((running, running))
                continue
            if t0 < 1:
                raise ValueError(
                    f"score_token_ranges[{i}] starts at {t0}, but must be >= 1 because logits at position t0-1 "
                    "are needed to score token_ids[t0]."
                )
            if t1 > dec_lens[i]:
                raise ValueError(f"score_token_ranges[{i}] ends at {t1}, exceeds sequence length {dec_lens[i]}")

            base = int(dec_cu[i].item())
            pos0 = base + (t0 - 1)
            length = t1 - t0
            pos1 = pos0 + length

            score_pos_chunks.append(torch.arange(pos0, pos1, device=self.device, dtype=torch.long))
            score_tgt_chunks.append(seq_token_ids[i][t0:t1].to(device=self.device, dtype=torch.long))
            score_ranges_flat.append((running, running + length))
            running += length

        # Flat buffers store per-scored-token stats. Later we reduce them back to per-sample.
        token_logprob = torch.empty(running, device=self.device, dtype=torch.float32)
        token_greedy_ok = torch.empty(running, device=self.device, dtype=torch.bool)

        if running > 0:
            score_pos = torch.cat(score_pos_chunks, dim=0)
            score_targets = torch.cat(score_tgt_chunks, dim=0)
            h_score = h.index_select(0, score_pos)

            if rows_per_chunk is None:
                # Chunking over rows is critical for large vocab models: the temporary
                # logits matrix scales with `rows * vocab`. Keep the default conservative.
                rows_per_chunk = int(getattr(getattr(self.model, "args", None), "cross_entropy_chunk", 8)) * 16
                rows_per_chunk = max(16, min(rows_per_chunk, 512))
            rows_per_chunk = max(8, int(rows_per_chunk))

            for off in range(0, running, rows_per_chunk):
                off2 = min(off + rows_per_chunk, running)
                h_chunk = h_score[off:off2]
                tgt_chunk = score_targets[off:off2]

                with torch.autocast(device_type="cuda", dtype=self._dtype):
                    logits_chunk = self.model.output(h_chunk)

                if self._model_parallel_group is not None:
                    from distributed.tensor_parallel import gather_from_model_parallel_region

                    logits_chunk = gather_from_model_parallel_region(logits_chunk, self._model_parallel_group)

                # Greedy-match is useful for lm-eval's "is_greedy" semantics.
                # For MCQ tasks, greedy-match is not the metric, but it is still
                # returned to match TemplateLM's contract.
                token_greedy_ok[off:off2] = logits_chunk.argmax(dim=-1).to(torch.long).eq(tgt_chunk)

                logits_f = logits_chunk.float()
                logprobs = F.log_softmax(logits_f, dim=-1)
                vocab = int(logprobs.shape[-1])
                invalid = (tgt_chunk < 0) | (tgt_chunk >= vocab)
                if bool(invalid.any().item()):
                    # If any targets are invalid (shouldn't happen for normal tokenizers),
                    # make the LL contribution -inf so the overall score is unusable.
                    safe_tgt = tgt_chunk.clone()
                    safe_tgt[invalid] = 0
                    gathered = logprobs.gather(-1, safe_tgt.unsqueeze(-1)).squeeze(-1)
                    gathered[invalid] = float("-inf")
                    token_logprob[off:off2] = gathered
                else:
                    token_logprob[off:off2] = logprobs.gather(-1, tgt_chunk.unsqueeze(-1)).squeeze(-1)
                del logits_f, logprobs, invalid

        # Reduce flat per-token stats back into per-sample summaries.
        per_sample: List[Dict[str, Any]] = []
        total_ll = 0.0
        total_toks = 0
        for i, (s, e) in enumerate(score_ranges_flat):
            s = int(s)
            e = int(e)
            nt = e - s
            if nt <= 0:
                ll = 0.0
                greedy = True
                ll_norm = 0.0
                loss = 0.0
                ppl = None
            else:
                ll = float(token_logprob[s:e].sum().item())
                greedy = bool(token_greedy_ok[s:e].all().item())
                denom = float(normalize_lengths[i]) if normalize_lengths is not None else float(nt)
                ll_norm = ll / denom if denom > 0 else ll
                loss = -ll / float(nt)
                try:
                    ppl = float(math.exp(loss))
                except OverflowError:
                    ppl = float("inf")
                total_ll += ll
                total_toks += nt
            per_sample.append({"ll": ll, "ll_norm": ll_norm, "loss": loss, "ppl": ppl, "tokens": nt, "greedy": greedy})

        if total_toks > 0:
            total_loss = -total_ll / float(total_toks)
            try:
                total_ppl = float(math.exp(total_loss))
            except OverflowError:
                total_ppl = float("inf")
        else:
            total_loss = None
            total_ppl = None

        return {"per_sample": per_sample, "total": {"ll": total_ll, "tokens": total_toks, "loss": total_loss, "ppl": total_ppl}}

    @torch.no_grad()
    def _forward_score_continuations(
        self,
        *,
        seq_embeds: List[torch.Tensor],
        cont_targets: List[torch.Tensor],
        prefix_lens: List[int],
        comp_mask_list: Optional[List[torch.Tensor]] = None,
        rows_per_chunk: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Forward a ragged batch and score continuation tokens starting at `prefix_lens[i]`.

        This function is the "common case" wrapper around `_forward_score_token_ranges`:
        - each sample has a single continuation segment (cont_targets[i])
        - the continuation starts immediately after a shared prefix of length `prefix_lens[i]`

        Alignment reminder:
        - to score target token at index `t`, we use logits from position `t-1`.
          Therefore `prefix_lens` must be >= 1.
        """
        n = len(seq_embeds)
        if len(cont_targets) != n or len(prefix_lens) != n:
            raise ValueError(
                "seq_embeds/cont_targets/prefix_lens must have same length; "
                f"got {n}/{len(cont_targets)}/{len(prefix_lens)}"
            )
        if comp_mask_list is not None and len(comp_mask_list) != n:
            raise ValueError(f"comp_mask_list must have length {n}, got {len(comp_mask_list)}")

        dec_lens: List[int] = []
        for i in range(n):
            e = seq_embeds[i]
            t = cont_targets[i]
            if e.ndim != 2:
                raise ValueError(f"seq_embeds[{i}] must be 2D [L,d]; got shape {tuple(e.shape)}")
            if t.ndim != 1:
                raise ValueError(f"cont_targets[{i}] must be 1D [T]; got shape {tuple(t.shape)}")
            dec_lens.append(int(e.shape[0]))

        if not dec_lens or sum(dec_lens) == 0:
            return {
                "per_sample": [
                    {"ll": 0.0, "ll_norm": 0.0, "loss": 0.0, "ppl": None, "tokens": 0, "greedy": True}
                    for _ in range(n)
                ],
                "total": {"ll": 0.0, "tokens": 0, "loss": None, "ppl": None},
            }

        # Same ragged-batch flattening strategy as `_forward_score_token_ranges`.
        dec_cu = torch.tensor(
            [0] + list(torch.tensor(dec_lens).cumsum(0).tolist()),
            device=self.device,
            dtype=torch.int32,
        )
        max_dec = max(dec_lens)
        dec_positions = torch.cat([torch.arange(l, device=self.device, dtype=torch.int32) for l in dec_lens], dim=0)

        if comp_mask_list is not None:
            comp_mask_flat = torch.cat(comp_mask_list, dim=0)
        else:
            comp_mask_flat = torch.zeros(sum(dec_lens), device=self.device, dtype=torch.bool)

        dec_ctx = {
            "cu_seqlens_q": dec_cu,
            "cu_seqlens_k": dec_cu,
            "max_seqlen_q": max_dec,
            "max_seqlen_k": max_dec,
            "positions": dec_positions,
            "compression_token_mask": comp_mask_flat,
        }

        embeds_flat = torch.cat(seq_embeds, dim=0)
        with torch.autocast(device_type="cuda", dtype=self._dtype):
            h = embeds_flat
            for layer in self.model.layers:
                h = layer(h, context=dec_ctx)
            h = self.model.norm(h)

        score_pos_chunks: List[torch.Tensor] = []
        score_tgt_chunks: List[torch.Tensor] = []
        score_ranges_flat: List[Tuple[int, int]] = []
        running = 0
        for i in range(n):
            cont_len = int(cont_targets[i].numel())
            pref_len = int(prefix_lens[i])
            # Score `cont_targets[i]` starting right after the prefix.
            #
            # To score token at index `pref_len`, we need logits from `pref_len-1`.
            rel_start = pref_len - 1
            rel_end = rel_start + cont_len
            if rel_start < 0 or rel_end > dec_lens[i] - 1 or cont_len <= 0:
                score_ranges_flat.append((running, running))
                continue

            base = int(dec_cu[i].item())
            pos0 = base + rel_start
            pos1 = pos0 + cont_len
            score_pos_chunks.append(torch.arange(pos0, pos1, device=self.device, dtype=torch.long))
            score_tgt_chunks.append(cont_targets[i])
            score_ranges_flat.append((running, running + cont_len))
            running += cont_len

        token_logprob = torch.empty(running, device=self.device, dtype=torch.float32)
        token_greedy_ok = torch.empty(running, device=self.device, dtype=torch.bool)

        if running > 0:
            score_pos = torch.cat(score_pos_chunks, dim=0)
            score_targets = torch.cat(score_tgt_chunks, dim=0)
            h_score = h.index_select(0, score_pos)

            if rows_per_chunk is None:
                rows_per_chunk = int(getattr(getattr(self.model, "args", None), "cross_entropy_chunk", 8)) * 16
                rows_per_chunk = max(8, min(rows_per_chunk, 512))
            rows_per_chunk = max(8, int(rows_per_chunk))

            for off in range(0, running, rows_per_chunk):
                off2 = min(off + rows_per_chunk, running)
                h_chunk = h_score[off:off2]
                tgt_chunk = score_targets[off:off2]

                with torch.autocast(device_type="cuda", dtype=self._dtype):
                    logits_chunk = self.model.output(h_chunk)

                if self._model_parallel_group is not None:
                    from distributed.tensor_parallel import gather_from_model_parallel_region

                    logits_chunk = gather_from_model_parallel_region(logits_chunk, self._model_parallel_group)

                token_greedy_ok[off:off2] = logits_chunk.argmax(dim=-1).to(torch.long).eq(tgt_chunk)

                logits_f = logits_chunk.float()
                lse = torch.logsumexp(logits_f, dim=-1)
                vocab = int(logits_f.shape[-1])
                invalid = (tgt_chunk < 0) | (tgt_chunk >= vocab)
                if bool(invalid.any().item()):
                    safe_tgt = tgt_chunk.clone()
                    safe_tgt[invalid] = 0
                    tgt_logits = logits_f.gather(-1, safe_tgt.unsqueeze(-1)).squeeze(-1)
                    tgt_logits[invalid] = float("-inf")
                else:
                    tgt_logits = logits_f.gather(-1, tgt_chunk.unsqueeze(-1)).squeeze(-1)
                token_logprob[off:off2] = (tgt_logits - lse)
                del logits_f, lse, tgt_logits, invalid

        per_sample: List[Dict[str, Any]] = []
        total_ll = 0.0
        total_toks = 0
        for i, (s, e) in enumerate(score_ranges_flat):
            s = int(s)
            e = int(e)
            nt = e - s
            if nt <= 0:
                ll = 0.0
                greedy = True
                ll_norm = 0.0
                loss = 0.0
                ppl = None
            else:
                ll = float(token_logprob[s:e].sum().item())
                greedy = bool(token_greedy_ok[s:e].all().item())
                ll_norm = ll / float(nt)
                loss = -ll / float(nt)
                try:
                    ppl = float(math.exp(loss))
                except OverflowError:
                    ppl = float("inf")
                total_ll += ll
                total_toks += nt
            per_sample.append({"ll": ll, "ll_norm": ll_norm, "loss": loss, "ppl": ppl, "tokens": nt, "greedy": greedy})

        if total_toks > 0:
            total_loss = -total_ll / float(total_toks)
            try:
                total_ppl = float(math.exp(total_loss))
            except OverflowError:
                total_ppl = float("inf")
        else:
            total_loss = None
            total_ppl = None

        return {"per_sample": per_sample, "total": {"ll": total_ll, "tokens": total_toks, "loss": total_loss, "ppl": total_ppl}}

    @torch.no_grad()
    def _score_continuation_fixed_base(
        self,
        *,
        base_embeds: torch.Tensor,
        cont_tokens: List[int],
        base_comp_mask: Optional[torch.Tensor] = None,
        decoder_budget: Optional[int] = None,
        rows_per_chunk: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Score `cont_tokens` conditioned on a fixed `base_embeds` prefix, optionally sliding
        over long continuations when `len(base)+len(cont) > decoder_budget`.

        Returns a dict compatible with debug rows:
          {"ll": float, "greedy": bool, "tokens": int, "loss": float|None, "ppl": float|None,
           "windows": int, "rolled": bool}
        """
        budget = int(self.decoder_budget if decoder_budget is None else decoder_budget)
        base_len = int(base_embeds.shape[0])
        cont_len = int(len(cont_tokens))
        if cont_len <= 0:
            return {"ll": 0.0, "greedy": True, "tokens": 0, "loss": 0.0, "ppl": None, "windows": 0, "rolled": False}
        if base_len <= 0:
            return {"ll": float("-inf"), "greedy": False, "tokens": 0, "loss": float("inf"), "ppl": float("inf"), "windows": 0, "rolled": True}

        if base_comp_mask is None:
            base_comp_mask = torch.zeros(base_len, device=self.device, dtype=torch.bool)
        else:
            base_comp_mask = base_comp_mask.to(device=self.device, dtype=torch.bool)
            if int(base_comp_mask.numel()) != base_len:
                base_comp_mask = torch.zeros(base_len, device=self.device, dtype=torch.bool)

        avail = budget - base_len
        if avail <= 0:
            return {"ll": float("-inf"), "greedy": False, "tokens": 0, "loss": float("inf"), "ppl": float("inf"), "windows": 0, "rolled": True}

        if rows_per_chunk is None:
            rows_per_chunk = int(getattr(getattr(self.model, "args", None), "cross_entropy_chunk", 8)) * 16
            rows_per_chunk = max(16, min(int(rows_per_chunk), 512))

        def _score_window(prefix: torch.Tensor, prefix_mask: torch.Tensor, targets: List[int], prefix_len: int) -> Tuple[float, bool]:
            if not targets:
                return 0.0, True
            t = torch.tensor(targets, device=self.device, dtype=torch.long)
            with torch.autocast(device_type="cuda", dtype=self._dtype):
                e = _token_embed(self.model, t).to(dtype=self._dtype)
            seq = torch.cat([prefix, e], dim=0)
            mask = torch.cat([prefix_mask, torch.zeros(int(t.numel()), device=self.device, dtype=torch.bool)], dim=0)
            out = self._forward_score_continuations(
                seq_embeds=[seq],
                cont_targets=[t],
                prefix_lens=[int(prefix_len)],
                comp_mask_list=[mask],
                rows_per_chunk=rows_per_chunk,
            )
            ps = (out.get("per_sample") or [{}])[0]
            return float(ps.get("ll", 0.0)), bool(ps.get("greedy", True))

        # Fits in one window: [base | cont] fits under decoder budget.
        if cont_len <= avail:
            ll, greedy = _score_window(base_embeds, base_comp_mask, cont_tokens, base_len)
            tokens = cont_len
            if tokens > 0 and math.isfinite(ll):
                loss = -ll / float(tokens)
                try:
                    ppl = float(math.exp(loss))
                except OverflowError:
                    ppl = float("inf")
            else:
                loss = float("inf")
                ppl = float("inf")
            return {"ll": ll, "greedy": greedy, "tokens": tokens, "loss": loss, "ppl": ppl, "windows": 1, "rolled": False}

        # Need rolling:
        #
        # If `base_len + cont_len > budget`, we cannot score the whole continuation in one forward.
        # We instead use a sliding window where each window has the fixed base prefix plus a chunk
        # of continuation tokens.
        #
        # IMPORTANT: We overlap windows by exactly 1 token. This preserves the standard next-token
        # scoring alignment at window boundaries (logits at position t score token t+1).
        #
        # We require >=2 available continuation positions so the overlapped window can score at
        # least 1 new token.
        if avail < 2:
            return {"ll": float("-inf"), "greedy": False, "tokens": 0, "loss": float("inf"), "ppl": float("inf"), "windows": 0, "rolled": True}

        total_ll = 0.0
        greedy_all = True
        windows = 0

        # Window 0: score the first `avail` tokens.
        ll0, g0 = _score_window(base_embeds, base_comp_mask, cont_tokens[:avail], base_len)
        total_ll += ll0
        greedy_all = greedy_all and g0
        windows += 1

        # Subsequent windows: overlap by 1 token (see comment above).
        step = avail - 1
        start = avail - 1
        while start < cont_len - 1:
            end = min(cont_len, start + avail)
            overlap_id = int(cont_tokens[start])
            overlap_t = torch.tensor([overlap_id], device=self.device, dtype=torch.long)
            with torch.autocast(device_type="cuda", dtype=self._dtype):
                overlap_e = _token_embed(self.model, overlap_t).to(dtype=self._dtype)
            prefix = torch.cat([base_embeds, overlap_e], dim=0)
            prefix_mask = torch.cat([base_comp_mask, torch.zeros(1, device=self.device, dtype=torch.bool)], dim=0)
            llw, gw = _score_window(prefix, prefix_mask, cont_tokens[start + 1 : end], base_len + 1)
            total_ll += llw
            greedy_all = greedy_all and gw
            windows += 1
            if end >= cont_len:
                break
            start += step

        tokens = cont_len
        if tokens > 0 and math.isfinite(total_ll):
            loss = -total_ll / float(tokens)
            try:
                ppl = float(math.exp(loss))
            except OverflowError:
                ppl = float("inf")
        else:
            loss = float("inf")
            ppl = float("inf")
        return {
            "ll": total_ll,
            "greedy": greedy_all,
            "tokens": tokens,
            "loss": loss,
            "ppl": ppl,
            "windows": windows,
            "rolled": True,
        }
    
