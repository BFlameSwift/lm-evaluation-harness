"""
Likelihood / scoring implementations for the `native` model.

This module contains the heavy log-likelihood code paths that were previously
inline in `native_impl/model.py`.

The functions are written as `self`-style helpers so `NativeCausalLM` can simply
delegate to them.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.ae_loader import (
    BEGIN_OF_MEMORY_INDEX,
    END_OF_MEMORY_INDEX,
    BEGIN_OF_RECONSTRUCTION_INDEX,
    END_OF_RECONSTRUCTION_INDEX,
)
from data.retrieval_loader import BEGIN_OF_QUERY_INDEX
from lm_eval.models.native_doc_utils import get_doc_query_keys_by_task_name, split_doc_and_query

from .model import _apply_mcq_verifier_tie_break, _apply_verifier_score_norm, _token_embed

_split_doc_and_query = split_doc_and_query
def _loglikelihood_tokens(
    self,
    requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
    disable_tqdm: bool = False,
    override_bs: Optional[int] = None,
) -> List[Tuple[float, bool]]:
    has_comp = hasattr(self.model, "compression_embeddings")
    # Optional mode: compress the context into memory first, then score the answer.
    if self._mode == "compress_answer" and has_comp:
        out = self._loglikelihood_tokens_compress_answer(requests, disable_tqdm, override_bs)

        return out
    if self._mode == "reconstruct_first" and has_comp:
        out = self._loglikelihood_tokens_reconstruct_first(requests, disable_tqdm)

        return out
    if self._mode in {"compress_answer", "reconstruct_first"} and not has_comp:
        if (self._distributed_args.rank == 0) and (not getattr(self, "_warned_mode_fallback_to_decoder", False)):
            print(
                f"[native] WARNING: mode={self._mode} requested, but model has no compression embeddings; "
                "falling back to decoder LL scoring."
            )
            self._warned_mode_fallback_to_decoder = True

    res: List[Tuple[float, bool]] = []
    bs = override_bs or self.batch_size
    try:
        bs = int(bs)
    except Exception:
        bs = 1
    rows_per_chunk = int(getattr(getattr(self.model, "args", None), "cross_entropy_chunk", 8)) * 16
    rows_per_chunk = max(16, min(int(rows_per_chunk), 512))
    iterator = range(0, len(requests), bs)
    pbar = tqdm(iterator, disable=disable_tqdm or self.rank != 0, desc="native loglikelihood")
    for batch_start in pbar:
        chunk = requests[batch_start : batch_start + bs]
        use_mcq_verifier = self._use_mcq_verifier()
        docs_slice: List[Optional[dict]] = [None] * len(chunk)
        choice_idxs_slice: List[Optional[int]] = [None] * len(chunk)
        if use_mcq_verifier:
            try:
                if self._active_loglikelihood_docs:
                    for bi in range(len(chunk)):
                        gi = batch_start + bi
                        if gi < len(self._active_loglikelihood_docs):
                            v = self._active_loglikelihood_docs[gi]
                            docs_slice[bi] = v if isinstance(v, dict) else None
                if self._active_loglikelihood_choice_idxs:
                    for bi in range(len(chunk)):
                        gi = batch_start + bi
                        if gi < len(self._active_loglikelihood_choice_idxs):
                            cidx = self._active_loglikelihood_choice_idxs[gi]
                            if cidx is not None:
                                choice_idxs_slice[bi] = int(cidx)
            except Exception:
                pass

        seq_lens = [min(self.max_length, len(c) + len(cont)) for _, c, cont in chunk]
        max_seq = max(seq_lens)
        inp_batch = torch.full((len(chunk), max_seq - 1), self.pad_token_id, device=self.device, dtype=torch.long)
        tgt_batch: List[List[int]] = []
        cont_lens: List[int] = []
        tgt_full_lens: List[int] = []
        for i, (_, context_enc, continuation_enc) in enumerate(chunk):
            tokens = (context_enc + continuation_enc)[-(max_seq):]
            if len(tokens) <= 1:
                tokens = [self.eot_token_id, self.eot_token_id]
            inp = tokens[:-1]
            tgt = tokens[1:]
            inp_batch[i, -len(inp) :] = torch.tensor(inp, device=self.device)
            tgt_batch.append(tgt)
            cont_lens.append(len(continuation_enc))
            tgt_full_lens.append(len(tgt))

        logits = self._model_call(inp_batch)
        logprobs = self._model_logits(logits)

        debug_rows: List[dict] = []
        for i, tgt in enumerate(tgt_batch):
            cont_len = int(cont_lens[i])
            verifier_stats: Optional[Dict[str, Any]] = None

            if use_mcq_verifier:
                try:
                    pair = chunk[i][0]
                    cont_str = pair[1] if pair and len(pair) > 1 else ""
                except Exception:
                    cont_str = ""
                candidate_tokens, candidate_meta = self._build_verifier_candidate_tokens(
                    continuation_tokens=list(chunk[i][2]),
                    continuation_str=str(cont_str),
                    doc=docs_slice[i],
                    choice_idx=choice_idxs_slice[i],
                    context_tokens=list(chunk[i][1]),
                    context_text=(chunk[i][0][0] if chunk[i] and chunk[i][0] and len(chunk[i][0]) > 0 else ""),
                )
                suffix_tokens = self._get_verifier_prompt_suffix_tokens()
                base_tokens = list(chunk[i][1]) + list(candidate_tokens) + list(suffix_tokens)
                max_verify_len = 1
                for _, toks in (self._verifier_yes_variants + self._verifier_no_variants):
                    if len(toks) > max_verify_len:
                        max_verify_len = len(toks)
                keep_base = max(1, int(self.max_length) - int(max_verify_len))
                truncated_for_budget = False
                if len(base_tokens) > keep_base:
                    base_tokens = base_tokens[-keep_base:]
                    truncated_for_budget = True

                if not base_tokens:
                    logprob = float("-inf")
                    greedy = False
                    verifier_stats = {
                        "metric": "verifier",
                        "score": logprob,
                        "raw_score": logprob,
                        "tokens": int(max(1, len(candidate_tokens))),
                        "candidate_source": candidate_meta.get("source"),
                        "candidate_preview": candidate_meta.get("candidate_preview"),
                        "candidate_tokens_len": int(len(candidate_tokens)),
                        "prompt_suffix_tokens_len": int(len(suffix_tokens)),
                        "truncated_for_budget": bool(truncated_for_budget),
                        "greedy": greedy,
                    }
                else:
                    has_native_decoder_blocks = all(
                        hasattr(self.model, attr) for attr in ("layers", "norm", "output")
                    )
                    if has_native_decoder_blocks:
                        bt = torch.tensor(base_tokens, device=self.device, dtype=torch.long)
                        with torch.autocast(device_type="cuda", dtype=self._dtype):
                            base_embeds = _token_embed(self.model, bt).to(dtype=self._dtype)
                        base_comp_mask = torch.zeros(int(bt.numel()), device=self.device, dtype=torch.bool)
                        verifier_out = self._score_verifier_yes_no_from_base(
                            base_embeds=base_embeds,
                            base_comp_mask=base_comp_mask,
                            decoder_budget=int(self.max_length),
                            rows_per_chunk=rows_per_chunk,
                        )
                    else:
                        verifier_out = self._score_verifier_yes_no_from_tokens(
                            base_tokens=base_tokens,
                            decoder_budget=int(self.max_length),
                        )
                    ll_yes = float(verifier_out.get("ll_yes", float("-inf")))
                    ll_no = float(verifier_out.get("ll_no", float("-inf")))
                    raw_score = float(verifier_out.get("score", float("-inf")))
                    normed_score = _apply_verifier_score_norm(
                        raw_score,
                        candidate_tokens=max(1, len(candidate_tokens)),
                        mode=self._verifier_apply_norm,
                    )
                    logprob = _apply_mcq_verifier_tie_break(
                        normed_score,
                        ll_yes=ll_yes,
                        ll_no=ll_no,
                        choice_idx=choice_idxs_slice[i],
                        mode=self._mcq_verifier_tie_break,
                    )
                    greedy = bool(verifier_out.get("greedy", False))
                    verifier_stats = {
                        "metric": "verifier",
                        "score": float(logprob),
                        "score_before_tie_break": float(normed_score),
                        "raw_score": float(raw_score),
                        "tokens": int(max(1, len(candidate_tokens))),
                        "ll_yes": ll_yes,
                        "ll_no": ll_no,
                        "best_yes_variant": verifier_out.get("best_yes_variant"),
                        "best_no_variant": verifier_out.get("best_no_variant"),
                        "candidate_source": candidate_meta.get("source"),
                        "candidate_preview": candidate_meta.get("candidate_preview"),
                        "candidate_tokens_len": int(len(candidate_tokens)),
                        "prompt_suffix_tokens_len": int(len(suffix_tokens)),
                        "choice_idx_from_label": candidate_meta.get("choice_idx_from_label"),
                        "truncated_for_budget": bool(truncated_for_budget),
                        "tie_break_mode": self._mcq_verifier_tie_break,
                        "greedy": greedy,
                    }
                res.append((float(logprob), greedy))
            else:
                if cont_len <= 0:
                    logprob = 0.0
                    greedy = True
                    res.append((logprob, greedy))
                else:
                    tgt_tensor = torch.tensor(tgt, device=self.device, dtype=torch.long)
                    seq_logprobs = logprobs[i, -tgt_full_lens[i] :, :]
                    tail_logprobs = seq_logprobs[-cont_len:, :]
                    token_logprobs = tail_logprobs.gather(-1, tgt_tensor[-cont_len:].unsqueeze(-1)).squeeze(-1)
                    logprob = float(token_logprobs.sum().item())
                    greedy = bool((tail_logprobs.argmax(dim=-1) == tgt_tensor[-cont_len:]).all().item())
                    res.append((logprob, greedy))

            if self._save_loglikelihood_debug and self._distributed_args.rank == 0:
                metric_tokens = int(cont_len)
                if verifier_stats is not None:
                    metric_tokens = int(verifier_stats.get("tokens", metric_tokens))
                loss = None
                ppl = None
                if metric_tokens > 0 and math.isfinite(float(logprob)):
                    loss = -float(logprob) / float(metric_tokens)
                    try:
                        ppl = math.exp(loss)
                    except OverflowError:
                        ppl = float("inf")

                row = {
                    "request_index": batch_start + i,
                    "mode": "decoder",
                    "configured_mode": str(self._mode),
                    "mcq_score_mode": self._mcq_score_mode,
                    "cont_len": int(cont_len),
                    "logprob": float(logprob),
                    "greedy": bool(greedy),
                    "ppl": ppl,
                }
                if self._mode in {"compress_answer", "reconstruct_first"} and (not has_comp):
                    row["compression_fallback"] = "no_compression_embeddings"
                if use_mcq_verifier:
                    row["verifier_score_mode"] = self._active_verifier_score_mode()
                    row["verifier_apply_norm"] = self._verifier_apply_norm
                    row["mcq_verifier_prompt_style"] = self._mcq_verifier_prompt_style
                    row["mcq_verifier_candidate_style"] = self._mcq_verifier_candidate_style
                    row["mcq_verifier_tie_break"] = self._mcq_verifier_tie_break
                # Best-effort: include identifiers for easier debugging.
                try:
                    if self._active_loglikelihood_task_names:
                        task_name = self._active_loglikelihood_task_names[batch_start + i]
                        if task_name:
                            row["task_name"] = task_name
                    if self._active_loglikelihood_doc_ids:
                        doc_id = self._active_loglikelihood_doc_ids[batch_start + i]
                        if doc_id is not None:
                            row["doc_id"] = int(doc_id)
                    if self._active_loglikelihood_choice_idxs:
                        choice_idx = self._active_loglikelihood_choice_idxs[batch_start + i]
                        if choice_idx is not None:
                            row["choice_idx"] = int(choice_idx)
                except Exception:
                    pass
                if loss is not None:
                    row["loss"] = loss

                try:
                    raw_cont = chunk[i][0][1] if chunk[i] and chunk[i][0] and len(chunk[i][0]) > 1 else ""
                except Exception:
                    raw_cont = ""
                if raw_cont:
                    row["cont_str_len"] = int(len(raw_cont))
                    row["cont_str_preview"] = raw_cont[:200]

                try:
                    raw_ctx = chunk[i][0][0] if chunk[i] and chunk[i][0] else ""
                except Exception:
                    raw_ctx = ""
                if raw_ctx:
                    row["ctx_str_len"] = int(len(raw_ctx))
                    row["ctx_str_preview"] = raw_ctx[:200]

                if cont_len > 0:
                    row["cont_tokens_len"] = int(cont_len)
                    row["cont_tokens_preview"] = list(
                        torch.tensor(tgt, device=self.device, dtype=torch.long)[-cont_len:]
                        .detach()
                        .to("cpu")
                        .tolist()[:20]
                    )

                if verifier_stats is not None:
                    row["verifier_score"] = float(verifier_stats.get("score", float(logprob)))
                    if verifier_stats.get("score_before_tie_break") is not None:
                        row["verifier_score_before_tie_break"] = float(
                            verifier_stats.get("score_before_tie_break")
                        )
                    row["verifier_raw_score"] = float(verifier_stats.get("raw_score", row["verifier_score"]))
                    row["score_tokens"] = int(verifier_stats.get("tokens", 0))
                    if verifier_stats.get("ll_yes") is not None:
                        row["ll_yes"] = float(verifier_stats.get("ll_yes"))
                    if verifier_stats.get("ll_no") is not None:
                        row["ll_no"] = float(verifier_stats.get("ll_no"))
                    if verifier_stats.get("best_yes_variant"):
                        row["best_yes_variant"] = verifier_stats.get("best_yes_variant")
                    if verifier_stats.get("best_no_variant"):
                        row["best_no_variant"] = verifier_stats.get("best_no_variant")
                    if verifier_stats.get("candidate_source"):
                        row["candidate_source"] = verifier_stats.get("candidate_source")
                    if verifier_stats.get("candidate_preview"):
                        row["candidate_preview"] = verifier_stats.get("candidate_preview")
                    if verifier_stats.get("candidate_tokens_len") is not None:
                        row["candidate_tokens_len"] = int(verifier_stats.get("candidate_tokens_len"))
                    if verifier_stats.get("prompt_suffix_tokens_len") is not None:
                        row["prompt_suffix_tokens_len"] = int(verifier_stats.get("prompt_suffix_tokens_len"))
                    if verifier_stats.get("choice_idx_from_label") is not None:
                        row["choice_idx_from_label"] = int(verifier_stats.get("choice_idx_from_label"))
                    if verifier_stats.get("truncated_for_budget") is not None:
                        row["truncated_for_budget"] = bool(verifier_stats.get("truncated_for_budget"))
                debug_rows.append(row)

        self._append_loglikelihood_debug_rows(debug_rows)
    return res

# ---------------------------------------------------------------------
# Harness integration: keep access to structured `Instance.doc` fields.
# ---------------------------------------------------------------------


def loglikelihood(self, requests, disable_tqdm: bool = False):  # type: ignore[override]
    """
    Override TemplateLM.loglikelihood so we can access `Instance.doc` for
    structured prompt parts (e.g., `doc["context"]`, `doc["question"]`,
    `doc["choices"]`) inside our custom loglikelihood implementations.

    Upstream TemplateLM.loglikelihood discards Instance.doc and only forwards
    (context_str, continuation_str) + tokenized ids into `_loglikelihood_tokens`.
    """
    self._active_loglikelihood_docs = [getattr(r, "doc", None) for r in requests]
    self._active_loglikelihood_task_names = [getattr(r, "task_name", None) for r in requests]
    self._active_loglikelihood_doc_ids = [getattr(r, "doc_id", None) for r in requests]
    self._active_loglikelihood_choice_idxs = [getattr(r, "idx", None) for r in requests]
    try:
        new_reqs: List[Tuple[Tuple[str, str], List[int], List[int]]] = []
        for req in requests:
            args = req.args
            if not isinstance(args, tuple) or len(args) < 2:
                raise ValueError(f"Unexpected loglikelihood Instance.args: {args!r}")
            context = args[0]
            continuation = args[1]

            if context == "":
                continuation_enc = self.tok_encode(continuation, add_special_tokens=False)
                if not continuation_enc:
                    context_enc, continuation_enc = ([self.prefix_token_id], [])
                else:
                    context_enc, continuation_enc = (
                        ([self.prefix_token_id], continuation_enc)
                        if self.prefix_token_id != continuation_enc[0]
                        else (continuation_enc[:1], continuation_enc[1:])
                    )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)
    finally:
        # Clear to avoid accidental reuse across evaluations.
        self._active_loglikelihood_docs = None
        self._active_loglikelihood_task_names = None
        self._active_loglikelihood_doc_ids = None
        self._active_loglikelihood_choice_idxs = None



def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
    results: List[float] = []
    for (text,) in [req.args for req in requests]:
        tokens = [self.eot_token_id] + self.tok_encode(text)
        total = 0.0
        for start in range(0, len(tokens) - 1, self.max_length):
            window = tokens[start : start + self.max_length + 1]
            if len(window) <= 1:
                continue
            inp = torch.tensor(window[:-1], device=self.device, dtype=torch.long).unsqueeze(0)
            tgt = torch.tensor(window[1:], device=self.device, dtype=torch.long)
            logits = self._model_call(inp)
            logprobs = self._model_logits(logits)[0, -len(tgt) :, :]
            token_lp = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
            total += token_lp
        results.append(total)
    return results



@torch.no_grad()
def _loglikelihood_tokens_compress_answer(
    self,
    requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
    disable_tqdm: bool = False,
    override_bs: Optional[int] = None,
) -> List[Tuple[float, bool]]:
    """
    Compress the context (question) into memory slots, then score only the continuation
    (answer) tokens conditioned on those slots, without reconstructing the question.
    """
    bs = override_bs or self.batch_size
    try:
        bs = int(bs)
    except Exception:
        bs = 1

    num_comp = int(getattr(self.model.args, "num_compression_tokens", 0))
    placeholder_id = 0
    max_mem_span_len = int(getattr(self.model.args, "max_mem_span_len", self.max_length))
    include_bor = False
    decoder_budget = int(self.decoder_budget)






    def _split_ctx_for_compression(ctx_tokens: List[int], cont_len: int) -> Tuple[List[int], List[int]]:
        if not ctx_tokens:
            return [], []
        force_min_span = bool(getattr(self, "_compress_answer_force_min_span", True))
        min_suffix_tokens = int(getattr(self, "_compress_answer_min_suffix_tokens", 128) or 0)
        if min_suffix_tokens < 0:
            min_suffix_tokens = 0
        max_len = int(self.max_length)
        span_cost = num_comp + 2  # decoder cost per span: BOM + slots + EOM
        saving = max_mem_span_len - span_cost
        if saving <= 0:
            # No compression benefit; let the span selector drop old spans.
            if force_min_span:
                return ctx_tokens, []
            return ctx_tokens, []

        raw_len = len(ctx_tokens)
        total_spans = math.ceil(raw_len / max_mem_span_len)
        # If even compressing all spans can't fit, keep no suffix and let the builder drop spans.
        if total_spans * span_cost + cont_len > max_len:
            return ctx_tokens, []

        # Need k spans compressed so that:
        #   raw_len + cont_len - k*(max_mem_span_len - (num_comp+2)) <= max_len
        need = raw_len + cont_len - max_len
        if need <= 0:
            k = raw_len // max_mem_span_len  # compress all full spans; keep remainder as suffix
        else:
            k = int(math.ceil(need / float(saving)))
        k = max(0, min(k, raw_len // max_mem_span_len))
        raw_comp_len = k * max_mem_span_len
        # In compress_answer mode, keep behavior deterministic: if we have any context,
        # force at least one compressed span so the mode is semantically different from decoder.
        # This primarily affects short MCQ prompts (e.g., MMLU) where raw_len < max_mem_span_len.
        # Keep a raw suffix to preserve near-answer semantics.
        if force_min_span and raw_len > 0 and raw_comp_len <= 0:
            # Preserve at least `min_suffix_tokens` raw tokens when possible.
            max_head_len = max(1, raw_len - min_suffix_tokens)
            raw_comp_len = min(max_mem_span_len, max_head_len)
        return ctx_tokens[:raw_comp_len], ctx_tokens[raw_comp_len:]



    res: List[Tuple[float, bool]] = []
    iterator = range(0, len(requests), bs)
    pbar = tqdm(iterator, disable=disable_tqdm or self.rank != 0, desc="native loglikelihood (compress)")
    for batch_start in pbar:
        chunk = requests[batch_start : batch_start + bs]
        use_mcq_verifier = self._use_mcq_verifier()
        docs_slice: List[Optional[dict]] = [None] * len(chunk)
        choice_idxs_slice: List[Optional[int]] = [None] * len(chunk)
        if use_mcq_verifier:
            try:
                if self._active_loglikelihood_docs:
                    for bi in range(len(chunk)):
                        gi = batch_start + bi
                        if gi < len(self._active_loglikelihood_docs):
                            v = self._active_loglikelihood_docs[gi]
                            docs_slice[bi] = v if isinstance(v, dict) else None
                if self._active_loglikelihood_choice_idxs:
                    for bi in range(len(chunk)):
                        gi = batch_start + bi
                        if gi < len(self._active_loglikelihood_choice_idxs):
                            cidx = self._active_loglikelihood_choice_idxs[gi]
                            if cidx is not None:
                                choice_idxs_slice[bi] = int(cidx)
            except Exception:
                # Keep verifier best-effort; fall back to continuation-only candidate building.
                pass
        # Build prefix prompt_embeds (memory blocks + optional BOQ) via the shared helper,
        # then score only the continuation tokens.
        prompt_tokens_override: List[List[int]] = [ctx for (_, ctx, _) in chunk]
        ctx_tokens_full_list: List[List[int]] = list(prompt_tokens_override)
        # In compress_answer, the decoder prefix is synthetic (memory blocks / chat scaffold),
        # so the continuation must be tokenized from the raw continuation string, not from
        # the (context+continuation) split performed in TemplateLM._encode_pair().
        cont_tokens_list = [
            self._tokenizer.encode(pair[1], bos=False, eos=False) if pair and len(pair) > 1 else []
            for (pair, _, _) in chunk
        ]
        prefix_tokens = self._get_likelihood_prefix_tokens("compress_answer")
        if prefix_tokens:
            cont_tokens_list = [prefix_tokens + list(cont) for cont in cont_tokens_list]
        suffix_tokens_list: List[List[int]] = [[] for _ in range(len(chunk))]
        split_source_by_idx: List[str] = ["prompt_tokens_override"] * len(chunk)

        # Always split non-chat prompts into:
        # - compressed prefix (older spans),
        # - raw suffix (latest short context near the answer).
        #
        # This avoids over-compressing short MCQ prompts (e.g., MMLU) into a single
        # memory span, which can degrade both LL and verifier-style yes/no scoring.
        if not getattr(self, "_chat_use_template", False):
            prompt_tokens_list, suffix_tokens_list_t = zip(
                *[_split_ctx_for_compression(p, len(c)) for p, c in zip(prompt_tokens_override, cont_tokens_list)]
            )
            prompt_tokens_override = list(prompt_tokens_list)
            suffix_tokens_list = list(suffix_tokens_list_t)
        else:
            suffix_tokens_list = [[] for _ in range(len(chunk))]


        # prompt_tokens, suffix_tokens = compute_needed_comp_slots(prompt_tokens_override[0])
        # Fast path: empty continuations
        chunk_results: List[Tuple[float, bool]] = [(float("-inf"), False)] * len(chunk)
        nonempty_idxs = [i for i, c in enumerate(cont_tokens_list) if len(c) > 0]
        for i in range(len(chunk)):
            if len(cont_tokens_list[i]) == 0:
                chunk_results[i] = (0.0, True)

        def _append_compress_debug_rows(
            prefix_embeds_list: Optional[List[Optional[torch.Tensor]]] = None,
            meta_n_spans: Optional[List[int]] = None,
            score_stats: Optional[Dict[int, Dict[str, Any]]] = None,
            skip_reasons: Optional[List[Optional[str]]] = None,
        ) -> None:
            if not self._save_loglikelihood_debug or self._distributed_args.rank != 0:
                return
            rows: List[dict] = []
            for i in range(len(chunk_results)):
                cont_len = len(cont_tokens_list[i])
                logprob, greedy = chunk_results[i]
                loss = None
                ppl = None
                score_info = score_stats.get(i) if score_stats else None
                if score_info and score_info.get("tokens", 0) > 0:
                    loss = score_info.get("loss")
                    ppl = score_info.get("ppl")
                elif cont_len > 0 and math.isfinite(logprob):
                    loss = -float(logprob) / float(cont_len)
                    try:
                        ppl = math.exp(loss)
                    except OverflowError:
                        ppl = float("inf")
                row = {
                    "request_index": batch_start + i,
                    "mode": "compress_answer",
                    "configured_mode": str(self._mode),
                    "mcq_score_mode": self._mcq_score_mode,
                    "cont_len": cont_len,
                    "logprob": logprob,
                    "greedy": greedy,
                    "ppl": ppl,
                }
                if use_mcq_verifier:
                    row["verifier_score_mode"] = self._active_verifier_score_mode()
                    row["verifier_apply_norm"] = self._verifier_apply_norm
                    row["mcq_verifier_prompt_style"] = self._mcq_verifier_prompt_style
                    row["mcq_verifier_candidate_style"] = self._mcq_verifier_candidate_style
                    row["mcq_verifier_tie_break"] = self._mcq_verifier_tie_break
                # Best-effort: include identifiers for easier debugging.
                try:
                    if self._active_loglikelihood_task_names:
                        task_name = self._active_loglikelihood_task_names[batch_start + i]
                        if task_name:
                            row["task_name"] = task_name
                    if self._active_loglikelihood_doc_ids:
                        doc_id = self._active_loglikelihood_doc_ids[batch_start + i]
                        if doc_id is not None:
                            row["doc_id"] = int(doc_id)
                    if self._active_loglikelihood_choice_idxs:
                        choice_idx = self._active_loglikelihood_choice_idxs[batch_start + i]
                        if choice_idx is not None:
                            row["choice_idx"] = int(choice_idx)
                except Exception:
                    pass
                try:
                    raw_cont = chunk[i][0][1] if chunk[i] and chunk[i][0] and len(chunk[i][0]) > 1 else ""
                except Exception:
                    raw_cont = ""
                if raw_cont:
                    row["cont_str_len"] = int(len(raw_cont))
                    row["cont_str_preview"] = raw_cont[:200]
                try:
                    raw_ctx = chunk[i][0][0] if chunk[i] and chunk[i][0] else ""
                except Exception:
                    raw_ctx = ""
                if raw_ctx:
                    row["ctx_str_len"] = int(len(raw_ctx))
                    row["ctx_str_preview"] = raw_ctx[:200]
                row["cont_tokens_len"] = int(cont_len)
                if cont_len > 0:
                    row["cont_tokens_preview"] = list(cont_tokens_list[i][:20])
                    if cont_len <= 50:
                        row["cont_tokens"] = list(cont_tokens_list[i])
                    try:
                        decoded = self.tok_decode_w_special_tokens(cont_tokens_list[i])
                        if decoded:
                            row["cont_decoded_preview"] = decoded[:200]
                    except Exception:
                        pass
                if loss is not None:
                    row["loss"] = loss
                if score_info is not None:
                    if score_info.get("metric") == "verifier":
                        row["verifier_score"] = float(score_info.get("score", logprob))
                        if score_info.get("score_before_tie_break") is not None:
                            row["verifier_score_before_tie_break"] = float(
                                score_info.get("score_before_tie_break")
                            )
                        row["verifier_raw_score"] = float(score_info.get("raw_score", row["verifier_score"]))
                        row["verifier_tokens"] = int(score_info.get("tokens", 0))
                        if score_info.get("ll_yes") is not None:
                            row["ll_yes"] = float(score_info.get("ll_yes"))
                        if score_info.get("ll_no") is not None:
                            row["ll_no"] = float(score_info.get("ll_no"))
                        if score_info.get("best_yes_variant"):
                            row["best_yes_variant"] = score_info.get("best_yes_variant")
                        if score_info.get("best_no_variant"):
                            row["best_no_variant"] = score_info.get("best_no_variant")
                        if score_info.get("candidate_source"):
                            row["candidate_source"] = score_info.get("candidate_source")
                        if score_info.get("candidate_preview"):
                            row["candidate_preview"] = score_info.get("candidate_preview")
                        if score_info.get("candidate_tokens_len") is not None:
                            row["candidate_tokens_len"] = int(score_info.get("candidate_tokens_len"))
                        if score_info.get("prompt_suffix_tokens_len") is not None:
                            row["prompt_suffix_tokens_len"] = int(score_info.get("prompt_suffix_tokens_len"))
                        if score_info.get("truncated_for_budget") is not None:
                            row["truncated_for_budget"] = bool(score_info.get("truncated_for_budget"))
                    row["score_tokens"] = int(score_info.get("tokens", 0))
                    if "windows" in score_info:
                        row["windows"] = int(score_info.get("windows") or 0)
                    if "rolled" in score_info:
                        row["rolled"] = bool(score_info.get("rolled"))
                if skip_reasons is not None and skip_reasons[i]:
                    row["skip_reason"] = skip_reasons[i]
                if meta_n_spans is not None:
                    n_spans = int(meta_n_spans[i]) if i < len(meta_n_spans) else 0
                    row["n_spans"] = n_spans
                    row["total_comp_slots"] = n_spans * int(num_comp)
                if suffix_tokens_list is not None:
                    row["suffix_len"] = len(suffix_tokens_list[i])
                if split_source_by_idx is not None and i < len(split_source_by_idx):
                    row["split_source"] = split_source_by_idx[i]
                if prefix_embeds_list is not None:
                    pe = prefix_embeds_list[i]
                    if pe is not None:
                        prefix_len = int(pe.shape[0]) + (len(suffix_tokens_list[i]) if suffix_tokens_list is not None else 0)
                        row["prefix_len"] = prefix_len
                rows.append(row)
            self._append_loglikelihood_debug_rows(rows)

        if not nonempty_idxs:
            _append_compress_debug_rows()
            res.extend(chunk_results)
            continue

        dummy_prompts = [""] * len(chunk)
        gen_lens = [len(c) + (len(suffix_tokens_list[i]) if suffix_tokens_list is not None else 0) for i, c in enumerate(cont_tokens_list)]
        if not getattr(self, "_chat_use_template", False):
            # Prefer structured (context, query) splitting when available so the query
            # (e.g. "Question: ...\nAnswer:") is not compressed away. This is critical
            # for long-context multiple-choice tasks like InfiniteBench longbook_choice_eng.
            split_context_list: Optional[List[str]] = None
            split_query_list: Optional[List[str]] = None
            try:
                doc_and_context = self._get_doc_and_context(
                    ctx_tokens_list=ctx_tokens_full_list, batch_start=batch_start
                )
                split_context_list = doc_and_context.get("context_list")
                split_query_list = doc_and_context.get("query_list")
            except Exception as e:
                # Never silently fall back to compressing the full rendered prompt.
                # For non-chat tasks, keep full prompt as query and leave context empty.
                split_context_list = [""] * len(chunk)
                split_query_list = [
                    self._tokenizer.decode_w_special_tokens(ctx) if ctx else ""
                    for ctx in ctx_tokens_full_list
                ]
                split_source_by_idx = [f"doc_split_fallback:{type(e).__name__}"] * len(chunk)
                if self._distributed_args.rank == 0 and not self._warned_doc_split_fallback:
                    print(
                        "[native][warn] _get_doc_and_context failed in compress_answer; "
                        "falling back to context='' + query=full prompt tokens.",
                        file=sys.stderr,
                    )
                    self._warned_doc_split_fallback = True

            if (
                split_context_list is not None
                and split_query_list is not None
                and len(split_context_list) == len(chunk)
                and len(split_query_list) == len(chunk)
            ):
                # For MCQ tasks like MMLU, structured doc split can place almost all
                # prompt content into `query` with empty `context`. In compress_answer
                # mode this leads to n_spans=0 (no compression applied).
                # When enabled, fall back to prompt-token compression path so we always
                # compress at least one span for non-empty prompts.
                force_min_span = bool(getattr(self, "_compress_answer_force_min_span", True))
                if force_min_span:
                    has_any_ctx = any(str(c or "").strip() for c in split_context_list)
                    if not has_any_ctx:
                        split_context_list = None
                        split_query_list = None
                        split_source_by_idx = ["doc_split_empty_context_fallback"] * len(chunk)

            if (
                split_context_list is not None
                and split_query_list is not None
                and len(split_context_list) == len(chunk)
                and len(split_query_list) == len(chunk)
            ):
                # When structured (context, query) split is available, the decoder
                # prefix is fully rebuilt from split fields. Keeping the previously
                # computed raw suffix (from full prompt token splitting) would append
                # duplicated query text and shift LL scores.
                suffix_tokens_list = [[] for _ in range(len(chunk))]
                # If split returned empty queries for all rows, preserve full prompt as query.
                if not any(str(q or "").strip() for q in split_query_list):
                    split_context_list = [""] * len(chunk)
                    split_query_list = [
                        self._tokenizer.decode_w_special_tokens(ctx) if ctx else ""
                        for ctx in ctx_tokens_full_list
                    ]
                    split_source_by_idx = ["doc_split_empty_query_fallback"] * len(chunk)

                key_to_group: Dict[Tuple[str, str], int] = {}
                group_contexts: List[str] = []
                group_queries: List[str] = []
                group_gen_lens: List[int] = []
                group_indices: List[List[int]] = []
                for i in range(len(chunk)):
                    key = (split_context_list[i], split_query_list[i])
                    gidx = key_to_group.get(key)
                    if gidx is None:
                        gidx = len(group_contexts)
                        key_to_group[key] = gidx
                        group_contexts.append(split_context_list[i])
                        group_queries.append(split_query_list[i])
                        group_gen_lens.append(int(gen_lens[i]))
                        group_indices.append([i])
                    else:
                        if int(gen_lens[i]) > group_gen_lens[gidx]:
                            group_gen_lens[gidx] = int(gen_lens[i])
                        group_indices[gidx].append(i)

                group_dummy_prompts = [""] * len(group_contexts)
                group_prefix_embeds, meta = self._build_compress_prompt_embeds_batch(
                    group_dummy_prompts,
                    group_gen_lens,
                    include_bor=False,
                    decoder_include_prompt_tokens=False,
                    decoder_memory_layout="per_span",
                    return_meta=True,
                    # Keep LL scoring aligned with decoder mode for no-memory rows (n_spans==0).
                    # Adding BOQ in that case shifts the distribution and can depress MCQ scores.
                    not_add_boq_index=True,
                    context_list=group_contexts,
                    query_list=group_queries,
                )
                meta_group_n_spans = (
                    meta.get("n_spans", [1] * len(group_contexts)) if meta else [1] * len(group_contexts)
                )
                meta_group_comp_masks = meta.get("comp_mask_list") if meta else None
                prefix_embeds_list = [None] * len(chunk)
                meta_n_spans = [1] * len(chunk)
                prefix_comp_masks_list = [None] * len(chunk)
                for gidx, idxs in enumerate(group_indices):
                    for i in idxs:
                        prefix_embeds_list[i] = group_prefix_embeds[gidx]
                        meta_n_spans[i] = int(meta_group_n_spans[gidx]) if gidx < len(meta_group_n_spans) else 1
                        if meta_group_comp_masks is not None and gidx < len(meta_group_comp_masks):
                            prefix_comp_masks_list[i] = list(meta_group_comp_masks[gidx])
                        if split_source_by_idx[i] == "prompt_tokens_override":
                            split_source_by_idx[i] = "doc_split"
            else:
                key_to_group: Dict[Tuple[int, ...], int] = {}
                group_prompt_tokens: List[List[int]] = []
                group_gen_lens: List[int] = []
                group_indices: List[List[int]] = []
                for i in range(len(chunk)):
                    key = tuple(prompt_tokens_override[i])
                    gidx = key_to_group.get(key)
                    if gidx is None:
                        gidx = len(group_prompt_tokens)
                        key_to_group[key] = gidx
                        group_prompt_tokens.append(prompt_tokens_override[i])
                        group_gen_lens.append(int(gen_lens[i]))
                        group_indices.append([i])
                    else:
                        if int(gen_lens[i]) > group_gen_lens[gidx]:
                            group_gen_lens[gidx] = int(gen_lens[i])
                        group_indices[gidx].append(i)

                group_dummy_prompts = [""] * len(group_prompt_tokens)
                group_prefix_embeds, meta = self._build_compress_prompt_embeds_batch(
                    group_dummy_prompts,
                    group_gen_lens,
                    include_bor=False,
                    decoder_include_prompt_tokens=False,
                    decoder_memory_layout="per_span",
                    prompt_tokens_override=group_prompt_tokens,
                    return_meta=True,
                    # for do not add boq index for decoder prefix
                    not_add_boq_index=True,
                )
                meta_group_n_spans = (
                    meta.get("n_spans", [1] * len(group_prompt_tokens))
                    if meta
                    else [1] * len(group_prompt_tokens)
                )
                meta_group_comp_masks = meta.get("comp_mask_list") if meta else None
                prefix_embeds_list = [None] * len(chunk)
                meta_n_spans = [1] * len(chunk)
                prefix_comp_masks_list = [None] * len(chunk)
                for gidx, idxs in enumerate(group_indices):
                    for i in idxs:
                        prefix_embeds_list[i] = group_prefix_embeds[gidx]
                        meta_n_spans[i] = (
                            int(meta_group_n_spans[gidx]) if gidx < len(meta_group_n_spans) else 1
                        )
                        if meta_group_comp_masks is not None and gidx < len(meta_group_comp_masks):
                            prefix_comp_masks_list[i] = list(meta_group_comp_masks[gidx])
                        split_source_by_idx[i] = "prompt_tokens_override"
        else:
            doc_and_context = self._get_doc_and_context(ctx_tokens_list=ctx_tokens_full_list, batch_start=batch_start)
            context_list, question_list, query_list = (
                doc_and_context["context_list"],
                doc_and_context["question_list"],
                doc_and_context["query_list"],
            )

            key_to_group: Dict[Tuple[str, str], int] = {}
            group_contexts: List[str] = []
            group_queries: List[str] = []
            group_gen_lens: List[int] = []
            group_indices: List[List[int]] = []
            for i in range(len(chunk)):
                key = (context_list[i], query_list[i])
                gidx = key_to_group.get(key)
                if gidx is None:
                    gidx = len(group_contexts)
                    key_to_group[key] = gidx
                    group_contexts.append(context_list[i])
                    group_queries.append(query_list[i])
                    group_gen_lens.append(int(gen_lens[i]))
                    group_indices.append([i])
                else:
                    if int(gen_lens[i]) > group_gen_lens[gidx]:
                        group_gen_lens[gidx] = int(gen_lens[i])
                    group_indices[gidx].append(i)

            group_dummy_prompts = [""] * len(group_contexts)
            group_prefix_embeds, meta = self._build_compress_prompt_embeds_batch(
                group_dummy_prompts,
                group_gen_lens,
                include_bor=False,
                decoder_include_prompt_tokens=False,
                decoder_memory_layout="per_span",
                return_meta=True,
                not_add_boq_index=False,
                context_list=group_contexts,
                query_list=group_queries,
            )
            meta_group_n_spans = meta.get("n_spans", [1] * len(group_contexts)) if meta else [1] * len(group_contexts)
            meta_group_comp_masks = meta.get("comp_mask_list") if meta else None
            prefix_embeds_list = [None] * len(chunk)
            meta_n_spans = [1] * len(chunk)
            prefix_comp_masks_list = [None] * len(chunk)
            for gidx, idxs in enumerate(group_indices):
                for i in idxs:
                    prefix_embeds_list[i] = group_prefix_embeds[gidx]
                    meta_n_spans[i] = int(meta_group_n_spans[gidx]) if gidx < len(meta_group_n_spans) else 1
                    if meta_group_comp_masks is not None and gidx < len(meta_group_comp_masks):
                        prefix_comp_masks_list[i] = list(meta_group_comp_masks[gidx])
                    split_source_by_idx[i] = "chat_split"
        # breakpoint()

        # Suffix tokens (uncompressed tail of the prompt) can be ragged across the batch.
        # Embed per-sample to avoid creating a rectangular tensor.
        d_model = int(getattr(self.model.args, "d_model", 0))
        suffix_embeds_list: List[torch.Tensor] = []
        for suffix in suffix_tokens_list:
            if suffix:
                st = torch.tensor(list(suffix), device=self.device, dtype=torch.long)
                with torch.autocast(device_type="cuda", dtype=self._dtype):
                    se = _token_embed(self.model, st).to(dtype=self._dtype)
                suffix_embeds_list.append(se)
            else:
                suffix_embeds_list.append(torch.empty((0, d_model), device=self.device, dtype=self._dtype))

        # meta_n_spans is now pre-mapped per sample for grouped compression
        seq_embeds: List[torch.Tensor] = []
        seq_targets: List[torch.Tensor] = []
        seq_loss_masks: List[torch.Tensor] = []
        seq_comp_masks: List[torch.Tensor] = []
        prefix_lens: List[int] = []
        cont_lens: List[int] = []
        cont_targets: List[torch.Tensor] = []
        valid_map: List[int] = []
        skip_reasons: List[Optional[str]] = [None] * len(chunk)
        score_stats_by_idx: Dict[int, Dict[str, Any]] = {}
        rows_per_chunk = int(getattr(getattr(self.model, "args", None), "cross_entropy_chunk", 8)) * 16
        rows_per_chunk = max(8, min(int(rows_per_chunk), 512))

        for i in nonempty_idxs:
            cont = cont_tokens_list[i]
            cont_len = len(cont)
            if cont_len <= 0:
                continue
            pe0 = prefix_embeds_list[i]
            if pe0 is None:
                skip_reasons[i] = "no_prefix"
                continue

            # Prefix consists of memory blocks + raw suffix tokens (kept uncompressed).
            suffix_e = suffix_embeds_list[i]
            pe = torch.cat([pe0, suffix_e], dim=0) if suffix_e.numel() else pe0
            prefix_len = int(pe.shape[0])

            # compression_token_mask: True for placeholder slots in memory blocks, False elsewhere.
            if prefix_comp_masks_list is not None and prefix_comp_masks_list[i] is not None:
                comp_mask_prefix = prefix_comp_masks_list[i]
            else:
                n_spans = int(meta_n_spans[i]) if i < len(meta_n_spans) else 1
                comp_mask_prefix = ([False] + ([True] * num_comp) + [False]) * n_spans
            suffix_len = len(suffix_tokens_list[i]) if suffix_tokens_list[i] else 0
            base_mask = comp_mask_prefix + ([False] * suffix_len)
            base_mask_t = torch.tensor(base_mask, device=self.device, dtype=torch.bool)
            if int(base_mask_t.numel()) != prefix_len:
                skip_reasons[i] = "comp_mask_mismatch"
                continue

            if use_mcq_verifier:
                try:
                    pair = chunk[i][0]
                    cont_str = pair[1] if pair and len(pair) > 1 else ""
                except Exception:
                    cont_str = ""
                candidate_tokens, candidate_meta = self._build_verifier_candidate_tokens(
                    continuation_tokens=cont,
                    continuation_str=str(cont_str),
                    doc=docs_slice[i],
                    choice_idx=choice_idxs_slice[i],
                    context_tokens=list(ctx_tokens_full_list[i]) if i < len(ctx_tokens_full_list) else list(chunk[i][1]),
                    context_text=(chunk[i][0][0] if chunk[i] and chunk[i][0] and len(chunk[i][0]) > 0 else ""),
                )
                suffix_tokens = self._get_verifier_prompt_suffix_tokens()
                verify_prefix_tokens = list(candidate_tokens) + list(suffix_tokens)
                if not verify_prefix_tokens:
                    verify_prefix_tokens = list(cont)
                vpt = torch.tensor(verify_prefix_tokens, device=self.device, dtype=torch.long)
                with torch.autocast(device_type="cuda", dtype=self._dtype):
                    verify_prefix_embeds = _token_embed(self.model, vpt).to(dtype=self._dtype)
                base_embeds = torch.cat([pe, verify_prefix_embeds], dim=0)
                base_comp_mask = torch.cat(
                    [
                        base_mask_t,
                        torch.zeros(
                            int(vpt.numel()),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    ],
                    dim=0,
                )

                max_verify_len = 1
                for _, toks in (self._verifier_yes_variants + self._verifier_no_variants):
                    if len(toks) > max_verify_len:
                        max_verify_len = len(toks)

                truncated_for_budget = False
                if int(base_embeds.shape[0]) + max_verify_len > decoder_budget:
                    keep_base = max(1, int(decoder_budget) - int(max_verify_len))
                    drop = int(base_embeds.shape[0]) - keep_base
                    if drop > 0:
                        base_embeds = base_embeds[drop:]
                        base_comp_mask = base_comp_mask[drop:]
                        truncated_for_budget = True

                verifier_out = self._score_verifier_yes_no_from_base(
                    base_embeds=base_embeds,
                    base_comp_mask=base_comp_mask,
                    decoder_budget=decoder_budget,
                    rows_per_chunk=rows_per_chunk,
                )
                ll_yes = float(verifier_out.get("ll_yes", float("-inf")))
                ll_no = float(verifier_out.get("ll_no", float("-inf")))
                raw_score = float(verifier_out.get("score", float("-inf")))
                normed_score = _apply_verifier_score_norm(
                    raw_score,
                    candidate_tokens=len(candidate_tokens),
                    mode=self._verifier_apply_norm,
                )
                score = _apply_mcq_verifier_tie_break(
                    normed_score,
                    ll_yes=ll_yes,
                    ll_no=ll_no,
                    choice_idx=choice_idxs_slice[i],
                    mode=self._mcq_verifier_tie_break,
                )
                greedy = bool(verifier_out.get("greedy", False))
                chunk_results[i] = (float(score), greedy)
                score_stats_by_idx[i] = {
                    "metric": "verifier",
                    "score": float(score),
                    "score_before_tie_break": float(normed_score),
                    "raw_score": float(raw_score),
                    "tokens": int(max(1, len(candidate_tokens))),
                    "ll_yes": ll_yes,
                    "ll_no": ll_no,
                    "best_yes_variant": verifier_out.get("best_yes_variant"),
                    "best_no_variant": verifier_out.get("best_no_variant"),
                    "candidate_source": candidate_meta.get("source"),
                    "candidate_preview": candidate_meta.get("candidate_preview"),
                    "candidate_tokens_len": int(len(candidate_tokens)),
                    "prompt_suffix_tokens_len": int(len(suffix_tokens)),
                    "choice_idx_from_label": candidate_meta.get("choice_idx_from_label"),
                    "truncated_for_budget": bool(truncated_for_budget),
                    "tie_break_mode": self._mcq_verifier_tie_break,
                    "greedy": greedy,
                }
                continue

            if prefix_len + cont_len > decoder_budget:
                skip_reasons[i] = "length_overflow"
                continue

            cont_t = torch.tensor(cont, device=self.device, dtype=torch.long)
            with torch.autocast(device_type="cuda", dtype=self._dtype):
                cont_e = _token_embed(self.model, cont_t).to(dtype=self._dtype)

            full_embeds = torch.cat([pe, cont_e], dim=0)
            total_len = int(full_embeds.shape[0])

            # Decoder token ids (for shifting/targets). Memory placeholders use a valid vocab id.
            prefix_mask = prefix_comp_masks_list[i] if prefix_comp_masks_list is not None else None
            if prefix_mask is not None:
                prefix_ids = [placeholder_id] * len(prefix_mask)
            else:
                n_spans = int(meta_n_spans[i]) if i < len(meta_n_spans) else 1
                prefix_ids = ([BEGIN_OF_MEMORY_INDEX] + ([placeholder_id] * num_comp) + [END_OF_MEMORY_INDEX]) * n_spans
            suffix_ids = list(suffix_tokens_list[i]) if suffix_tokens_list[i] else []
            token_ids = prefix_ids + suffix_ids + list(cont)
            if len(token_ids) != total_len:
                skip_reasons[i] = "token_len_mismatch"
                continue

            targets_ids = token_ids[1:] + [int(self.eos_token_id)]
            targets_t = torch.tensor(targets_ids, device=self.device, dtype=torch.long)

            # Score only continuation: first continuation token is predicted at position prefix_len-1.
            score_start = prefix_len - 1
            score_end = score_start + cont_len
            if score_start < 0 or score_end > total_len:
                skip_reasons[i] = "score_range_invalid"
                continue
            loss_mask = torch.zeros(total_len, device=self.device, dtype=torch.bool)
            loss_mask[score_start:score_end] = True

            # compression_token_mask: True for placeholder slots in memory blocks, False elsewhere.
            if prefix_mask is not None:
                comp_mask = list(prefix_mask) + ([False] * (total_len - len(prefix_mask)))
            else:
                comp_mask_prefix = ([False] + ([True] * num_comp) + [False]) * n_spans
                comp_mask = comp_mask_prefix + ([False] * (total_len - len(comp_mask_prefix)))
            comp_mask_t = torch.tensor(comp_mask, device=self.device, dtype=torch.bool)

            seq_embeds.append(full_embeds)
            seq_targets.append(targets_t)
            seq_loss_masks.append(loss_mask)
            seq_comp_masks.append(comp_mask_t)
            prefix_lens.append(prefix_len)
            cont_lens.append(cont_len)
            cont_targets.append(cont_t)
            valid_map.append(i)

        if not seq_embeds:
            _append_compress_debug_rows(prefix_embeds_list, meta_n_spans, score_stats_by_idx, skip_reasons)
            res.extend(chunk_results)
            continue

        score_out = self._forward_score_continuations(
            seq_embeds=seq_embeds,
            cont_targets=cont_targets,
            prefix_lens=prefix_lens,
            comp_mask_list=seq_comp_masks,
            rows_per_chunk=rows_per_chunk,
        )
        per_sample = score_out.get("per_sample") or []
        for j, orig_idx in enumerate(valid_map):
            if j >= len(per_sample):
                break
            ps = per_sample[j] or {}
            ll = float(ps.get("ll", float("-inf")))
            greedy = bool(ps.get("greedy", False))
            chunk_results[orig_idx] = (ll, greedy)
            score_stats_by_idx[orig_idx] = ps

        _append_compress_debug_rows(prefix_embeds_list, meta_n_spans, score_stats_by_idx, skip_reasons)
        res.extend(chunk_results)
    # Safety: if we somehow produced fewer responses than requests, pad with -inf.
    if len(res) < len(requests):
        missing = len(requests) - len(res)
        res.extend([(float("-inf"), False)] * missing)
    return res


    # Decide how much of the context to compress (prefix) vs keep raw (suffix),
    # without ever truncating continuation tokens.


@torch.no_grad()
def _loglikelihood_tokens_reconstruct_first(
    self,
    requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
    disable_tqdm: bool = False,
) -> List[Tuple[float, bool]]:
    """
    Reconstruct-first loglikelihood for compression models.

    Differences vs `compress_answer`:
    - We **generate reconstruction once per unique context** (not once per option).
    - Then we score each continuation option conditioned on (memory + reconstruction [+ optional query suffix]).

    If `add_query_before_likelihood=True`:
    - We try to split the original prompt into a long document block and a short query/choices suffix
      using `<text> ... </text>` markers (LongBench-style).
    - Only the document block is compressed/reconstructed; the query suffix is appended before scoring.
    """
    self._ensure_vllm_manager(caller="reconstruct_first loglikelihood")
    if self._vllm_manager is None:
        raise RuntimeError("reconstruct_first loglikelihood requires vLLM but initialization failed.")

    num_comp = int(getattr(self.model.args, "num_compression_tokens", 0))
    max_mem_span_len = int(getattr(self.model.args, "max_mem_span_len", self.max_length))
    add_bor = bool(self._reconstruct_add_bor)
    max_bor = int(self._reconstruct_max_bor)
    add_query = bool(getattr(self, "_add_query_before_likelihood", False))
    chat_enabled = bool(getattr(self, "_chat_use_template", False))
    decoder_budget = int(self.decoder_budget)
    stop_id_set = set(self.stop_ids)
    newline_ids = self._tokenizer.encode("\n", bos=False, eos=False)
    newline_id = newline_ids[0] if newline_ids else None

    def _postprocess_recon(tokens: List[int], max_len: int) -> Tuple[List[int], dict]:
        out: List[int] = []
        bor_count = 0
        eor_count = 0
        stop_reason: Optional[str] = None
        eos_id = int(self.eos_token_id)
        eot_id = int(self.eot_token_id)
        for t in tokens:
            t = int(t)
            # Treat EOS/EOT as a hard stop (and do not include it).
            if t in (eos_id, eot_id):
                stop_reason = "eos"
                break
            if t == BEGIN_OF_RECONSTRUCTION_INDEX:
                bor_count += 1
                if max_bor > 0 and bor_count > max_bor:
                    stop_reason = "bor_limit"
                    break
            if t == END_OF_RECONSTRUCTION_INDEX:
                eor_count += 1
                out.append(t)
                # If `max_bor` is disabled (<=0), stop at the first EOR.
                if max_bor <= 0:
                    stop_reason = "eor"
                    break
                if eor_count >= max_bor:
                    stop_reason = "max_eor"
                    break
                continue
            out.append(t)
            if len(out) >= max_len:
                stop_reason = "max_recon"
                break
        if stop_reason is None:
            stop_reason = "max_recon"
        # Strip trailing EOS/EOT just in case.
        while out and out[-1] in (eos_id, eot_id):
            out.pop()
        # Strip trailing stop_ids.
        while out and (out[-1] in stop_id_set or (newline_id is not None and out[-1] == newline_id)):
            out.pop()


        return out, {"stop_reason": stop_reason, "bor_count": bor_count, "eor_count": eor_count}

    def _build_pre_output_tokens(
        gi: int,
        group_query_tokens: List[List[int]],
        group_suffix_lens: List[int],
        group_n_spans: List[int],
    ) -> Tuple[List[int], List[int]]:
        if chat_enabled:
            memory_start = self._tokenizer.encode("<|im_start|>memory\n", bos=False, eos=False)
            user_start = self._tokenizer.encode("<|im_start|>user\n", bos=False, eos=False)
            assistant_start = self._tokenizer.encode("<|im_start|>assistant\n", bos=False, eos=False)
            im_end = self._tokenizer.encode("<|im_end|>\n", bos=False, eos=False)
            span_cost = num_comp + 2
            query_tokens = group_query_tokens[gi]
            fixed_len = (
                len(memory_start)
                + len(im_end)
                + len(user_start)
                + (1 if getattr(self, "_add_boq_index", False) else 0)
                + len(query_tokens)
                + len(im_end)
                + len(assistant_start)
            )
            avail_for_memory = int(decoder_budget) - int(fixed_len) - int(group_suffix_lens[gi])
            max_spans = avail_for_memory // max(1, span_cost)
            if max_spans < 0:
                max_spans = 0
            ret = self._format_chat(
                user_text=group_query_list[gi],
                contexts=group_doc_list[gi],
                max_spans=max_spans,
            )
            prefix_tokens = list(ret.get("decoder_prefix_tokens") or [])
        else:
            prefix_tokens = []
            for _ in range(int(group_n_spans[gi])):
                prefix_tokens.extend([BEGIN_OF_MEMORY_INDEX] + ([0] * num_comp) + [END_OF_MEMORY_INDEX])
            if self._add_boq_index:
                prefix_tokens.append(BEGIN_OF_QUERY_INDEX)
            if add_bor:
                prefix_tokens.append(BEGIN_OF_RECONSTRUCTION_INDEX)

        pre_output_tokens = list(prefix_tokens) + list(group_recon_tokens[gi])
        if add_query:
            pre_output_tokens.extend(group_query_tokens[gi])
        return pre_output_tokens, list(prefix_tokens)

    res: List[Tuple[float, bool]] = []
    bs = max(1, int(self._vllm_reconstruct_batch_size))
    iterator = range(0, len(requests), bs)
    pbar = tqdm(iterator, disable=disable_tqdm or self.rank != 0, desc="native loglikelihood (reconstruct_first)")

    for batch_start in pbar:
        chunk = requests[batch_start : batch_start + bs]
        # ctx_str_list: List[str] = [ctx_str for (ctx_str, _, _) in chunk]
        ctx_tokens_list: List[List[int]] = [ctx for (_, ctx, _) in chunk]
        # Re-encode continuations from raw strings: reconstruct_first uses a synthetic prefix
        # (memory + reconstruction [+ optional query]) so TemplateLM._encode_pair() boundaries
        # are not applicable.
        cont_tokens_list = [
            self._tokenizer.encode(pair[1], bos=False, eos=False) if pair and len(pair) > 1 else []
            for (pair, _, _) in chunk
        ]
        prefix_tokens = self._get_likelihood_prefix_tokens("reconstruct_first")
        if prefix_tokens:
            cont_tokens_list = [prefix_tokens + list(cont) for cont in cont_tokens_list]

        # Per-request results in this chunk (keep order).
        chunk_results: List[Tuple[float, bool]] = [(float("-inf"), False)] * len(chunk)
        for i, cont in enumerate(cont_tokens_list):
            if not cont:
                chunk_results[i] = (0.0, True)

        # Group by identical context tokens (e.g., multiple-choice options share the same prompt).
        groups: Dict[Tuple[int, ...], List[int]] = {}
        for i, ctx in enumerate(ctx_tokens_list):
            groups.setdefault(tuple(ctx), []).append(i)

        group_keys = list(groups.keys())
        # breakpoint()
        n_groups = len(group_keys)
        if n_groups == 0:
            res.extend(chunk_results)
            continue

        group_doc_list: List[str] = [""] * n_groups
        group_query_list: List[str] = [""] * n_groups
        group_query_tokens: List[List[int]] = [[] for _ in range(n_groups)]
        group_prompt_tokens: List[List[int]] = [[] for _ in range(n_groups)]
        group_max_cont_lens: List[int] = [0] * n_groups
        group_suffix_lens: List[int] = [0] * n_groups
        group_indices: List[List[int]] = [[] for _ in range(n_groups)]

        doc_and_context = self._get_doc_and_context(ctx_tokens_list=ctx_tokens_list, batch_start=batch_start)
        context_list = doc_and_context["context_list"]
        query_list = doc_and_context["query_list"]

        for group_idx, gk in enumerate(group_keys):
            idxs = groups[gk]
            rep = idxs[0]

            max_cont = max((len(cont_tokens_list[i]) for i in idxs), default=0)
            group_max_cont_lens[group_idx] = max_cont
            group_indices[group_idx] = idxs

            doc_text = context_list[rep]
            query_text = query_list[rep]
            group_doc_list[group_idx] = doc_text
            group_query_list[group_idx] = query_text

            # Always keep query tokens for debug; only append them for scoring in non-chat mode.
            qtoks = self._tokenizer.encode(query_text, bos=False, eos=False)
            group_query_tokens[group_idx] = qtoks

            if chat_enabled:
                group_suffix_lens[group_idx] = int(max_cont)
                if add_query:
                    group_suffix_lens[group_idx] += int(len(qtoks))
                group_prompt_tokens[group_idx] = self._tokenizer.encode(doc_text, bos=False, eos=False)
            else:
                group_suffix_lens[group_idx] = int(len(qtoks) + max_cont)
                if add_query:
                    group_suffix_lens[group_idx] += int(len(qtoks))
                # Compress the full prompt tokens (context + query).
                group_prompt_tokens[group_idx] = list(ctx_tokens_list[rep])

        # Build compression prefix prompt_embeds per unique context (once per group).
        dummy_prompts = [""] * n_groups
        prefix_embeds_list, meta = self._build_compress_prompt_embeds_batch(
            dummy_prompts,
            group_suffix_lens,
            include_bor=add_bor,
            decoder_include_prompt_tokens=False,
            decoder_memory_layout="per_span",
            prompt_tokens_override=group_prompt_tokens,
            return_meta=True,
            not_add_boq_index=False,
            context_list=group_doc_list if chat_enabled else None,
            query_list=group_query_list if chat_enabled else None,
        )

        meta_n_spans = (meta or {}).get("n_spans", [0] * n_groups)
        meta_flat_ctx = (meta or {}).get("flat_ctx_len", [0] * n_groups)

        group_prefix_lens: List[int] = [0] * n_groups
        group_max_recon_lens: List[int] = [0] * n_groups
        group_n_spans: List[int] = [0] * n_groups
        group_n_slots: List[int] = [0] * n_groups

        for gi, pe in enumerate(prefix_embeds_list):
            if pe is None:
                continue
            pl = int(pe.shape[0])
            group_prefix_lens[gi] = pl
            n_spans = int(meta_n_spans[gi]) if gi < len(meta_n_spans) else 1
            if n_spans <= 0:
                n_spans = 1
            group_n_spans[gi] = n_spans
            group_n_slots[gi] = num_comp * int(n_spans)
            # Reserve room for query suffix + the longest continuation option for this context.
            budget_recon = max(0, decoder_budget - pl - int(group_suffix_lens[gi]))
            flat_len = int(meta_flat_ctx[gi]) if gi < len(meta_flat_ctx) else budget_recon
            group_max_recon_lens[gi] = min(flat_len, budget_recon)

        # Reconstruct ONCE per unique context (group).
        group_recon_tokens: List[List[int]] = [[] for _ in range(n_groups)]
        group_recon_infos: List[dict] = [{"stop_reason": None} for _ in range(n_groups)]
        valid_gis = [gi for gi in range(n_groups) if prefix_embeds_list[gi] is not None and group_max_recon_lens[gi] > 0]
        if valid_gis:
            sampling_params: Dict[str, Any] = {
                "temperature": float(self._temperature),
                "top_p": 1.0,
                "max_tokens": int(max(group_max_recon_lens[gi] for gi in valid_gis)),
            }
            batch_embeds = [prefix_embeds_list[gi] for gi in valid_gis]  # type: ignore[list-item]
            outputs = self._vllm_manager.generate_from_embeddings(batch_embeds, sampling_params=sampling_params)
            for gi, out in zip(valid_gis, outputs):
                if not out.outputs:
                    continue
                token_ids = out.outputs[0].token_ids or []
                recon_tokens, info = _postprocess_recon(token_ids, group_max_recon_lens[gi])
                group_recon_tokens[gi] = recon_tokens
                group_recon_infos[gi] = info

        # Pre-embed query tokens per group to avoid re-encoding for every option.
        d_model = int(getattr(self.model.args, "d_model", 0))
        group_query_embeds: List[torch.Tensor] = []
        for qtoks in group_query_tokens:
            if add_query and qtoks:
                q = torch.tensor(qtoks, device=self.device, dtype=torch.long)
                with torch.autocast(device_type="cuda", dtype=self._dtype):
                    qe = _token_embed(self.model, q).to(dtype=self._dtype)
                group_query_embeds.append(qe)
            else:
                group_query_embeds.append(torch.empty((0, d_model), device=self.device, dtype=self._dtype))

        # Build base embeds per group: prefix_embeds + recon_embeds (+ optional query suffix).
        group_base_embeds: List[Optional[torch.Tensor]] = [None] * n_groups
        group_base_lens: List[int] = [0] * n_groups

        for gi, pe in enumerate(prefix_embeds_list):
            if pe is None:
                continue
            recon = group_recon_tokens[gi]
            with torch.autocast(device_type="cuda", dtype=self._dtype):
                if recon:
                    rt = torch.tensor(recon, device=self.device, dtype=torch.long)
                    re = _token_embed(self.model, rt).to(dtype=self._dtype)
                else:
                    re = torch.empty((0, d_model), device=self.device, dtype=self._dtype)
            qe = group_query_embeds[gi]
            base = torch.cat([pe, re, qe], dim=0) if (re.numel() or qe.numel()) else pe
            group_base_embeds[gi] = base
            group_base_lens[gi] = int(base.shape[0])

        # Reconstruct-first legacy path uses an all-false compression mask in decoder scoring.
        group_base_masks: List[Optional[torch.Tensor]] = [None] * n_groups
        for gi, base in enumerate(group_base_embeds):
            if base is None:
                continue
            group_base_masks[gi] = torch.zeros(int(group_base_lens[gi]), device=self.device, dtype=torch.bool)

        rows_per_chunk = int(getattr(getattr(self.model, "args", None), "cross_entropy_chunk", 8)) * 16
        rows_per_chunk = max(8, min(int(rows_per_chunk), 512))

        option_stats: Dict[int, Dict[str, Any]] = {}
        fit_pairs: List[Tuple[int, int]] = []

        for gi, idxs in enumerate(group_indices):
            base = group_base_embeds[gi]
            base_mask = group_base_masks[gi]
            base_len = int(group_base_lens[gi])
            if base is None or base_mask is None or base_len <= 0:
                continue
            for req_idx in idxs:
                cont = cont_tokens_list[req_idx]
                if not cont:
                    option_stats[req_idx] = {
                        "ll": 0.0,
                        "greedy": True,
                        "tokens": 0,
                        "loss": 0.0,
                        "ppl": None,
                    }
                    continue
                if base_len + len(cont) <= decoder_budget:
                    fit_pairs.append((req_idx, gi))

        # Batch score options that fit.
        score_bs = max(1, int(self._ppl_batch_size))
        for score_start in range(0, len(fit_pairs), score_bs):
            pairs = fit_pairs[score_start : score_start + score_bs]
            if bool(getattr(self, "_verbose_compress", False)):
                print("pairs,", pairs, "score_start,", score_start, "score_bs,", score_bs)

            seq_embeds: List[torch.Tensor] = []
            pref_lens: List[int] = []
            cont_targets: List[torch.Tensor] = []
            comp_masks: List[torch.Tensor] = []
            orig_req_idxs: List[int] = []

            for req_idx, gi in pairs:
                base = group_base_embeds[gi]
                base_mask = group_base_masks[gi]
                if base is None or base_mask is None:
                    continue
                cont = cont_tokens_list[req_idx]
                if not cont:
                    continue
                cont_t = torch.tensor(cont, device=self.device, dtype=torch.long)
                with torch.autocast(device_type="cuda", dtype=self._dtype):
                    cont_e = _token_embed(self.model, cont_t).to(dtype=self._dtype)
                seq = torch.cat([base, cont_e], dim=0)
                seq_embeds.append(seq)
                base_len = int(group_base_lens[gi])
                pref_lens.append(base_len)
                cont_targets.append(cont_t)
                comp_masks.append(torch.cat([base_mask, torch.zeros(len(cont), device=self.device, dtype=torch.bool)], dim=0))
                orig_req_idxs.append(req_idx)

            if not seq_embeds:
                continue

            score_out = self._forward_score_continuations(
                seq_embeds=seq_embeds,
                cont_targets=cont_targets,
                prefix_lens=pref_lens,
                comp_mask_list=comp_masks,
                rows_per_chunk=rows_per_chunk,
            )
            per_sample = score_out.get("per_sample", [])
            for j, req_idx in enumerate(orig_req_idxs):
                if j >= len(per_sample):
                    continue
                ps = per_sample[j]
                if ps.get("tokens", 0) <= 0:
                    continue
                chunk_results[req_idx] = (ps["ll"], ps["greedy"])
                option_stats[req_idx] = ps

        # Debug rows: interleave group summary with its options.
        debug_rows: List[dict] = []
        verbose_payload = bool(getattr(self, "_verbose_compress", False))
        for gi in range(n_groups):
            pre_tokens_full, _ = _build_pre_output_tokens(gi, group_query_tokens, group_suffix_lens, group_n_spans)
            pre_tokens_no_prefix = list(group_recon_tokens[gi])
            if add_query:
                pre_tokens_no_prefix.extend(group_query_tokens[gi])
            info = group_recon_infos[gi]
            group_first_idx = group_indices[gi][0] if group_indices[gi] else 0

            group_row = {
                "request_index": batch_start + group_first_idx,
                "mode": "reconstruct_first",
                "use_chat_template": chat_enabled,
                "add_query_before_likelihood": add_query,
                "reconstruct_add_bor": add_bor,
                "reconstruct_max_bor": max_bor,
                "num_comp": num_comp,
                "max_mem_span_len": max_mem_span_len,
                "n_spans": group_n_spans[gi],
                "total_comp_slots": group_n_slots[gi],
                "prefix_len": group_prefix_lens[gi],
                "base_len": group_base_lens[gi],
                "max_recon_len": group_max_recon_lens[gi],
                "recon_tokens_len": len(group_recon_tokens[gi]),
                "recon_stop_reason": info.get("stop_reason"),
                "query_len": len(group_query_tokens[gi]),
                "group_max_cont_len": group_max_cont_lens[gi],
                "recon_text_preview": self.tok_decode_w_special_tokens(group_recon_tokens[gi]) if group_recon_tokens[gi] else "",
                # "pre_output_text_preview": self.tok_decode_w_special_tokens(pre_tokens_full) if pre_tokens_full else "",
                "pre_output_text_no_prefix_preview": self.tok_decode_w_special_tokens(pre_tokens_no_prefix)
                if pre_tokens_no_prefix
                else "",
            }
            if verbose_payload:
                group_row["recon_text"] = self.tok_decode_w_special_tokens(group_recon_tokens[gi]) if group_recon_tokens[gi] else ""
                group_row["pre_output_text"] = self.tok_decode_w_special_tokens(pre_tokens_full) if pre_tokens_full else ""
                group_row["pre_output_text_no_prefix"] = (
                    self.tok_decode_w_special_tokens(pre_tokens_no_prefix) if pre_tokens_no_prefix else ""
                )
            debug_rows.append(group_row)

            for req_idx in group_indices[gi]:
                ps = option_stats.get(req_idx)
                opt_row = {
                    "request_index": batch_start + req_idx,
                    "mode": "reconstruct_first_option",
                    "add_query_before_likelihood": add_query,
                    "cont_len": len(cont_tokens_list[req_idx]),
                    "logprob": chunk_results[req_idx][0],
                    "greedy": chunk_results[req_idx][1],
                }
                if ps is not None and int(ps.get("tokens", 0) or 0) > 0:
                    opt_row["loss"] = ps.get("loss")
                    opt_row["ppl"] = ps.get("ppl")
                    opt_row["score_tokens"] = int(ps.get("tokens", 0) or 0)
                    if "windows" in ps:
                        opt_row["windows"] = int(ps.get("windows") or 0)
                    if "rolled" in ps:
                        opt_row["rolled"] = bool(ps.get("rolled"))
                debug_rows.append(opt_row)

        self._append_loglikelihood_debug_rows(debug_rows)

        res.extend(chunk_results)

    return res
