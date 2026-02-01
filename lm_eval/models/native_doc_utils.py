from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def _split_niah_input(input_text: str) -> Tuple[str, str]:
    """Split RULER NIAH `input` into (context, question) best-effort."""
    if not input_text:
        return "", ""
    # The official template appends a final question line starting with "What ...".
    marker = "\nWhat "
    idx = input_text.rfind(marker)
    if idx == -1:
        return input_text, ""
    context = input_text[:idx].rstrip()
    question = input_text[idx + 1 :].strip()
    return context, question


def split_doc_and_query(
    active_lg_docs: List[Optional[dict]],
    active_tasks_names: List[Optional[str]],
    prompt_list: List[str],
    *,
    doc_key: str = "context",
    question_key: str = "question",
    niah_use_bor: bool = False,
) -> Dict[str, List[str]]:
    """
    Build (context, question, query) triples from structured docs.

    This is used by native-rag's harness adapter to compress only the long context
    portion while keeping the query prompt uncompressed (completion-style).
    """
    if not active_lg_docs or not active_tasks_names:
        return {"context_list": [], "question_list": [], "query_list": [], "assistant_prefix_list": []}

    ret_context_list: List[str] = []
    ret_question_list: List[str] = []
    ret_query_list: List[str] = []
    ret_assistant_prefix_list: List[str] = []

    for doc, task_name, prompt in zip(active_lg_docs, active_tasks_names, prompt_list):
        if doc is None or task_name is None:
            continue
        task_name_lower = str(task_name).lower()
        is_longbench2_task = "longbench2" in task_name_lower
        is_longbench1_task = task_name_lower.startswith("longbench_") and ("longbench2" not in task_name_lower)
        is_babilong_task = task_name_lower.startswith("babilong")
        is_ruler_custom_task = task_name_lower.startswith("ruler_custom")
        is_ruler_qa_task = task_name_lower.startswith("ruler_qa_")

        if is_longbench2_task:
            context_text = doc[doc_key]
            question_text = doc[question_key]
            real_query_text = prompt.split("\n</text>\n\n")[1].strip()

            ret_question_list.append(question_text)
            ret_query_list.append(real_query_text)
            ret_context_list.append(context_text)
            ret_assistant_prefix_list.append("")

        elif "niah" in task_name_lower:
            input_text = doc.get("input", "")
            if not isinstance(input_text, str):
                input_text = str(input_text)
            gen_prefix = doc.get("gen_prefix", "")
            if not isinstance(gen_prefix, str):
                gen_prefix = str(gen_prefix)
            gen_prefix = gen_prefix.strip()

            context_text, question_text = _split_niah_input(input_text)
            query_text = question_text.strip()
            assistant_prefix = "" if niah_use_bor else gen_prefix

            ret_question_list.append(question_text)
            ret_query_list.append(query_text)
            ret_context_list.append(context_text)
            ret_assistant_prefix_list.append(assistant_prefix)

        elif "infinitebench" in task_name_lower:
            context_text = doc.get("context", "")
            if not isinstance(context_text, str):
                context_text = str(context_text)
            question_text = doc.get("input", "")
            if not isinstance(question_text, str):
                question_text = str(question_text)
            real_query_text = "\n\n" + f"Question: {question_text}\nAnswer:"

            ret_question_list.append(question_text)
            ret_query_list.append(real_query_text)
            ret_context_list.append(context_text)
            ret_assistant_prefix_list.append("")

        elif is_longbench1_task:
            context_text = doc.get(doc_key, "")
            if not isinstance(context_text, str):
                context_text = str(context_text)
            question_text = doc.get(question_key, "")
            if not isinstance(question_text, str):
                question_text = str(question_text)

            real_query_text = prompt
            if context_text and context_text in real_query_text:
                prefix, suffix = real_query_text.split(context_text, 1)
                real_query_text = (prefix + suffix).strip()
            elif question_text and question_text in real_query_text:
                real_query_text = real_query_text[real_query_text.index(question_text) :].strip()
            else:
                real_query_text = real_query_text.strip()

            ret_question_list.append(question_text)
            ret_query_list.append(real_query_text)
            ret_context_list.append(context_text)
            ret_assistant_prefix_list.append("")

        elif is_babilong_task:
            context_text = doc.get(doc_key, "")
            if not isinstance(context_text, str):
                context_text = str(context_text)
            question_text = doc.get(question_key, "")
            if not isinstance(question_text, str):
                question_text = str(question_text)
            question_text = question_text.strip()
            real_query_text = f"Question: {question_text}\nAnswer:"

            ret_question_list.append(question_text)
            ret_query_list.append(real_query_text)
            ret_context_list.append(context_text)
            ret_assistant_prefix_list.append("")

        elif is_ruler_custom_task:
            input_text = doc.get("input", "")
            if not isinstance(input_text, str):
                input_text = str(input_text)

            # Our custom RULER-style tasks store context + a short QA suffix:
            #   <context>\n\nQuestion: ...\nAnswer:
            # Keep the "Question/Answer:" part uncompressed so base models can
            # complete naturally.
            marker = "\n\nQuestion:"
            idx = input_text.rfind(marker)
            if idx == -1:
                # Fallback: try a single newline marker.
                marker = "\nQuestion:"
                idx = input_text.rfind(marker)

            if idx == -1:
                context_text = input_text
                query_text = ""
            else:
                context_text = input_text[:idx].rstrip()
                query_text = input_text[idx:].strip()

            # Extract the human-readable question (best-effort).
            question_text = query_text
            if query_text:
                q = query_text
                if q.lower().startswith("question:"):
                    q = q[len("question:") :].lstrip()
                # Strip trailing "Answer:" if present.
                lower_q = q.lower()
                ans_idx = lower_q.rfind("\nanswer:")
                if ans_idx != -1:
                    q = q[:ans_idx].strip()
                question_text = q.strip()

            ret_question_list.append(question_text)
            ret_query_list.append(query_text)
            ret_context_list.append(context_text)
            ret_assistant_prefix_list.append("")

        elif is_ruler_qa_task:
            # RULER QA datasets (SQuAD / HotpotQA) store:
            # - `context`: long documents (to be compressed)
            # - `question`: the question string
            # - `query_prompt`: short prompt suffix including the Answer: prefix (uncompressed)
            context_text = doc.get("context", "")
            if not isinstance(context_text, str):
                context_text = str(context_text)

            question_text = doc.get("question", "")
            if not isinstance(question_text, str):
                question_text = str(question_text)
            question_text = question_text.strip()

            query_text = doc.get("query_prompt", "")
            if not isinstance(query_text, str):
                query_text = str(query_text)
            query_text = query_text.strip()

            # Fallback: if `query_prompt` isn't present, try to remove the long
            # document block from the full rendered prompt.
            if not query_text:
                real_query_text = prompt or ""
                if context_text and context_text in real_query_text:
                    prefix, suffix = real_query_text.split(context_text, 1)
                    real_query_text = (prefix + suffix).strip()
                if not real_query_text:
                    real_query_text = f"Question: {question_text}\nAnswer:"
                query_text = real_query_text

            ret_question_list.append(question_text)
            ret_query_list.append(query_text)
            ret_context_list.append(context_text)
            ret_assistant_prefix_list.append("")

        else:
            raise ValueError(f"Unsupported task: {task_name}")

    return {
        "context_list": ret_context_list,
        "question_list": ret_question_list,
        "query_list": ret_query_list,
        "assistant_prefix_list": ret_assistant_prefix_list,
    }


def get_doc_query_keys_by_task_name(task_name: str) -> Dict[str, str]:
    task_name_lower = str(task_name).lower()
    if "longbench2" in task_name_lower:
        return {
            "doc_key": "context",
            "question_key": "question",
        }
    if "infinitebench" in task_name_lower:
        return {
            "doc_key": "context",
            "question_key": "input",
        }
    if task_name_lower.startswith("longbench_") and ("longbench2" not in task_name_lower):
        return {
            "doc_key": "context",
            # LongBench v1 uses `input` for the query field.
            "question_key": "input",
        }
    if "niah" in task_name_lower:
        return {
            "doc_key": "input",
            "question_key": "input",
        }
    if task_name_lower.startswith("babilong"):
        return {
            "doc_key": "input",
            "question_key": "question",
        }
    if task_name_lower.startswith("ruler_custom"):
        # RULER extra synthetic tasks use a single `input` field that contains
        # both long context and the QA suffix.
        return {
            "doc_key": "input",
            "question_key": "input",
        }
    if task_name_lower.startswith("ruler_qa_"):
        return {
            "doc_key": "context",
            "question_key": "question",
        }
    return {
        "doc_key": "context",
        "question_key": "question",
    }
