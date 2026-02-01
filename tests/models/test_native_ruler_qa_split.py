import unittest


from lm_eval.models.native_doc_utils import get_doc_query_keys_by_task_name, split_doc_and_query


class TestNativeRulerQASplit(unittest.TestCase):
    def test_get_doc_query_keys_ruler_qa(self):
        keys = get_doc_query_keys_by_task_name("ruler_qa_squad")
        self.assertEqual(keys["doc_key"], "context")
        self.assertEqual(keys["question_key"], "question")

    def test_split_doc_and_query_ruler_qa_prefers_query_prompt(self):
        doc = {
            "context": "Document 1:\nHello world.\n\nDocument 2:\nGoodbye.",
            "question": "Where is the cat?",
            "query_prompt": "Question: Where is the cat? Answer:",
        }
        out = split_doc_and_query(
            active_lg_docs=[doc],
            active_tasks_names=["ruler_qa_squad"],
            # Prompt can contain the full raw documents; for compress_answer we
            # prefer the short `query_prompt` stored in the doc.
            prompt_list=["<very long prompt omitted>"],
            doc_key="context",
            question_key="question",
        )
        self.assertEqual(out["context_list"], [doc["context"]])
        self.assertEqual(out["question_list"], [doc["question"]])
        self.assertEqual(out["query_list"], [doc["query_prompt"]])
        self.assertEqual(out["assistant_prefix_list"], [""])

