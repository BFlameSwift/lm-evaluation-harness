import unittest


from lm_eval.models.native_doc_utils import get_doc_query_keys_by_task_name, split_doc_and_query


class TestNativeBabiLongSplit(unittest.TestCase):
    def test_get_doc_query_keys_babilong(self):
        keys = get_doc_query_keys_by_task_name("babilong_qa1")
        self.assertEqual(keys["doc_key"], "input")
        self.assertEqual(keys["question_key"], "question")

    def test_split_doc_and_query_babilong(self):
        doc = {
            "input": "Alice went to the kitchen. Bob went to the hallway.",
            "question": "Where is Alice?",
        }
        out = split_doc_and_query(
            active_lg_docs=[doc],
            active_tasks_names=["babilong_qa1"],
            prompt_list=[""],
            doc_key="input",
            question_key="question",
        )
        self.assertEqual(out["context_list"], [doc["input"]])
        self.assertEqual(out["question_list"], [doc["question"]])
        self.assertEqual(out["query_list"], ["Question: Where is Alice?\nAnswer:"])
        self.assertEqual(out["assistant_prefix_list"], [""])
