from lm_eval.tasks.ruler.common_utils import postprocess_pred_for_niah


def test_niah_number_extract_plain_7_digits():
    pred = ["The special magic number is 3344545."]
    gold = ["3344545"]
    assert postprocess_pred_for_niah(pred, gold) == ["3344545"]


def test_niah_number_extract_with_commas():
    pred = ["The special magic number is 3,344,545."]
    gold = ["3344545"]
    assert postprocess_pred_for_niah(pred, gold) == ["3344545"]


def test_niah_number_extract_with_spaces():
    pred = ["The special magic number is 3 344 545."]
    gold = ["3344545"]
    assert postprocess_pred_for_niah(pred, gold) == ["3344545"]

