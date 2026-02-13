import inspect


def test_native_scoring_mixin_is_used():
    # Importing these modules should not require GPU-only deps; native_impl/model.py
    # keeps `arch/` and `eval_func/` imports best-effort/lazy.
    from lm_eval.models.native_impl.model import NativeCausalLM
    from lm_eval.models.native_impl.scoring_mixin import ScoringMixin

    assert issubclass(NativeCausalLM, ScoringMixin)

    # Spot-check that key methods come from the mixin (not duplicated in model.py).
    for name in (
        "_build_verifier_candidate_tokens",
        "_score_verifier_yes_no_from_tokens",
        "_chunked_logprob_and_greedy",
        "_forward_score_token_ranges",
        "_forward_score_continuations",
    ):
        fn = getattr(NativeCausalLM, name)
        assert inspect.isfunction(fn)
        assert fn.__qualname__.startswith("ScoringMixin.")

