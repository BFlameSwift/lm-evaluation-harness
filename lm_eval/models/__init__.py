from . import (
    anthropic_llms,
    api_models,
    dummy,
    gguf,
    hf_audiolm,
    hf_steered,
    hf_vlms,
    huggingface,
    ibm_watsonx_ai,
    mamba_lm,
    nemo_lm,
    neuron_optimum,
    openai_completions,
    optimum_ipex,
    optimum_lm,
    sglang_causallms,
    sglang_generate_API,
    native,
    textsynth,
)

# The `native` backend is a local extension in some downstream forks and may
# depend on extra modules (e.g. custom model code) not present in a vanilla
# lm-evaluation-harness install. Keep it optional so the package can still be
# imported in minimal environments (including CI).
try:
    from . import native  # noqa: F401
except Exception:
    pass

# vLLM is an optional dependency. Importing it unconditionally can break
# environments where CUDA runtime libs are unavailable or mismatched.
try:
    from . import vllm_causallms, vllm_vlms  # noqa: F401
except Exception:
    pass


# TODO: implement __all__


try:
    # enable hf hub transfer if available
    import hf_transfer  # type: ignore # noqa
    import huggingface_hub.constants  # type: ignore

    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
except ImportError:
    pass
