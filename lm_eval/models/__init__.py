from importlib import import_module

# Keep the core local backends eager so the common hf path is always registered.
from . import dummy, huggingface  # noqa: F401


_OPTIONAL_MODEL_MODULES = (
    "anthropic_llms",
    "api_models",
    "gguf",
    "hf_audiolm",
    "hf_steered",
    "hf_vlms",
    "ibm_watsonx_ai",
    "mamba_lm",
    "nemo_lm",
    "native",
    "neuron_optimum",
    "openai_completions",
    "optimum_ipex",
    "optimum_lm",
    "sglang_causallms",
    "sglang_generate_API",
    "textsynth",
    "vllm_causallms",
    "vllm_vlms",
)


for _module_name in _OPTIONAL_MODEL_MODULES:
    try:
        import_module(f"{__name__}.{_module_name}")
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
