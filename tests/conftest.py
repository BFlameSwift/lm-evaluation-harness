import os
from pathlib import Path


def _setdefault_env(key: str, value: str) -> None:
    if key not in os.environ or not os.environ.get(key):
        os.environ[key] = value


# pytest imports `conftest.py` early. Set offline + writable cache locations
# before any `datasets`/`transformers` modules are imported by other tests.
_HF_ROOT = Path(os.environ.get("LM_EVAL_TEST_HF_ROOT", "/tmp/hf")).resolve()
_setdefault_env("HF_HOME", str(_HF_ROOT))
_setdefault_env("HF_DATASETS_CACHE", str(_HF_ROOT / "datasets"))
_setdefault_env("HF_HUB_CACHE", str(_HF_ROOT / "hub"))
_setdefault_env("HF_MODULES_CACHE", str(_HF_ROOT / "modules"))

# Run in offline mode in CI/sandbox environments to avoid flaky network calls.
_setdefault_env("HF_HUB_OFFLINE", "1")
_setdefault_env("HF_DATASETS_OFFLINE", "1")
_setdefault_env("TRANSFORMERS_OFFLINE", "1")
_setdefault_env("TOKENIZERS_PARALLELISM", "false")

(_HF_ROOT / "datasets").mkdir(parents=True, exist_ok=True)
(_HF_ROOT / "hub").mkdir(parents=True, exist_ok=True)
(_HF_ROOT / "modules").mkdir(parents=True, exist_ok=True)

