"""
vLLM integration tests.

This file is intentionally skipped by default because it requires CUDA and can
be slow/flaky in CI environments (downloads weights, spawns subprocesses, etc).
"""

import pytest

pytest.skip("vLLM integration test (requires CUDA)", allow_module_level=True)
