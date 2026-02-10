"""
Implementation package for the `native` harness model.

Why this exists:
- `lm_eval/models/native.py` is kept as the stable entrypoint that registers the
  model name `native` with lm-evaluation-harness.
- The actual implementation lives under `native_impl/` so we can split large
  subsystems (likelihood / generation / reconstruction backends) into multiple
  modules without fighting `native.py`'s historical size.
"""

from .model import NativeCausalLM

__all__ = ["NativeCausalLM"]

