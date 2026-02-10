"""
Stable entrypoint for the `native` model backend.

The full implementation lives in `lm_eval.models.native_impl` to keep this file
small and make the codebase easier to refactor.
"""

from lm_eval.api.registry import register_model

from .native_impl import NativeCausalLM as _NativeCausalLM


@register_model("native")
class NativeCausalLM(_NativeCausalLM):
    """Compatibility wrapper that preserves the public import path."""

    pass

