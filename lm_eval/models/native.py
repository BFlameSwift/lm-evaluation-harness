"""
Stable entrypoint for the `native` model backend.

The full implementation lives in `lm_eval.models.native_impl` to keep this file
small and make the codebase easier to refactor.
"""

from lm_eval.api.registry import register_model

from .native_impl import NativeCausalLM as _NativeCausalLM
from .native_impl.model import _split_doc_and_query
from .native_impl.utils import parse_mode as _parse_mode


@register_model("native")
class NativeCausalLM(_NativeCausalLM):
    """Compatibility wrapper that preserves the public import path."""

    pass


__all__ = ["NativeCausalLM", "_parse_mode", "_split_doc_and_query"]
