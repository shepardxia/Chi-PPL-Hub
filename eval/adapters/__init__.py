"""Pluggable LLM adapters for eval/run_eval.py.

An adapter is any callable (or class with a `.generate(entry) -> GenerationResult`
method) that maps a dataset entry to a `GenerationResult`.

The registry below lets the CLI select adapters by name.
"""

from .base import GenerationResult, ModelAdapter  # noqa: F401
from .mock import GroundTruthAdapter


ADAPTER_REGISTRY = {
    "groundtruth": GroundTruthAdapter,
}


def _register_anthropic():
    """Lazy import so missing SDK doesn't break other adapters."""
    try:
        from .anthropic_adapter import AnthropicAdapter
        ADAPTER_REGISTRY["anthropic"] = AnthropicAdapter
    except ImportError:
        pass


_register_anthropic()
