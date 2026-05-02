"""Base types for atom-based adapters.

An adapter takes one atom (dict with `prompt`, etc.) and returns a
GenerationResult with one WebPPL program. Parsing failures and API
errors are captured in fields, not raised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class GenerationResult:
    code: str
    raw_response: str = ""
    parse_warnings: list[str] = field(default_factory=list)
    api_metadata: dict = field(default_factory=dict)


class ModelAdapter(Protocol):
    def generate(self, atom: dict) -> GenerationResult:
        ...
