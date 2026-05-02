"""Anthropic (Claude) adapter — atom-based prompting.

For each atom, makes one API call with [system, user] messages and parses
the response into a single WebPPL program. The system prompt optionally
includes the WebPPL primer (default on).

Reads `ANTHROPIC_API_KEY` from env.
"""

from __future__ import annotations

import time

from anthropic import Anthropic

from .base import GenerationResult
from ..prompt import format_messages, parse_response


DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0


class AnthropicAdapter:
    """One API call per atom."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        with_primer: bool = True,
        client: Anthropic | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.with_primer = with_primer
        self.client = client or Anthropic()
        self.name = f"anthropic:{model}{'' if with_primer else '-noprimer'}"

    def generate(self, atom: dict) -> GenerationResult:
        messages = format_messages(atom, with_primer=self.with_primer)
        system_msg = messages[0]["content"]
        user_msgs = [{"role": m["role"], "content": m["content"]} for m in messages[1:]]
        t0 = time.time()
        try:
            resp = self.client.messages.create(
                model=self.model,
                system=system_msg,
                messages=user_msgs,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            return GenerationResult(
                code="",
                raw_response="",
                parse_warnings=[f"API error: {type(e).__name__}: {e}"],
                api_metadata={
                    "error": str(e),
                    "latency_sec": round(time.time() - t0, 3),
                },
            )
        latency = round(time.time() - t0, 3)
        text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        text = "\n".join(text_parts)
        code, warnings = parse_response(text)
        return GenerationResult(
            code=code,
            raw_response=text,
            parse_warnings=warnings,
            api_metadata={
                "model": self.model,
                "with_primer": self.with_primer,
                "stop_reason": resp.stop_reason,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "latency_sec": latency,
            },
        )
