"""LLM clients: Anthropic's Messages API (standard library only) and an offline fallback."""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

# USD per 1M tokens, Anthropic list prices (Nov 2026). Haiku 5.5's rate is for prompts under 100K tokens.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
RETRYABLE = {408, 409, 429, 500, 502, 503, 504, 529}


@dataclass
class Reply:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class LLMError(Exception):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class AnthropicLLM:
    def __init__(self, model: str = "claude-haiku-5-5", api_key: str | None = None, retries: int = 4):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.retries = retries

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> Reply:
        payload = {"model": self.model, "max_tokens": max_tokens, "system": system,
                   "messages": [{"role": "user", "content": user}]}
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json", "x-api-key": self.api_key,
                     "anthropic-version": "2023-06-01"},
            method="POST",
        )
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:300]
                if e.code not in RETRYABLE or attempt == self.retries:
                    raise LLMError(f"HTTP {e.code}: {body}", retryable=e.code in RETRYABLE) from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.retries:
                    raise LLMError(f"Network error: {e}", retryable=True) from e
            time.sleep(min(30, 2 ** attempt))
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        in_tok, out_tok = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        price_in, price_out = PRICES.get(self.model, (0.0, 0.0))
        return Reply(text, in_tok, out_tok, (in_tok * price_in + out_tok * price_out) / 1_000_000)


class OfflineLLM:
    """No API key? Answers by quoting the best passage, so the app and tests still run.

    It is not a language model: it returns the first sentences of passage [1] with a citation,
    and for re-ranking it scores passages by word overlap with the question.
    """

    model = "offline-extractive"

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> Reply:
        if "Score each passage" in system:
            q = set(re.findall(r"\w+", user.split("Passages:")[0].lower()))
            passages = re.findall(r"\[(\d+)\] (.*?)(?=\n\n\[\d+\] |\Z)", user, re.S)
            scores = [min(3, len(q & set(re.findall(r"\w+", p.lower()))) // 3) for _, p in passages]
            return Reply(json.dumps({"scores": scores}))
        m = re.search(r"\[1\] \([^)]*\)\n(.*?)(?=\n\n\[2\] |\n\nQuestion:)", user, re.S)
        if not m:
            return Reply("I don't know: no documentation passages were provided.")
        sentences = re.split(r"(?<=[.!?])\s+", " ".join(m.group(1).split()))
        return Reply(" ".join(sentences[:2]) + " [1]")


def make_llm(model: str = "claude-haiku-5-5", offline_ok: bool = True):
    """Anthropic client when a key is set; otherwise the offline fallback (if allowed)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicLLM(model)
    if offline_ok:
        return OfflineLLM()
    raise LLMError("ANTHROPIC_API_KEY is not set")
