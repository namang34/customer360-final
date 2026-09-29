"""LLM access -- one seam, two providers, and an offline fallback."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

# Model choices, overridable from .env so no code change is needed to swap them.
# Defaults verified against the live provider catalogues on 18 Sept 2026.
FAST_MODEL = os.getenv("C360_FAST_MODEL", "gemini-3.1-flash-lite")
REASONING_MODEL = os.getenv("C360_REASONING_MODEL", "gemini-3.1-flash-lite")
GROQ_MODEL = os.getenv("C360_GROQ_MODEL", "openai/gpt-oss-20b")

# Seconds to wait for one request before giving up on it.
REQUEST_TIMEOUT = float(os.getenv("C360_REQUEST_TIMEOUT", "25"))


@dataclass
class LLMCall:
    """One request/response pair, kept for the trace and the run log."""

    role: str
    provider: str
    model: str
    system: str
    user: str
    response: str
    latency_ms: float
    error: str | None = None


class LLMUnavailable(RuntimeError):
    pass


class LLM(Protocol):
    provider: str
    model: str

    def complete(self, system: str, user: str) -> str: ...


# The offline fallback

class NullLLM:
    """Returns nothing useful, on purpose."""

    provider = "null"
    model = "none"

    def complete(self, system: str, user: str) -> str:
        raise LLMUnavailable("no LLM configured; caller must use its deterministic fallback")


# Real providers

def message_text(content: Any) -> str:
    """Pull the text out of a chat response."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)
    return str(content)


class GeminiLLM:
    provider = "gemini"

    def __init__(self, model: str = FAST_MODEL, temperature: float = 0.0) -> None:
        # temperature=0 throughout.
        from langchain_google_genai import ChatGoogleGenerativeAI

        self.model = model
        # timeout: without one a hung request blocks the replay forever.
        self._client = ChatGoogleGenerativeAI(
            model=model, temperature=temperature, timeout=REQUEST_TIMEOUT, max_retries=0
        )

    def complete(self, system: str, user: str) -> str:
        message = self._client.invoke([("system", system), ("human", user)])
        return message_text(message.content)


class GroqLLM:
    provider = "groq"

    def __init__(self, model: str = GROQ_MODEL, temperature: float = 0.0) -> None:
        from langchain_groq import ChatGroq

        self.model = model
        self._client = ChatGroq(
            model=model, temperature=temperature, timeout=REQUEST_TIMEOUT, max_retries=0
        )

    def complete(self, system: str, user: str) -> str:
        message = self._client.invoke([("system", system), ("human", user)])
        return message_text(message.content)


# Resilience wrapper

class ResilientLLM:
    """Retries with backoff, then fails over to a second provider, then gives up."""

    def __init__(
        self,
        primary: LLM,
        fallback: LLM | None = None,
        attempts: int = 3,
        base_delay: float = 2.0,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.attempts = attempts
        self.base_delay = base_delay
        self.calls: list[LLMCall] = []
        self.failures = 0

    @property
    def provider(self) -> str:
        return self.primary.provider

    @property
    def model(self) -> str:
        return self.primary.model

    def complete(self, system: str, user: str, *, role: str = "unknown") -> str:
        for client in [c for c in (self.primary, self.fallback) if c is not None]:
            for attempt in range(self.attempts):
                started = time.perf_counter()
                try:
                    response = client.complete(system, user)
                    self.calls.append(
                        LLMCall(
                            role=role,
                            provider=client.provider,
                            model=client.model,
                            system=system,
                            user=user,
                            response=response,
                            latency_ms=(time.perf_counter() - started) * 1000,
                        )
                    )
                    return response
                except LLMUnavailable:
                    raise
                except Exception as exc:  # noqa: BLE001 -- provider SDKs raise many types
                    self.calls.append(
                        LLMCall(
                            role=role,
                            provider=client.provider,
                            model=client.model,
                            system=system,
                            user=user,
                            response="",
                            latency_ms=(time.perf_counter() - started) * 1000,
                            error=str(exc)[:300],
                        )
                    )
                    if attempt < self.attempts - 1:
                        time.sleep(self.base_delay * (2**attempt))
        self.failures += 1
        # Carry the provider's own error into the message.
        recent = [c for c in self.calls[-(self.attempts * 2):] if c.error]
        seen: dict[str, str] = {}
        for call in recent:
            seen.setdefault(f"{call.provider}/{call.model}", call.error or "")
        detail = " | ".join(f"{k}: {v}" for k, v in seen.items()) or "no provider error recorded"
        raise LLMUnavailable(f"all providers exhausted -- {detail}")


# Construction

def _build(factory, label: str, notes: list[str]):
    """Construct one provider, recording why it could not be built."""
    try:
        return factory()
    except Exception as exc:  # noqa: BLE001 -- provider SDKs raise many types
        notes.append(f"{label}: {type(exc).__name__}: {exc}")
        return None


def get_llm(
    role: str = "fast",
    *,
    allow_network: bool | None = None,
    problems: list[str] | None = None,
) -> LLM:
    """Build a client for a role, or NullLLM if nothing is configured."""
    if allow_network is None:
        allow_network = os.getenv("C360_OFFLINE", "").strip() not in ("1", "true", "yes")
    if not allow_network:
        return NullLLM()

    has_gemini = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    has_groq = bool(os.getenv("GROQ_API_KEY"))

    # Each provider is built independently and its failure recorded rather than raised.
    notes: list[str] = [] if problems is None else problems
    model = REASONING_MODEL if role == "reasoning" else FAST_MODEL

    gemini = _build(lambda: GeminiLLM(model), f"gemini({model})", notes) if has_gemini else None
    groq = _build(GroqLLM, f"groq({GROQ_MODEL})", notes) if has_groq else None

    if gemini is not None:
        return ResilientLLM(gemini, groq)
    if groq is not None:
        return ResilientLLM(groq)
    if not has_gemini and not has_groq:
        notes.append("no API key found in the environment (GOOGLE_API_KEY / GROQ_API_KEY)")
    return NullLLM()


# Parsing what comes back

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_response(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response."""
    if not text:
        raise ValueError("empty response")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(cleaned)
    if not match:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    return json.loads(match.group(0))


def complete_json(
    llm: LLM, system: str, user: str, *, role: str = "unknown"
) -> dict[str, Any]:
    """Call the model and parse JSON, letting both failure modes reach the caller."""
    if isinstance(llm, ResilientLLM):
        raw = llm.complete(system, user, role=role)
    else:
        raw = llm.complete(system, user)
    return parse_json_response(raw)
