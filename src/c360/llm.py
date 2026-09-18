"""
LLM access -- one seam, two providers, and an offline fallback.

WHY AN ABSTRACTION RATHER THAN CALLING GEMINI DIRECTLY
------------------------------------------------------
Three reasons, all of them practical:

1. THE TESTS MUST RUN WITHOUT A NETWORK OR A KEY. A test suite that needs a live
   free-tier API is a test suite that fails at the worst moment, gives different
   answers on different runs, and cannot be run by an examiner. Every component
   here works with `NullLLM`, which is deterministic; the real models improve
   quality without being load-bearing for correctness.

2. FREE TIER MEANS RATE LIMITS. Gemini flash-lite and Groq Llama have per-minute
   caps. Having one place that retries, backs off and can fail over from one
   provider to the other is much better than that logic being sprinkled across
   four agents.

3. TWO MODEL TIERS. The brief allocates flash-lite to volume work and a stronger
   model to heavy reasoning. `get_llm("fast")` and `get_llm("reasoning")` make
   that a configuration decision rather than a hard-coded model name in an agent.

WHAT THE SYSTEM USES AN LLM FOR -- AND WHAT IT DELIBERATELY DOES NOT
--------------------------------------------------------------------
Used for free text, where language understanding is the actual job: support
ticket bodies, in-app search queries, consented social posts, and the synthesis /
action / critique reasoning steps.

NOT used for arithmetic. Spend rates, income drops, balance trends, login
frequency and the corroboration count are computed in code. That is not laziness:

  - The guardrail is required by the problem statement to be an actual code
    check, not a prompt asking a model to be careful.
  - An LLM asked "is 0.14 logins/day a big drop from 0.9?" will usually be right
    and occasionally be confidently wrong, with no way to tell which. The same
    comparison in Python is right every time, costs nothing, and can be shown to
    an examiner as a line of code.

Being able to state that division of labour clearly is worth more in a viva than
routing everything through a model.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

# Model choices, overridable from .env so no code change is needed to swap them.
# Defaults verified against the live provider catalogues on 18 Sept 2026. The
# originals -- gemini-3.1-pro and llama-3.3-70b-versatile -- had both been
# retired, so a clean clone with valid keys still answered 404 on every call.
# Model names are configuration and they rot; `python compare_live.py --models`
# lists what a given key can actually reach.
FAST_MODEL = os.getenv("C360_FAST_MODEL", "gemini-3.1-flash-lite")
REASONING_MODEL = os.getenv("C360_REASONING_MODEL", "gemini-3.1-flash-lite")
GROQ_MODEL = os.getenv("C360_GROQ_MODEL", "openai/gpt-oss-20b")

# Seconds to wait for one request before giving up on it. Deliberately short:
# every caller has a deterministic fallback, so a slow answer is worth less than
# a prompt one, and the run has to finish.
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


# ---------------------------------------------------------------------------
# The offline fallback
# ---------------------------------------------------------------------------

class NullLLM:
    """
    Returns nothing useful, on purpose.

    Every caller in this system must supply a deterministic fallback for when the
    LLM is unavailable, and NullLLM is how that path gets exercised in the tests.
    It raises rather than inventing an answer: a stub that quietly returns
    plausible-looking JSON would let a broken fallback pass its own tests.
    """

    provider = "null"
    model = "none"

    def complete(self, system: str, user: str) -> str:
        raise LLMUnavailable("no LLM configured; caller must use its deterministic fallback")


# ---------------------------------------------------------------------------
# Real providers
# ---------------------------------------------------------------------------

def message_text(content: Any) -> str:
    """
    Pull the text out of a chat response.

    LangChain returns `.content` as a plain string for some providers and as a
    LIST OF CONTENT BLOCKS for others -- Gemini returns
    [{"type": "text", "text": "...", "extras": {...}}]. The original code did
    `str(content)` for the non-string case, which stringified the Python list
    and handed the JSON parser `{'type': 'text', ...}` -- Python repr, single
    quotes, not JSON. Every model call in the system therefore failed to parse
    and fell back to its deterministic path, silently, while the model was
    answering correctly all along.
    """
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
        # temperature=0 throughout. This system is graded on reproducibility and
        # on a scoring harness comparing runs; sampling variation would mean the
        # same input scores differently on different days.
        from langchain_google_genai import ChatGoogleGenerativeAI

        self.model = model
        # timeout: without one a hung request blocks the replay forever. An
        # ambient system that stops advancing is worse than one that misses a
        # classification, because the fallback is designed to cover the second.
        # max_retries=0: ResilientLLM already retries with backoff and then fails
        # over. Leaving the client's own retries on stacks two loops and turns a
        # 30-second failure into several minutes.
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


# ---------------------------------------------------------------------------
# Resilience wrapper
# ---------------------------------------------------------------------------

class ResilientLLM:
    """
    Retries with backoff, then fails over to a second provider, then gives up.

    Giving up is a first-class outcome, not an exception that kills the run. A
    73-day replay that dies on day 40 because a free-tier quota reset is worse
    than one that completes with a few checkpoints decided by the deterministic
    fallback -- and the run log records exactly which ones, so the evaluation
    write-up can report it honestly.
    """

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
        # Carry the provider's own error into the message. "all providers
        # exhausted" alone says a call failed but not why, which is the
        # difference between a quota problem, a bad model name and a network
        # fault -- three very different fixes.
        # One error per distinct provider/model, not the last N. The failover
        # means the final errors are always the fallback's, which hides why the
        # primary was abandoned in the first place.
        recent = [c for c in self.calls[-(self.attempts * 2):] if c.error]
        seen: dict[str, str] = {}
        for call in recent:
            seen.setdefault(f"{call.provider}/{call.model}", call.error or "")
        detail = " | ".join(f"{k}: {v}" for k, v in seen.items()) or "no provider error recorded"
        raise LLMUnavailable(f"all providers exhausted -- {detail}")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

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
    """
    Build a client for a role, or NullLLM if nothing is configured.

    role="fast"      high-volume text classification -> flash-lite
    role="reasoning" synthesis / action / critique   -> pro, failing over to Groq

    Set C360_OFFLINE=1 to force NullLLM -- used by the test suite so a machine
    that happens to have keys in its environment still runs the deterministic
    path rather than silently making paid-tier-shaped API calls during pytest.
    """
    if allow_network is None:
        allow_network = os.getenv("C360_OFFLINE", "").strip() not in ("1", "true", "yes")
    if not allow_network:
        return NullLLM()

    has_gemini = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    has_groq = bool(os.getenv("GROQ_API_KEY"))

    # Each provider is built independently and its failure recorded rather than
    # raised. The earlier version constructed both inside one try/except, so a
    # missing langchain-groq -- the FALLBACK -- also took out Gemini, the
    # primary, and the run went silently deterministic with keys sitting right
    # there in .env. A provider that cannot be built is one provider lost, not
    # the whole live path.
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


# ---------------------------------------------------------------------------
# Parsing what comes back
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_response(text: str) -> dict[str, Any]:
    """
    Pull a JSON object out of a model response.

    Models wrap JSON in ```json fences, prepend "Here's the analysis:", or add a
    trailing sentence -- despite being told not to. Rather than trusting the
    instruction, extract the outermost {...} and parse that. A response with no
    JSON at all raises, and the caller falls back to its deterministic path.
    """
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
