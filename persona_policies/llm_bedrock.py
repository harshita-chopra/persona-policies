"""
LiteLLM calls (AWS Bedrock, Gemini, OpenRouter, etc.).

Uses ``litellm.completion``. For ``bedrock/...`` models, OpenAI env vars are
dropped so LiteLLM does not route to OpenAI.
OpenRouter: set ``OPENROUTER_API_KEY`` and use model ids ``openrouter/<provider>/<model>``
(see LiteLLM OpenRouter docs).
Gemini: set ``GEMINI_API_KEY`` and use model ids ``gemini/<model>``.

OpenRouter hardening (enabled automatically for ``openrouter/...`` models):
  * Bounded transport timeout + bounded wall-clock per call.
  * Exponential-backoff retries that treat ``httpx.RemoteProtocolError``
    (the common "peer closed connection / incomplete chunked read" mid-stream
    cut), ``httpx.ReadTimeout``/``ConnectError``, empty completions, and HTTP
    429/5xx as transient.
  * OpenRouter provider routing via ``extra_body``: ``allow_fallbacks=true``,
    ``sort='throughput'``, and a ``preferred_max_latency`` hint so slow
    backend providers get deprioritized.
  * ``HTTP-Referer`` / ``X-Title`` client identity headers — OpenRouter
    preferentially routes identified clients.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List

import litellm

logger = logging.getLogger(__name__)

# Reduce generic "Give Feedback / Get Help" stderr spam on transient API errors.
os.environ.setdefault("LITELLM_LOG", "ERROR")
if hasattr(litellm, "suppress_debug_info"):
    litellm.suppress_debug_info = True  # type: ignore[attr-defined]
litellm.set_verbose = False  # type: ignore[misc]

# Per-call transport timeout. OpenRouter free-tier preview models (e.g.
# qwen3.6-plus) can stall for minutes behind a cheapest-provider router; cap
# the wait so retries can actually fire.
_OPENROUTER_TIMEOUT_S = float(os.environ.get("PERSONA_OPENROUTER_TIMEOUT_S", "90"))
# Max retry attempts (total = 1 initial + _MAX_RETRIES). OpenRouter transient
# stream cuts are common enough that <3 retries often burns an iteration.
_OPENROUTER_RETRIES = int(os.environ.get("PERSONA_OPENROUTER_RETRIES", "3"))
# Base delay for exponential backoff, capped at 30 s.
_OPENROUTER_RETRY_BASE_S = float(os.environ.get("PERSONA_OPENROUTER_RETRY_BASE_S", "2"))
_OPENROUTER_RETRY_MAX_S = 30.0
_GEMINI_REASONING_EFFORT = os.environ.get(
    "PERSONA_GEMINI_REASONING_EFFORT", "minimal"
).strip().lower()

# Client identity headers — OpenRouter gives preferential routing to identified
# clients and shows this in their rankings dashboard.
_OPENROUTER_HEADERS: Dict[str, str] = {
    "HTTP-Referer": os.environ.get(
        "OPENROUTER_HTTP_REFERER", "https://github.com/persona-policies"
    ),
    "X-Title": os.environ.get("OPENROUTER_X_TITLE", "persona-policies"),
}

_cached_fallback_models: List[str] | None = None


def _openrouter_fallback_models() -> List[str]:
    """Lazy-load fallback model ids from PersonaPoliciesConfig (cached)."""
    global _cached_fallback_models
    if _cached_fallback_models is None:
        try:
            from persona_policies.config import PersonaPoliciesConfig
            _cached_fallback_models = list(PersonaPoliciesConfig().llm_fallback_models)
        except Exception:
            _cached_fallback_models = []
    return _cached_fallback_models


def _direct_fallback_models(primary_model: str) -> List[str]:
    """Fallbacks not handled by OpenRouter's own ``models`` failover list."""
    out: List[str] = []
    for model in _openrouter_fallback_models():
        m = str(model or "").strip()
        if not m or m == primary_model or m.lower().startswith("openrouter/"):
            continue
        out.append(m)
    return out


def _to_openrouter_body_id(model_id: str) -> str:
    """OpenRouter's request-body ``models`` list expects bare ids without the
    ``openrouter/`` LiteLLM prefix (e.g. ``qwen/qwen3-235b-a22b-2507``)."""
    m = str(model_id)
    return m[len("openrouter/"):] if m.lower().startswith("openrouter/") else m


def prepare_litellm_env_for_models(*model_ids: str) -> None:
    """Unset OpenAI env vars when any configured model is Bedrock (unless ``TAU2_ALLOW_OPENAI=1``)."""
    if os.environ.get("TAU2_ALLOW_OPENAI") == "1":
        return
    if not any(str(m).lower().startswith("bedrock/") for m in model_ids if m):
        return
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        os.environ.pop(key, None)


def prepare_litellm_env(model: str) -> None:
    """Unset OpenAI env vars for a single LiteLLM call when ``model`` is Bedrock."""
    prepare_litellm_env_for_models(model)


def _is_openrouter(model: str) -> bool:
    return str(model).lower().startswith("openrouter/")


def _is_gemini(model: str) -> bool:
    return str(model).lower().startswith("gemini/")


def _uses_gemini_thinking(model: str) -> bool:
    m = str(model).lower()
    return _is_gemini(m) and "gemini" in m and "gemma" not in m


def _openrouter_extra_body(primary_model: str) -> Dict[str, Any]:
    """OpenRouter provider routing hints that trade a little price for reliability.

    - ``sort='throughput'`` picks the fastest currently-healthy provider for the
      model (disables OpenRouter's default price-based load balancing).
    - ``allow_fallbacks=True`` keeps automatic failover between providers.
    - ``preferred_max_latency.p90: 30`` deprioritizes providers whose recent p90
      latency exceeded 30 s (soft hint, never blocks the request).
    - ``models=[...]`` enables model-level failover: if the primary model errors
      out upstream, OpenRouter transparently tries the next id in the list
      before returning a failure to us. Priced by whichever model actually
      fulfilled the request.
    """
    body: Dict[str, Any] = {
        "provider": {
            "sort": "throughput",
            "allow_fallbacks": True,
            "preferred_max_latency": {"p90": 30},
        }
    }
    # Build the model failover chain: primary first, then configured fallbacks
    # (dedup so the primary isn't repeated).
    chain: List[str] = [_to_openrouter_body_id(primary_model)]
    for fb in _openrouter_fallback_models():
        if not str(fb).lower().startswith("openrouter/"):
            continue
        fb_id = _to_openrouter_body_id(fb)
        if fb_id and fb_id not in chain:
            chain.append(fb_id)
    if len(chain) > 1:
        body["models"] = chain
    return body


def _is_transient(exc: BaseException) -> bool:
    """Retry policy: treat stream cuts / timeouts / 429 / 5xx as transient."""
    # Check the class name so we don't have to import httpx here.
    name = type(exc).__name__
    if name in {
        "RemoteProtocolError",
        "ReadTimeout",
        "ConnectTimeout",
        "ConnectError",
        "PoolTimeout",
        "WriteError",
        "Timeout",
        "APIConnectionError",
        "APIError",  # LiteLLM wraps upstream transport errors as APIError
        "ServiceUnavailableError",
        "RateLimitError",
        "InternalServerError",
    }:
        return True
    # Inspect status_code if the exception carries one (LiteLLM APIError,
    # OpenAI-style errors, etc.).
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    # Sometimes only the message has the signal.
    msg = str(exc).lower()
    if any(s in msg for s in (
        "incomplete chunked",
        "peer closed",
        "remote protocol",
        "timed out",
        "timeout",
        "overloaded",
        "rate limit",
        "too many requests",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "connection reset",
    )):
        return True
    return False


def _fmt_exc(exc: BaseException) -> str:
    parts = [type(exc).__name__]
    msg = str(exc).strip()
    if msg:
        parts.append(msg[:400])
    status = getattr(exc, "status_code", None)
    if status is not None:
        parts.append(f"status={status}")
    return ": ".join(parts)


def completion_text(
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """Return assistant message text from a chat completion.

    For ``openrouter/...`` models this retries transient stream cuts /
    timeouts / 429/5xx with exponential backoff and adds provider-routing
    hints that prefer fast, healthy providers.
    """
    prepare_litellm_env(model)

    params: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if _is_openrouter(model):
        params["timeout"] = _OPENROUTER_TIMEOUT_S
        params["extra_headers"] = _OPENROUTER_HEADERS
        params["extra_body"] = _openrouter_extra_body(model)
        attempts = _OPENROUTER_RETRIES + 1
    else:
        attempts = 1

    if _GEMINI_REASONING_EFFORT and _uses_gemini_thinking(model):
        params["reasoning_effort"] = _GEMINI_REASONING_EFFORT

    last_err: BaseException | None = None
    for attempt in range(attempts):
        try:
            resp = litellm.completion(**params)
            ch = resp.choices[0].message
            text = (ch.content or "").strip()
            if not text:
                # Empty completion is the same kind of failure as a stream cut
                # for our purposes — retry it.
                raise RuntimeError("empty completion from model")
            return text
        except Exception as exc:
            last_err = exc
            if attempt >= attempts - 1 or not _is_transient(exc):
                break
            backoff = min(
                _OPENROUTER_RETRY_BASE_S * (2 ** attempt), _OPENROUTER_RETRY_MAX_S
            )
            logger.warning(
                "LiteLLM transient failure model=%s (attempt %d/%d, sleep=%.1fs): %s",
                model,
                attempt + 1,
                attempts,
                backoff,
                _fmt_exc(exc),
            )
            time.sleep(backoff)

    assert last_err is not None
    for fallback_model in _direct_fallback_models(model):
        fb_params = dict(params)
        fb_params["model"] = fallback_model
        fb_params.pop("extra_headers", None)
        fb_params.pop("extra_body", None)
        if _GEMINI_REASONING_EFFORT and _uses_gemini_thinking(fallback_model):
            fb_params["reasoning_effort"] = _GEMINI_REASONING_EFFORT
        try:
            prepare_litellm_env(fallback_model)
            logger.warning(
                "LiteLLM primary model=%s failed; trying fallback model=%s",
                model,
                fallback_model,
            )
            resp = litellm.completion(**fb_params)
            ch = resp.choices[0].message
            text = (ch.content or "").strip()
            if text:
                return text
            last_err = RuntimeError(f"empty completion from fallback model {fallback_model}")
        except Exception as exc:
            last_err = exc
            logger.warning(
                "LiteLLM fallback failure model=%s: %s",
                fallback_model,
                _fmt_exc(exc),
            )
    raise last_err


def parse_json_object(text: str) -> Dict[str, Any]:
    """Parse first JSON object from model output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return json.loads(m.group(0))
    raise json.JSONDecodeError("No JSON object", text, 0)
