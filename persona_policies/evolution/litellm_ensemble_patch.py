"""
Patch OpenEvolve's LLMEnsemble to call Bedrock, Gemini, OpenRouter, and other LiteLLM routes.

OpenEvolve's default client is ``openai.OpenAI``, which requires ``OPENAI_API_KEY`` and does
not route ``bedrock/...`` or ``openrouter/...`` model ids. Worker processes import ``LLMEnsemble`` before your
evaluator module, so importing this patch from the main script is not enough — install the
``.pth`` hook (see ``install_openevolve_litellm_pth.py``) or run the installer once per venv.

Import this module early (e.g. via .pth) so ``LLMEnsemble.__init__`` is patched before any
OpenAI client is constructed.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Dict, List

import litellm

from openevolve.config import LLMModelConfig
from openevolve.llm import ensemble as ens
from openevolve.llm.base import LLMInterface
from openevolve.llm.openai import OpenAILLM

from persona_policies.llm_bedrock import prepare_litellm_env

logger = logging.getLogger(__name__)

_PATCHED = False

# Minimum retry budget for the OpenEvolve mutator. Upstream default (2) is not
# enough for OpenRouter where transient 429/5xx/timeout is common; we force at
# least ``_MIN_RETRIES`` attempts so a single hiccup doesn't skip an iteration.
_MIN_RETRIES = 3


def _fmt_exc(e: BaseException) -> str:
    """Render an exception the way that's useful in logs when ``str(e)`` is empty."""
    t = type(e).__name__
    msg = str(e).strip()
    parts = [t]
    if msg:
        parts.append(msg)
    status = getattr(e, "status_code", None)
    if status is not None:
        parts.append(f"status={status}")
    resp = getattr(e, "response", None)
    if resp is not None:
        body = getattr(resp, "text", None) or getattr(resp, "content", None)
        if body:
            body_s = body if isinstance(body, str) else body.decode("utf-8", errors="replace")
            parts.append(f"body={body_s[:400]}")
    return ": ".join(parts)


def _use_litellm(model_name: str | None) -> bool:
    if not model_name:
        return False
    n = str(model_name).lower()
    return (
        n.startswith("bedrock/")
        or n.startswith("openrouter/")
        or n.startswith("anthropic/")
        or n.startswith("azure/")
        or n.startswith("vertex_ai/")
        or n.startswith("gemini/")
    )


def _direct_fallback_models(primary_model: str) -> List[str]:
    """Non-OpenRouter fallbacks to try only after the primary model gives up."""
    try:
        from persona_policies.config import PersonaPoliciesConfig

        fallbacks = list(PersonaPoliciesConfig().llm_fallback_models)
    except Exception:
        fallbacks = []
    out: List[str] = []
    for model in fallbacks:
        m = str(model or "").strip()
        if not m or m == primary_model or m.lower().startswith("openrouter/"):
            continue
        if _use_litellm(m):
            out.append(m)
    return out


class LiteLLMOpenEvolve(LLMInterface):
    """OpenEvolve LLMInterface backed by LiteLLM (Bedrock, etc.)."""

    def __init__(self, model_cfg: LLMModelConfig):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = (
            model_cfg.temperature if model_cfg.temperature is not None else 0.7
        )
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens or 4096
        self.timeout = model_cfg.timeout or 120
        cfg_retries = (
            model_cfg.retries if model_cfg.retries is not None else _MIN_RETRIES
        )
        self.retries = max(_MIN_RETRIES, int(cfg_retries))
        self.retry_delay = model_cfg.retry_delay or 2
        # Gemini thinking models: minimal thinking when configured. Gemma on
        # the Gemini provider does not accept reasoning_effort.
        self.reasoning_effort: str | None = getattr(model_cfg, "reasoning_effort", None)

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context(
            system_message=self.system_message or "",
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        formatted = [{"role": "system", "content": system_message}]
        formatted.extend(messages)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Per-call LiteLLM transport timeout (separate from our wait_for below).
            "timeout": float(self.timeout),
        }
        if self.top_p is not None:
            params["top_p"] = kwargs.get("top_p", self.top_p)

        # OpenRouter-only: route to fastest healthy provider, identify the
        # client so OpenRouter's rankings give us preferential scheduling, and
        # enable model-level failover (``models=[primary, *fallbacks]``) from
        # PersonaPoliciesConfig.llm_fallback_models.
        if str(self.model).lower().startswith("openrouter/"):
            from persona_policies.llm_bedrock import (
                _OPENROUTER_HEADERS,
                _openrouter_extra_body,
            )
            params["extra_headers"] = _OPENROUTER_HEADERS
            body = dict(_openrouter_extra_body(self.model))
            # OpenRouter: Gemini 3+ thinkingLevel via ``reasoning.effort``; do not send for Gemma, etc.
            mlow = str(self.model).lower()
            if self.reasoning_effort and "gemini" in mlow:
                body["reasoning"] = {"effort": str(self.reasoning_effort).strip().lower()}
            params["extra_body"] = body
        elif str(self.model).lower().startswith("gemini/"):
            mlow = str(self.model).lower()
            if self.reasoning_effort and "gemini" in mlow and "gemma" not in mlow:
                params["reasoning_effort"] = str(self.reasoning_effort).strip().lower()

        last_err: Exception | None = None
        total_attempts = self.retries + 1
        for attempt in range(total_attempts):
            try:
                prepare_litellm_env(self.model)

                def _call() -> Any:
                    return litellm.completion(**params)

                loop = asyncio.get_event_loop()
                resp = await asyncio.wait_for(
                    loop.run_in_executor(None, _call),
                    timeout=float(self.timeout) + 30.0,
                )
                ch = resp.choices[0].message
                content = (ch.content or "").strip()
                if not content:
                    # Empty completion counts as a retryable failure instead of
                    # bubbling up a silent "generation failed".
                    raise RuntimeError("empty completion from model")
                return content
            except asyncio.TimeoutError as e:
                last_err = e
                reason = f"wait_for timeout after {self.timeout + 30:.0f}s"
            except Exception as e:
                last_err = e
                reason = _fmt_exc(e)

            # Exponential backoff (capped) so we ride out short rate-limit bursts.
            backoff = min(self.retry_delay * (2 ** attempt), 30.0)
            logger.warning(
                "OpenEvolve LLM call failed model=%s (attempt %d/%d, next_delay=%.1fs): %s",
                self.model,
                attempt + 1,
                total_attempts,
                backoff if attempt < self.retries else 0.0,
                reason,
            )
            if attempt < self.retries:
                await asyncio.sleep(backoff)

        for fallback_model in _direct_fallback_models(self.model):
            fb_params = dict(params)
            fb_params["model"] = fallback_model
            fb_params.pop("extra_headers", None)
            fb_params.pop("extra_body", None)
            if str(fallback_model).lower().startswith("gemini/"):
                mlow = str(fallback_model).lower()
                if self.reasoning_effort and "gemini" in mlow and "gemma" not in mlow:
                    fb_params["reasoning_effort"] = str(self.reasoning_effort).strip().lower()
            logger.warning(
                "OpenEvolve LLM primary model=%s failed after retries; trying fallback model=%s",
                self.model,
                fallback_model,
            )
            try:
                prepare_litellm_env(fallback_model)

                def _fallback_call() -> Any:
                    return litellm.completion(**fb_params)

                loop = asyncio.get_event_loop()
                resp = await asyncio.wait_for(
                    loop.run_in_executor(None, _fallback_call),
                    timeout=float(self.timeout) + 30.0,
                )
                ch = resp.choices[0].message
                content = (ch.content or "").strip()
                if content:
                    return content
                last_err = RuntimeError(f"empty completion from fallback model {fallback_model}")
            except Exception as e:
                last_err = e
                logger.warning(
                    "OpenEvolve fallback LLM call failed model=%s: %s",
                    fallback_model,
                    _fmt_exc(e),
                )

        final = _fmt_exc(last_err) if last_err is not None else "unknown error"
        logger.error(
            "OpenEvolve LLM call gave up model=%s after %d attempts: %s",
            self.model,
            total_attempts,
            final,
        )
        raise last_err  # type: ignore[misc]


def _patched_ensemble_init(self, models_cfg: List[LLMModelConfig]):
    active_models_cfg = [m for m in models_cfg if float(getattr(m, "weight", 1.0) or 0.0) > 0.0]
    if not active_models_cfg:
        active_models_cfg = models_cfg
    self.models_cfg = active_models_cfg
    self.models = []
    for model_cfg in active_models_cfg:
        if getattr(model_cfg, "init_client", None):
            self.models.append(model_cfg.init_client(model_cfg))
        elif _use_litellm(getattr(model_cfg, "name", None)):
            self.models.append(LiteLLMOpenEvolve(model_cfg))
            logger.info(
                "Persona Policies: LiteLLM bridge for OpenEvolve model %s",
                model_cfg.name,
            )
        else:
            self.models.append(OpenAILLM(model_cfg))

    self.weights = [model.weight for model in active_models_cfg]
    total = sum(self.weights)
    self.weights = [w / total for w in self.weights]

    self.random_state = random.Random()
    if (
        active_models_cfg
        and hasattr(active_models_cfg[0], "random_seed")
        and active_models_cfg[0].random_seed is not None
    ):
        self.random_state.seed(active_models_cfg[0].random_seed)

    if len(active_models_cfg) > 1 or not hasattr(logger, "_ensemble_logged_pp"):
        logger.info(
            "Initialized LLM ensemble with models: "
            + ", ".join(
                f"{model.name} (weight: {weight:.2f})"
                for model, weight in zip(active_models_cfg, self.weights)
            )
        )
        logger._ensemble_logged_pp = True  # type: ignore[attr-defined]


def apply_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    ens.OpenAILLM = LiteLLMOpenEvolve  # type: ignore[assignment]
    ens.LLMEnsemble.__init__ = _patched_ensemble_init  # type: ignore[assignment]
    _PATCHED = True
    logger.debug("Applied litellm_ensemble_patch for OpenEvolve")


apply_patch()
