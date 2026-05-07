"""
Stable plumbing for the evolved persona generator.

This module is NOT mutated by OpenEvolve. It holds infrastructure the evolved `initial_generator.py` relies on but should never need to change.
"""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _llm_call(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.7,
    max_tokens: int = 800,
) -> str:
    """Single entrypoint to the Bedrock LiteLLM client (kept out of the evolved file)."""
    from persona_policies.llm_bedrock import completion_text
    return completion_text(
        model, messages, temperature=temperature, max_tokens=max_tokens,
    )


_cached_llm_model: Optional[str] = None


def _llm_model_id() -> str:
    """Default persona-generation model id from ``PersonaPoliciesConfig.llm_model`` (cached)."""
    global _cached_llm_model
    if _cached_llm_model is None:
        from persona_policies.config import PersonaPoliciesConfig
        _cached_llm_model = PersonaPoliciesConfig().llm_model
    return _cached_llm_model


def _chat(
    system: str,
    user: str,
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    """Standard two-message chat shim used by population and roleplay steps."""
    return _llm_call(
        _llm_model_id(),
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _normalize_placement_bool(raw: Any) -> bool:
    """Normalize axis_placement values to bool."""
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("true", "1", "yes", "on")


def _playbook(axis: Dict[str, Any], on: bool) -> str:
    presence = axis.get("presence") or {}
    return str(presence.get("true" if on else "false") or "").strip()


def _format_axes_description(axes: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for axis in axes:
        behavior = str(axis.get("behavior") or "").strip() or "?"
        definition = str(axis.get("definition") or "").strip()
        lines.append(behavior)
        if definition:
            lines.append(f"  - definition: {definition}")
        lines.append(f"  - presence true (on): {_playbook(axis, True)}")
        lines.append(f"  - presence false (off): {_playbook(axis, False)}")
    return "\n".join(lines)


def _format_active_traits(
    axis_placement: Dict[str, Any],
    axes: List[Dict[str, Any]],
) -> str:
    axes_by_behavior = {
        behavior: axis
        for axis in axes
        if (behavior := str(axis.get("behavior") or "").strip())
    }
    lines: List[str] = []
    for key, raw in axis_placement.items():
        behavior_key = str(key).strip()
        if not _normalize_placement_bool(raw):
            continue
        axis = axes_by_behavior.get(behavior_key, {})
        behavior = str(axis.get("behavior") or "").strip() or behavior_key
        definition = str(axis.get("definition") or "").strip()
        playbook = _playbook(axis, True)
        if definition:
            lines.append(
                f"- {behavior} (on)\n  Definition: {definition}\n  Playbook: {playbook}"
            )
        elif playbook:
            lines.append(f"- {behavior} (on): {playbook}")
    if not lines:
        lines = ["- (neutral baseline — no strongly active behavioral traits)"]
    return "\n".join(lines)


def _parse_stage1_json_array(text: str) -> List[Any] | None:
    """Parse a JSON array from imperfect LLM output, tolerating fences and chatter."""
    stripped = text.strip("\ufeff \n\r\t")
    source = re.sub(r"^\s*```[a-zA-Z0-9]*\s*|\s*```\s*$", "", stripped)
    decoder = json.JSONDecoder()
    start = -1
    out: List[Any] = []
    i = 0
    while (start := source.find("[", start + 1)) >= 0:
        try:
            value, _ = decoder.raw_decode(source, start)
            if isinstance(value, list) and any(isinstance(x, dict) for x in value):
                return [x for x in value if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass
    while (j := source.find("{", i)) >= 0:
        try:
            value, i = decoder.raw_decode(source, j)
            if isinstance(value, dict):
                out.append(value)
        except json.JSONDecodeError:
            i = j + 1
    return out or None


def generate_population(
    *,
    system_prompt: str,
    prompt_template: str,
    task_context: str,
    axes: List[Dict[str, Any]],
    n: int,
    temperature: float = 0.8,
    max_tokens: int = 8000,
) -> List[Dict[str, Any]]:
    """Run the population step and return validated per-persona dicts."""
    prompt = prompt_template.format(
        N=n,
        axes_description=_format_axes_description(axes),
        task_context=(task_context or "").strip(),
    )
    raw = _chat(system_prompt, prompt, temperature=temperature, max_tokens=max_tokens).strip()
    archetypes = _parse_stage1_json_array(raw) or []

    axis_keys = {b for axis in axes if (b := str(axis.get("behavior") or "").strip())}
    valid: List[Dict[str, Any]] = []
    for archetype in archetypes[:n]:
        if not isinstance(archetype, dict):
            continue
        raw_placement = archetype.get("axis_placement")
        placement = raw_placement if isinstance(raw_placement, dict) else {}
        archetype["axis_placement"] = {
            key: _normalize_placement_bool(placement.get(key))
            for key in axis_keys
        }
        valid.append(archetype)

    if len(valid) < n:
        raise RuntimeError(
            f"generate_population: LLM returned only {len(valid)}/{n} valid "
            f"archetypes after retries (transient errors already retried "
            f"inside _chat). Refusing to pad with template fallbacks. "
            f"output_excerpt={raw[:400]!r}"
        )
    return valid[:n]


def expand_stage2_archetype(
    *,
    system_prompt: str,
    prompt_template: str,
    archetype: Dict[str, Any],
    task_context: str,
    axes: List[Dict[str, Any]],
    temperature: float = 0.7,
    max_tokens: int = 800,
) -> str:
    """Run the roleplay expansion for one population member and return the roleplay instruction."""
    prompt = prompt_template.format(
        persona_id=archetype.get("persona_id", "unknown"),
        description=archetype.get("description", "A real human."),
        active_traits=_format_active_traits(archetype.get("axis_placement", {}), axes),
        task_context=task_context,
    )
    return _chat(
        system_prompt,
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    ).strip()


def expand_personas_parallel(
    *,
    system_prompt: str,
    prompt_template: str,
    archetypes: List[Dict[str, Any]],
    task_context: str,
    axes: List[Dict[str, Any]],
    temperature: float = 0.7,
    max_tokens: int = 800,
    max_workers: Optional[int] = None,
) -> List[str]:
    """Expand population members with the single-person prompt, parallelized within one task."""
    if not archetypes:
        return []
    n_workers = max(1, min(len(archetypes), int(max_workers or len(archetypes))))
    if n_workers == 1:
        return [
            expand_stage2_archetype(
                system_prompt=system_prompt,
                prompt_template=prompt_template,
                archetype=archetype,
                task_context=task_context,
                axes=axes,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for archetype in archetypes
        ]

    expanded = [""] * len(archetypes)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                expand_stage2_archetype,
                system_prompt=system_prompt,
                prompt_template=prompt_template,
                archetype=archetype,
                task_context=task_context,
                axes=axes,
                temperature=temperature,
                max_tokens=max_tokens,
            ): i
            for i, archetype in enumerate(archetypes)
        }
        for fut in as_completed(futures):
            expanded[futures[fut]] = fut.result()
    return expanded
