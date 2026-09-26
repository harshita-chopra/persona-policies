"""
Direct LLM persona baseline.

This intentionally skips the evolved Stage-1/Stage-2 axis pipeline. Given one
task context, it asks the configured persona-generation LLM for N full persona
policy blocks directly.
"""

from __future__ import annotations

from typing import Any, Dict, List

from persona_policies.evolution._generator_utils import _chat, _parse_stage1_json_array


DIVERSITY_AXES: List[Dict[str, Any]] = []

DIRECT_SYSTEM = """You create realistic task-conditioned user personas for a customer-service chat simulator."""

DIRECT_PROMPT = """Generate {N} distinct personas: roleplay instructions for the task context below. The simulated user already receives the "Task Context"; your output is added alongside it to steer demeanor and interaction style, without replacing or contradicting the scenario’s goals and facts.
Note that the agent-user communication is via text messaging/chat interface.

## Task Context (Base Persona Scenario)
{task_context}

## Instructions
- Generate plausible real personas of users for this exact scenario.
- Make the set of personas diverse across behaviors like terse, skeptical, frustrated, ambiguous, etc.

- For each persona, write a detailed roleplay instruction (150-250 words) that tells the user simulator HOW to play this persona in this specific task. The instruction should:
  - GROUND the persona in this specific Task Context and behavior profile.
  - Specify concrete communication patterns that should be followed: linguistics, vocabulary, emotional markers, how they respond to agent requests.
  - Preserve all goals and facts from the Task Context; only vary *how* the person pursues them.
  - Do NOT break the character — no mention of "simulation", "benchmark", or "AI".


Return ONLY valid JSON: an array of exactly {N} objects:
[
  {{
    "persona_id": "short_snake_case_name",
    "description": "brief description of the user type",
    "expanded_instruction": "full persona policy text"
  }}
]
"""


def generate_personas_detailed(
    c: str,
    axes: List[Dict[str, Any]],
    n: int,
) -> List[Dict[str, Any]]:
    """Generate N full persona instructions in one direct LLM call."""
    del axes
    prompt = DIRECT_PROMPT.format(N=int(n), task_context=(c or "").strip())
    raw = _chat(
        DIRECT_SYSTEM,
        prompt,
        temperature=0.8,
        max_tokens=max(1800, int(n) * 700),
    ).strip()
    rows = _parse_stage1_json_array(raw) or []
    personas: List[Dict[str, Any]] = []
    for i, row in enumerate(rows[: int(n)]):
        text = str(row.get("expanded_instruction") or "").strip()
        if not text:
            continue
        personas.append(
            {
                "persona_id": row.get("persona_id") or f"direct_persona_{i + 1}",
                "description": row.get("description"),
                "axis_placement": {},
                "reasoning": None,
                "expanded_instruction": text,
            }
        )
    if len(personas) < int(n):
        raise RuntimeError(
            f"direct_llm_personas returned only {len(personas)}/{int(n)} valid personas"
        )
    return personas[: int(n)]
