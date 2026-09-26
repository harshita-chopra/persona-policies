"""
This is evolution/initial_generator.py PROGRAM — Source code of function generate_personas_detailed(c, D, N):

  c — Task context: user scenario (base persona + given instructions).
  D — DIVERSITY_AXES: canonical, evolvable list (behavior name, definition, presence on/off text).
  N — Number of personas to generate.
"""

from typing import Any, Dict, List
from persona_policies.evolution._generator_utils import generate_population, expand_personas_parallel

# List of common behaviors observed in real humans.
# Update, add or remove behaviors to generate more diverse and natural personas.

DIVERSITY_AXES: List[Dict[str, Any]] = [
    {
        "behavior": "terse",
        "definition": "Sparing in the use of words; concise; pithy; often suggests an abruptness that might feel unfriendly or blunt.",
        "presence": {
            "true": "Uses terse language, short sentences, and minimal punctuation, often makes grammatical errors.",
            "false": "Uses verbose language, long sentences, and excessive punctuation. Unnecessary words, phrases, or emojis.",
        },
    },
    {
        "behavior": "skeptical",
        "definition": "Treats assistant statements as unreliable until checked. Seeks confirmation, rationale, or evidence before assenting to recommendations or consequential actions.",
        "presence": {
            "true": "Challenges material claims; ask for sources and verification before each step.",
            "false": "Follows guidance without insisting on proof or cross-examination.",
        },
    },
    {
        "behavior": "frustrated",
        "definition": "A state of annoyance or dissatisfaction arising from unresolved issues or unmet expectations.",
        "presence": {
            "true": "Accusatory language, aggressive tone, no politeness; blunt, repetitive, or frustrated commands in an attempt to correct the agent's incompetence.",
            "false": "Neutral, and tries to be cooperative, by using a gentle tone to express frustration.",
        },
    },
    {
        "behavior": "ambiguous",
        "definition": "Tends to give vague, partial, or noncommittal responses instead of fully clear information.",
        "presence": {
            "true": "Frequently withholds details, trails off, or gives answers that leave things unclear or open to interpretation; needs to be prompted to provide more information.",
            "false": "Always provides direct and complete information with no room for doubt or confusion, but only when asked.",
        },
   
    },
]

# Stage 1: Population generation: jointly generate N high-level persona descriptions with behavior axis placements.
# Update this prompt to improve persona quality.

POPULATION_SYSTEM = """Your task is to create diverse, psychologically coherent human personas that will interact with AI agents via text."""

POPULATION_PROMPT = """We need {N} distinct user personas for given task scenario. 

## Behavioral Dimensions (D)
These are the axes along which personas can vary. For each persona, set axis_placement to a boolean per axis: \
``true`` means the behavior is active for that persona, ``false`` means it is not. 

{axes_description}

## Task context c (Base Persona Scenario)

{task_context}

## Requirements
- Generate exactly {N} personas that are plausible humans in this situation.
- Each persona must be psychologically coherent; if two behaviors would clash if both were on, set at most one to ``true``.
- Maximize DIVERSITY across the {N} personas. They should cover different regions of the behavioral space (D), not cluster around the same profile.
- Each persona needs a short "who they are" description (2-3 sentences) that makes the axis placement feel natural and grounded in a real person's life situation — describe the PERSON, not the configuration.

Respond with ONLY valid JSON: one array of exactly {N} objects. Each axis_placement must list every behavior name from D as a key (true/false).
[
  {{
    "persona_id": "short_snake_case_name",
    "description": "2-3 sentence description of who this person is",
    "axis_placement": {{
      "<behavior_name>": true,
      "<behavior_name>": false,
      ...one entry per behavior name listed in D above...
    }},
    "reasoning": "one sentence on why these placements work together for this person"
  }},
  ...
]"""


# Stage 2: Roleplay expansion: expand each population member into full roleplay instructions for a task context.
# Update this prompt to improve persona quality.

ROLEPLAY_SYSTEM = """You write detailed roleplay instructions that steers HOW a simulated user plays a task, on top of the given scenario. The persona must feel like a real human, not a script."""

ROLEPLAY_PROMPT = """Expand the behavior profile below into concrete roleplay instructions. The simulated user already receives the "Task Context"; your output is added alongside it to steer demeanor and interaction style, without replacing or contradicting the scenario’s goals and facts.
Note that the agent-user communication is via text messaging/chat interface.

## Task Context (Base Persona Scenario)
{task_context}

## Behavior profile to superimpose
Name: {persona_id}
Description: {description}

Active behavioral traits:
{active_traits}

## Instructions
Write a detailed roleplay instruction (150-250 words) that tells the user simulator HOW to play this persona in this specific task. The instruction should:

1. GROUND the persona in this specific Task Context and behavior profile.
2. Specify concrete communication patterns that should be followed: linguistics, vocabulary, emotional markers, how they respond to agent requests.
3. Preserve all goals and facts from the Task Context; only vary *how* the person pursues them.
4. Do NOT break the character — no mention of "simulation", "benchmark", or "AI".

Respond with ONLY the roleplay instruction text:"""


def generate_personas_detailed(c: str, axes: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """G(c, D, N) — the single public entrypoint. expanded_instruction of each persona is fed to the user simulator.
    """
    population = generate_population(
        system_prompt=POPULATION_SYSTEM,
        prompt_template=POPULATION_PROMPT,
        task_context=c,
        axes=axes,
        n=n,
    )
    
    expanded_instructions = expand_personas_parallel(
        system_prompt=ROLEPLAY_SYSTEM,
        prompt_template=ROLEPLAY_PROMPT,
        archetypes=[member for member in population if isinstance(member, dict)],
        task_context=c,
        axes=axes,
    )

    personas: List[Dict[str, Any]] = []
    for i, member in enumerate(population):
        if not isinstance(member, dict):
            continue
        expanded_instruction = expanded_instructions[i] if i < len(expanded_instructions) else ""
        personas.append(
            {
                "persona_id": member.get("persona_id"),
                "description": member.get("description"),
                "axis_placement": dict(member.get("axis_placement") or {}),
                "reasoning": member.get("reasoning"),
                "expanded_instruction": expanded_instruction,
            }
        )
    return personas
