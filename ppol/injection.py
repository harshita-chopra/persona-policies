"""Canonical persona-injection helpers — used by every EpisodeRunner.
"""

from __future__ import annotations

PERSONA_INJECTION_TEMPLATE = """
## YOUR BEHAVIORAL STYLE FOR THIS CONVERSATION
In addition to your role and task above, you must embody the following behavioral persona throughout this conversation. This persona affects HOW you communicate, NOT WHAT you want or need. Your goal, preferences, and private information remain exactly as described above.

---
{persona_policy_text}
--- 

Apply the behavioral style consistently across all turns: follow the assigned communication pattern, reveal information only as appropriate, and express emotional traits naturally over the course of the interaction. 
Your goal and private information DO NOT change. You still want to accomplish the same task. Do NOT break character; never mention having a behavioral instruction, and aim for a natural human interaction.
""".strip()


def inject_persona_into_system_prompt(
    original_system_prompt: str,
    persona_policy_text: str,
    injection_template: str = PERSONA_INJECTION_TEMPLATE,
) -> str:
    """Append the persona block to a base user-simulator system prompt."""
    injected_block = injection_template.format(
        persona_policy_text=persona_policy_text.strip()
    )
    return original_system_prompt.strip() + "\n\n" + injected_block.strip()
