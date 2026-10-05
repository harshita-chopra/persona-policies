"""inject_persona_into_system_prompt — pure string helper, no LLM/network."""

from __future__ import annotations

from ppol import inject_persona_into_system_prompt


BASE = "You are a customer. Goal: return a defective laptop."
PERSONA = "Be terse. Reveal one symptom at a time."


def test_persona_appears_in_output():
    out = inject_persona_into_system_prompt(BASE, PERSONA)
    assert PERSONA in out


def test_original_prompt_preserved():
    out = inject_persona_into_system_prompt(BASE, PERSONA)
    assert BASE in out


def test_empty_persona_is_a_noop_or_close():
    out = inject_persona_into_system_prompt(BASE, "")
    # Empty persona should not corrupt the base prompt.
    assert BASE in out
