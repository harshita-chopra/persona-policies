"""
Load τ²-bench human dialogues from ``tau_bench_human.json`` with domain filtering.

Default domain (``PersonaPoliciesConfig.taubench_domain``, usually ``retail``) matches
the τ² simulator domain used for evolution and baseline collection.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def _is_setup_turn(turn: dict) -> bool:
    """UI scaffolding: task line + canvas instruction — not real dialogue."""
    content = (turn.get("content") or "").strip()
    if content.startswith("\\tau ") or content.startswith(r"\tau "):
        return True
    if "<|canvas|>" in content:
        return True
    return False


def load_domain_dialogues(path: str, domain: str) -> List[Dict[str, Any]]:
    """Load ``tau_bench_human.json``; keep keys ``{domain}_*``; strip setup turns."""
    with open(path) as f:
        data = json.load(f)

    prefix = domain + "_"
    dialogues: List[Dict[str, Any]] = []
    for key, entry in data.items():
        if not key.startswith(prefix):
            continue
        conv = entry.get("conversation", [])
        if not conv:
            continue

        clean: List[dict] = []
        past_setup = False
        for turn in conv:
            if not past_setup and _is_setup_turn(turn):
                continue
            past_setup = True
            clean.append(turn)

        if len(clean) < 2:
            continue

        dialogues.append({"conversation": clean, "instance_id": key})

    return dialogues
