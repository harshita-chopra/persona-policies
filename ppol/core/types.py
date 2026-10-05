from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


# Standard conversation format: list of {"role": "user"|"assistant", "content": str}
Conversation = List[Dict[str, str]]


@dataclass
class Task:
    """A single task for the user simulator to accomplish."""

    task_id: str
    description: str          # what the simulated user is trying to do
    context: str = ""         # optional background / domain info
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """Result of a single episode run."""

    task_id: str
    trajectory: Conversation  # [{role, content}, ...]
    success: bool = False
    reward: float = 0.0
    n_turns: int = 0
    persona_policy: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.n_turns:
            self.n_turns = len(self.trajectory)
