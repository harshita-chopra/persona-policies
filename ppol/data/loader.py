"""
DataLoader for ppol standard data formats.

dialogs.json format::

    [
        {
            "conversation": [
                {"role": "user", "content": "Hi, I want to return my order."},
                {"role": "assistant", "content": "Sure, what's your order number?"},
                ...
            ],
            "metadata": {}   // optional
        },
        ...
    ]

tasks.json format::

    [
        {
            "task_id": "t001",
            "description": "Return a defective laptop within 30 days of purchase.",
            "context": "Customer service — retail domain."   // optional
            // any extra fields are stored in Task.metadata
        },
        ...
    ]

Plain-text dialogs (.txt) use ``load_txt_dialogs()``. Conversations are
separated by lines containing only ``---``. Turns are prefixed with
``USER:`` or ``ASSISTANT:``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ppol.core.types import Task


class DataLoader:

    @staticmethod
    def load_dialogs(path: str) -> List[dict]:
        """Load dialogs from a JSON file in ppol standard format."""
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"dialogs file must be a JSON array, got {type(data).__name__}")
        for i, item in enumerate(data):
            if not isinstance(item, dict) or "conversation" not in item:
                raise ValueError(
                    f"dialogs[{i}] must be a dict with a 'conversation' key. "
                    "See DataLoader docstring for the expected format."
                )
        return data

    @staticmethod
    def load_tasks(path: str) -> List[Task]:
        """Load tasks from a JSON file in ppol standard format."""
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"tasks file must be a JSON array, got {type(data).__name__}")
        tasks: List[Task] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"tasks[{i}] must be a dict")
            if "description" not in item:
                raise ValueError(f"tasks[{i}] is missing required field 'description'")
            task_id = str(item.get("task_id", i))
            description = str(item["description"])
            context = str(item.get("context", ""))
            metadata = {k: v for k, v in item.items() if k not in ("task_id", "description", "context")}
            tasks.append(Task(task_id=task_id, description=description, context=context, metadata=metadata))
        return tasks

    @staticmethod
    def load_txt_dialogs(path: str, separator: str = "---") -> List[dict]:
        """
        Load dialogs from a plain-text file.

        Conversations are separated by lines containing only ``separator``
        (default: ``---``). Turns are prefixed with ``USER:`` or
        ``ASSISTANT:`` (case-insensitive). Continuation lines (no prefix) are
        appended to the previous turn.

        Example file::

            USER: Hi, I'd like to return my laptop.
            ASSISTANT: Of course! What's wrong with it?
            USER: The screen is cracked.
            ASSISTANT: Got it. I'll process the return now.
            ---
            USER: Where is my order?
            ASSISTANT: Let me check that for you.
        """
        text = Path(path).read_text(encoding="utf-8")
        raw_convs = [block.strip() for block in text.split(separator) if block.strip()]
        dialogs: List[dict] = []
        for raw in raw_convs:
            turns: List[dict] = []
            current_role: str | None = None
            current_lines: List[str] = []

            for line in raw.splitlines():
                stripped = line.strip()
                upper = stripped.upper()
                if upper.startswith("USER:"):
                    if current_role is not None:
                        turns.append({"role": current_role, "content": " ".join(current_lines).strip()})
                    current_role = "user"
                    current_lines = [stripped[len("USER:"):].strip()]
                elif upper.startswith("ASSISTANT:"):
                    if current_role is not None:
                        turns.append({"role": current_role, "content": " ".join(current_lines).strip()})
                    current_role = "assistant"
                    current_lines = [stripped[len("ASSISTANT:"):].strip()]
                elif current_role is not None and stripped:
                    current_lines.append(stripped)

            if current_role is not None:
                turns.append({"role": current_role, "content": " ".join(current_lines).strip()})

            turns = [t for t in turns if t["content"]]
            if turns:
                dialogs.append({"conversation": turns, "metadata": {}})
        return dialogs

    @staticmethod
    def save_dialogs(dialogs: List[dict], path: str) -> None:
        """Write dialogs list to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dialogs, f, indent=2, ensure_ascii=False)

    @staticmethod
    def save_tasks(tasks: List[Task], path: str) -> None:
        """Write tasks list to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = [
            {"task_id": t.task_id, "description": t.description, "context": t.context, **t.metadata}
            for t in tasks
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
