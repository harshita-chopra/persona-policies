"""DataLoader: JSON and plain-text formats."""

from __future__ import annotations

import json
from pathlib import Path

from ppol import DataLoader, Task


def test_load_tasks_json(tmp_path: Path):
    src = tmp_path / "tasks.json"
    src.write_text(
        json.dumps(
            [
                {"task_id": "t1", "description": "return laptop", "context": "retail"},
                {"task_id": "t2", "description": "book flight", "context": "airline"},
            ]
        )
    )
    tasks = DataLoader.load_tasks(str(src))
    assert len(tasks) == 2
    assert isinstance(tasks[0], Task)
    assert tasks[0].task_id == "t1"


def test_load_dialogs_json(tmp_path: Path):
    src = tmp_path / "d.json"
    src.write_text(
        json.dumps(
            [
                {
                    "conversation": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "hello"},
                    ],
                    "metadata": {},
                }
            ]
        )
    )
    dialogs = DataLoader.load_dialogs(str(src))
    assert len(dialogs) == 1
    conv = dialogs[0]["conversation"] if isinstance(dialogs[0], dict) else dialogs[0]
    assert len(conv) == 2
