"""Build a ppol human reference from SALT-NLP/SWE-chat, register-matched to ColBench.

SWE-chat (https://huggingface.co/datasets/SALT-NLP/SWE-chat) is a corpus of real
developer <-> coding-agent sessions; its ``role=="user"`` turns are genuine human
messages. 

Sessions are grouped by ``repo_id`` (the dataset-native analog of a ColBench
task); ``min_sessions_per_repo`` keeps only repos with enough sessions to form
a task cell, and each dialog carries its ``repo_id`` for per-task Coverage/Dice.
The reference is split repo-disjointly into train and held-out test files.

Only ``conversations.parquet`` (~1.3 GB) is auto-downloaded on first use.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

# XML/HTML/CLI-style tag (e.g. <command-message>, <task-notification>). Real human
# prose has no markup tags; these are tool/command/system artifacts.
_TAG = re.compile(r"<[A-Za-z/]")

_HF_REPO = "SALT-NLP/SWE-chat"

# ColBench backend = collaboratively authoring one Python function from a spec.
_CODE_INTENTS = frozenset(
    {"create new code", "refactor", "debug", "test", "understand"}
)

_COLUMNS = [
    "session_id", "repo_id", "conversation_turn_number", "role",
    "is_conversational", "language", "prompt_intent", "content", "word_count",
]


def _is_prose(content: str, max_line_chars: int, max_newlines: int) -> bool:
    """True for a natural-language turn; False for a pasted code/log artifact or
    tool/command markup."""
    if not content or not content.strip():
        return False
    if "```" in content:
        return False
    if _TAG.search(content):          # CLI/command/system tags, not human prose
        return False
    lines = content.split("\n")
    if len(lines) - 1 >= max_newlines:
        return False
    if max((len(x) for x in lines), default=0) > max_line_chars:
        return False
    return True


def build_swe_chat_dialogs(
    *,
    max_sessions: Optional[int] = None,
    min_user_turns: int = 2,
    min_sessions_per_repo: int = 3,
    max_line_chars: int = 400,
    max_newlines: int = 15,
    parquet_path: Optional[str] = None,
) -> List[dict]:
    """Return ppol-format dialogs: [{"conversation": [{role:"user", content}...],
    "metadata": {...}}]. One dialog per SWE-chat session, grouped into per-task
    cells by repo_id.
    """
    import pandas as pd
    from collections import Counter

    if parquet_path is None:
        from huggingface_hub import hf_hub_download
        parquet_path = hf_hub_download(
            repo_id=_HF_REPO, filename="conversations.parquet", repo_type="dataset"
        )

    df = pd.read_parquet(parquet_path, columns=_COLUMNS)
    # Register match: human natural-language coding turns, no length cut.
    df = df[
        (df["role"] == "user")
        & (df["is_conversational"] == True)  # noqa: E712
        & (df["language"] == "ENGLISH")
        & (df["prompt_intent"].isin(_CODE_INTENTS))
    ]

    # Form one prose dialog per session (>=min_user_turns prose turns).
    formed: List[tuple] = []  # (repo_id, session_id, turns)
    for (repo_id, session_id), grp in df.groupby(
        ["repo_id", "session_id"], sort=False
    ):
        grp = grp.sort_values("conversation_turn_number", na_position="last")
        turns = [
            {"role": "user", "content": str(c).strip()}
            for c in grp["content"]
            if _is_prose(str(c), max_line_chars, max_newlines)
        ]
        if len(turns) >= min_user_turns:
            formed.append((str(repo_id), str(session_id), turns))

    per_repo = Counter(r for r, _, _ in formed)
    formed = [x for x in formed if per_repo[x[0]] >= min_sessions_per_repo]
    if max_sessions:
        formed = formed[:max_sessions]

    return [
        {"conversation": turns,
         "metadata": {"session_id": sid, "repo_id": rid, "source": "swe_chat"}}
        for rid, sid, turns in formed
    ]


def _split_by_repo(
    dialogs: List[dict], train_frac: float, split_seed: int
) -> Tuple[List[dict], List[dict]]:
    """Partition dialogs into (train, held-out), disjoint at the repo_id level
    and sized so each side holds ~train_frac / 1-train_frac of the dialogs.
    """
    import random

    by_repo: dict = {}
    for d in dialogs:
        by_repo.setdefault(d["metadata"]["repo_id"], []).append(d)

    repos = list(by_repo)
    random.Random(split_seed).shuffle(repos)              # seed-controlled tie-break
    repos.sort(key=lambda r: len(by_repo[r]), reverse=True)  # largest first

    total = len(dialogs)
    quota = {"train": train_frac * total, "test": (1.0 - train_frac) * total}
    filled = {"train": 0.0, "test": 0.0}
    side = {}
    for r in repos:
        s = max(quota, key=lambda k: quota[k] - filled[k])  # furthest below quota
        side[r] = s
        filled[s] += len(by_repo[r])

    train = [d for r in repos if side[r] == "train" for d in by_repo[r]]
    test = [d for r in repos if side[r] == "test" for d in by_repo[r]]
    return train, test


def _test_path(output_path: str) -> str:
    """`.../human_fingerprints.json` -> `.../human_fingerprints.test.json`."""
    p = Path(output_path)
    return str(p.with_name(p.stem + ".test" + p.suffix))


def compute_swe_chat_human_reference(
    output_path: str,
    *,
    max_sessions: Optional[int] = None,
    min_user_turns: int = 2,
    min_sessions_per_repo: int = 3,
    max_line_chars: int = 400,
    max_newlines: int = 15,
    train_frac: float = 0.75,
    split_seed: int = 0,
    parquet_path: Optional[str] = None,
) -> str:

    from ppol.pipeline import compute_human_reference

    dialogs = build_swe_chat_dialogs(
        max_sessions=max_sessions, min_user_turns=min_user_turns,
        min_sessions_per_repo=min_sessions_per_repo,
        max_line_chars=max_line_chars, max_newlines=max_newlines,
        parquet_path=parquet_path,
    )
    if not dialogs:
        raise RuntimeError("No SWE-chat dialogs survived filtering — loosen the filters.")
    n_repos = len({d["metadata"]["repo_id"] for d in dialogs})
    print(f"Built {len(dialogs)} SWE-chat human dialogs across {n_repos} repos "
          f"({sum(len(d['conversation']) for d in dialogs)} user turns)")

    train, test = _split_by_repo(dialogs, train_frac, split_seed)
    if not train or not test:
        raise RuntimeError(
            f"Repo split left an empty side (train={len(train)}, test={len(test)}); "
            "lower --min-sessions-per-repo or adjust --train-frac.")
    n_train_repos = len({d["metadata"]["repo_id"] for d in train})
    n_test_repos = len({d["metadata"]["repo_id"] for d in test})
    print(f"Repo split (seed {split_seed}, train_frac {train_frac}): "
          f"train {len(train)} dialogs / {n_train_repos} repos | "
          f"held-out {len(test)} dialogs / {n_test_repos} repos (disjoint repos)")

    compute_human_reference(train, output_path, domain="colbench_code", source="swe_chat")
    compute_human_reference(test, _test_path(output_path),
                            domain="colbench_code", source="swe_chat")
    return output_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build ppol human reference from SWE-chat")
    ap.add_argument("output_path")
    ap.add_argument("--max-sessions", type=int, default=None,
                    help="optional debug cap on #dialogs (default: use all eligible)")
    ap.add_argument("--min-user-turns", type=int, default=2)
    ap.add_argument("--min-sessions-per-repo", type=int, default=3)
    ap.add_argument("--max-line-chars", type=int, default=400)
    ap.add_argument("--max-newlines", type=int, default=15)
    ap.add_argument("--train-frac", type=float, default=0.75,
                    help="fraction of DIALOGS in the train reference (repo-disjoint); rest held out")
    ap.add_argument("--split-seed", type=int, default=0)
    a = ap.parse_args()
    compute_swe_chat_human_reference(
        a.output_path, max_sessions=a.max_sessions,
        min_user_turns=a.min_user_turns,
        min_sessions_per_repo=a.min_sessions_per_repo,
        max_line_chars=a.max_line_chars, max_newlines=a.max_newlines,
        train_frac=a.train_frac, split_seed=a.split_seed,
    )
