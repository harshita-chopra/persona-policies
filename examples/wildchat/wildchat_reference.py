"""Build a ppol human reference from Salesforce/RealUserSim (WildChat-derived).

RealUserSim (https://huggingface.co/datasets/Salesforce/RealUserSim) turns real
WildChat user<->assistant conversations into a user-simulation benchmark. Each
row's ``original_messages`` is the real conversation; its ``role=="user"`` turns
are genuine human messages to an AI assistant. ppol's behavioral fingerprint
reads only ``role=="user"`` turns, so we group each conversation's user turns
into one dialog and hand them to ``compute_human_reference``.

The reference is split disjointly by ``user_ip`` (the anonymized real user), so
no human appears in both the train reference (discriminator + evolution
coverage) and the held-out test reference (eval) — the WildChat analog
of ColBench's repo-disjoint / tau2's annotator-disjoint split. Each dialog keeps
its ``user_ip`` and ``domain`` for downstream per-task Coverage/Dice.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

_HF_REPO = "Salesforce/RealUserSim"

# RealUserSim ships as 6 per-domain JSONL test sets (all fields inline). We read
# them directly rather than via `datasets.load_dataset`, whose schema inference
# chokes on the nested `user_profile` Json/null fields.
_TEST_SETS = ("business_finance", "ecommerce", "medical_health",
              "mixed_domain", "technology_it", "travel_hospitality")

# XML/HTML/CLI-style markup — real human prose has none; these mark pasted
# artifacts / tool output that would inflate the reference bandwidth.
_TAG = re.compile(r"<[A-Za-z/]")


def load_realusersim_rows() -> List[dict]:
    """All RealUserSim rows, read from the per-domain JSONL test sets on the Hub."""
    import json
    from huggingface_hub import hf_hub_download
    rows: List[dict] = []
    for name in _TEST_SETS:
        p = hf_hub_download(_HF_REPO, f"evaluation/test_sets/{name}.jsonl", repo_type="dataset")
        rows += [json.loads(line) for line in open(p) if line.strip()]
    return rows


def canonical_user_split(rows: List[dict], train_frac: float = 0.75) -> dict:
    """Deterministic ``user_ip -> 'train'|'test'`` map over ALL rows, disjoint at
    the user level. Greedy-packs users (most rows first, ties broken by user_ip —
    no shuffle) into whichever side is furthest below its row quota, so the split
    is fully order-independent and IDENTICAL wherever it's computed (reference
    builder AND runner). This is the single source of truth for the split."""
    from collections import Counter
    counts = Counter(str(r["user_ip"]) for r in rows)
    users = sorted(counts, key=lambda u: (-counts[u], u))   # deterministic order
    total = sum(counts.values())
    quota = {"train": train_frac * total, "test": (1.0 - train_frac) * total}
    filled = {"train": 0.0, "test": 0.0}
    side: dict = {}
    for u in users:
        s = max(quota, key=lambda k: quota[k] - filled[k])
        side[u] = s
        filled[s] += counts[u]
    return side


def _is_prose(content: str, max_line_chars: int, max_newlines: int) -> bool:
    """True for a natural-language user turn; False for a pasted code/artifact
    dump or markup (a fenced block, too many newlines, or an over-long line)."""
    if not content or not content.strip():
        return False
    if "```" in content:
        return False
    if _TAG.search(content):
        return False
    lines = content.split("\n")
    if len(lines) - 1 >= max_newlines:
        return False
    if max((len(x) for x in lines), default=0) > max_line_chars:
        return False
    return True


def build_wildchat_dialogs(
    *,
    max_conversations: Optional[int] = None,
    min_user_turns: int = 2,
    max_line_chars: int = 400,
    max_newlines: int = 15,
) -> List[dict]:
    """Return ppol-format dialogs: ``[{"conversation": [{role:"user", content}...],
    "metadata": {...}}]`` — one dialog per RealUserSim conversation.

    A dialog is kept when it has >= ``min_user_turns`` prose user turns.
    ``max_conversations`` is an optional debug cap (None = use everything)."""
    formed: List[dict] = []
    for row in load_realusersim_rows():
        turns = [
            {"role": "user", "content": str(m["content"]).strip()}
            for m in (row["original_messages"] or [])
            if m.get("role") == "user"
            and _is_prose(str(m.get("content", "")), max_line_chars, max_newlines)
        ]
        if len(turns) < min_user_turns:
            continue
        formed.append({
            "conversation": turns,
            "metadata": {
                "user_ip": str(row["user_ip"]),
                "conversation_hash": str(row.get("conversation_hash", "")),
                "domain": str(row.get("domain", "")),
                "task_type": str(row.get("task_type", "")),
                "source": "real_user_sim",
            },
        })
        if max_conversations and len(formed) >= max_conversations:
            break
    return formed


def _test_path(output_path: str) -> str:
    """`.../human_fingerprints.json` -> `.../human_fingerprints.test.json`."""
    p = Path(output_path)
    return str(p.with_name(p.stem + ".test" + p.suffix))


def compute_wildchat_human_reference(
    output_path: str,
    *,
    max_conversations: Optional[int] = None,
    min_user_turns: int = 2,
    max_line_chars: int = 400,
    max_newlines: int = 15,
    train_frac: float = 0.75,
    split_seed: int = 0,
) -> str:
    """Build dialogs + write TWO ppol human references, split disjointly by
    ``user_ip``: ``output_path`` (train users — discriminator + evolution
    coverage) and ``<output_path>.test.json`` (held-out users — eval).
    Returns the train path. Mirrors colbench/swe_chat_reference.py."""
    from ppol.pipeline import compute_human_reference

    dialogs = build_wildchat_dialogs(
        max_conversations=max_conversations, min_user_turns=min_user_turns,
        max_line_chars=max_line_chars, max_newlines=max_newlines,
    )
    if not dialogs:
        raise RuntimeError("No RealUserSim dialogs survived filtering — loosen the filters.")
    n_users = len({d["metadata"]["user_ip"] for d in dialogs})
    print(f"Built {len(dialogs)} RealUserSim human dialogs across {n_users} users "
          f"({sum(len(d['conversation']) for d in dialogs)} user turns)")

    side = canonical_user_split(load_realusersim_rows(), train_frac)
    train = [d for d in dialogs if side[d["metadata"]["user_ip"]] == "train"]
    test = [d for d in dialogs if side[d["metadata"]["user_ip"]] == "test"]
    if not train or not test:
        raise RuntimeError(
            f"User split left an empty side (train={len(train)}, test={len(test)}); "
            "adjust --train-frac.")
    n_train_u = len({d["metadata"]["user_ip"] for d in train})
    n_test_u = len({d["metadata"]["user_ip"] for d in test})
    print(f"User split (seed {split_seed}, train_frac {train_frac}): "
          f"train {len(train)} dialogs / {n_train_u} users | "
          f"held-out {len(test)} dialogs / {n_test_u} users (disjoint users)")

    compute_human_reference(train, output_path, domain="wildchat", source="real_user_sim")
    compute_human_reference(test, _test_path(output_path),
                            domain="wildchat", source="real_user_sim")
    return output_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build ppol human reference from RealUserSim")
    ap.add_argument("output_path")
    ap.add_argument("--max-conversations", type=int, default=None,
                    help="optional debug cap on #dialogs (default: use all eligible)")
    ap.add_argument("--min-user-turns", type=int, default=2)
    ap.add_argument("--max-line-chars", type=int, default=400)
    ap.add_argument("--max-newlines", type=int, default=15)
    ap.add_argument("--train-frac", type=float, default=0.75,
                    help="fraction of DIALOGS in the train reference (user-disjoint); rest held out")
    ap.add_argument("--split-seed", type=int, default=0)
    a = ap.parse_args()
    compute_wildchat_human_reference(
        a.output_path, max_conversations=a.max_conversations,
        min_user_turns=a.min_user_turns,
        max_line_chars=a.max_line_chars, max_newlines=a.max_newlines,
        train_frac=a.train_frac, split_seed=a.split_seed,
    )
