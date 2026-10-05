"""τ²-bench human-data reference for ppol (analog of swe_chat_reference.py /
wildchat_reference.py).

Loads the real human dialogues (``tau_bench_human.json`` format) and builds the
ppol human-reference fingerprint files. The dialogue data is
`cmu-lti/tau-usi <https://huggingface.co/datasets/cmu-lti/tau-usi>`_ on the
HF Hub — auto-downloaded on first use (or set ``PPolConfig.tau_bench_human_path`` 
to a local copy). Splits follow the official ``split_tasks.json`` train/val split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ppol.config import PPolConfig, canonical_domain_name, domain_list

# τ²-bench human dialogues on the HF Hub.
_HF_REPO = "cmu-lti/tau-usi"
_HF_FILE = "data/tau_bench_tasks_unified.json"


def resolve_tau_human_data(cfg: PPolConfig | None = None) -> str:
    """Path to the τ² human-dialogue JSON: ``PPolConfig.tau_bench_human_path``
    if that file exists locally, else auto-downloaded from ``cmu-lti/tau-usi``
    (cached by huggingface_hub)."""
    cfg = cfg or PPolConfig()
    local = Path(cfg.tau_bench_human_path)
    if local.is_file():
        return str(local)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(_HF_REPO, _HF_FILE, repo_type="dataset")


def _is_setup_turn(turn: dict) -> bool:
    """UI scaffolding: task line + canvas instruction — not real dialogue."""
    content = (turn.get("content") or "").strip()
    return content.startswith(("\\tau ", r"\tau ")) or "<|canvas|>" in content


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
        if len(clean) >= 2:
            dialogues.append({"conversation": clean, "instance_id": key})
    return dialogues


def compute_tau2bench_human_reference(
    output_path: str, *, domain: str = "retail", split: str = "all",
) -> str:
    """Fingerprint the τ²-bench human dialogues and save the reference distribution.

    Reads the dialogue JSON via :func:`resolve_tau_human_data` (local
    ``PPolConfig.tau_bench_human_path`` if present, else auto-downloaded from
    ``cmu-lti/tau-usi``); delegates the fingerprint + save to
    ``ppol.pipeline.compute_human_reference``.

    ``split``:
      * ``"all"``   — every dialogue (used for discriminator training).
      * ``"train"`` — train-split humans only: excludes official test-split tasks
        and the evolution val slice (the evolution-coverage cloud).
      * ``"test"``  — the exact complement of ``"train"`` (official-test + val
        humans), so train ∪ test = all with no overlap — the held-out side for
        ``ppol.evaluation.evaluate_simulators``.
    """
    from ppol.pipeline import compute_human_reference

    cfg = PPolConfig()
    data_json = Path(resolve_tau_human_data(cfg))

    split_filtered = split in ("train", "test")
    if split_filtered:
        from tau_train_context import (
            official_split_for_task_id,
            split_train_val,
            task_id_from_tau_human_instance_key,
        )
        taubench_root = Path(cfg.taubench_root)
        _, evolution_val_ids = split_train_val(
            seed=cfg.seed, val_fraction=cfg.val_fraction,
            domain=domain, taubench_root=taubench_root,
        )
        evolution_val_set = set(evolution_val_ids)
        qualify_task_ids = len(domain_list(domain)) > 1

    dialogues: List[Dict[str, Any]] = []
    for one_dom in domain_list(domain):
        rows = load_domain_dialogues(str(data_json), one_dom)
        if split_filtered:
            kept = []
            for d in rows:
                trace = d.get("conversation", d.get("turns", []))
                if len(trace) < 2:
                    continue
                key = str(d.get("instance_id", ""))
                tid = task_id_from_tau_human_instance_key(key, one_dom) if key else ""
                try:
                    sp = official_split_for_task_id(one_dom, tid, taubench_root) if tid else None
                except FileNotFoundError:
                    sp = None
                task_key = f"{one_dom}:{tid}" if qualify_task_ids and tid else str(tid or "")
                is_train = sp != "test" and task_key not in evolution_val_set
                if is_train == (split == "train"):
                    kept.append(d)
            rows = kept
        dialogues.extend(rows)
        print(f"  {one_dom}: {len(rows)} conversations" + (f" ({split} split)" if split_filtered else ""))

    if not dialogues:
        raise ValueError(f"No '{domain}' dialogues found in the τ²-bench human data.")

    return compute_human_reference(
        dialogues, output_path,
        domain=canonical_domain_name(domain),
        source="tau_bench_human",
    )
