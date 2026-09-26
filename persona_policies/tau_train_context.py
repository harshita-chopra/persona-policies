"""
Build simulation context **c** from τ²-bench tasks: original user-scenario
``persona`` + ``instructions`` (same information the user simulator sees for that task).

Provides **train / val** split from the official ``split_tasks.json`` train set
(80/20 by default) so evolution evaluates on a held-out subset.
"""

from __future__ import annotations

import json
import random
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from persona_policies.config import domain_list, is_combined_domain

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_TAU2_ROOT = _REPO / "tau2-bench"


def _domain_paths(
    domain: str = "retail",
    taubench_root: Optional[Path] = None,
) -> tuple[Path, Path]:
    root = Path(taubench_root) if taubench_root is not None else _DEFAULT_TAU2_ROOT
    domain = str(domain or "retail")
    return (
        root / "data/tau2/domains" / domain / "split_tasks.json",
        root / "data/tau2/domains" / domain / "tasks.json",
    )


def load_split_task_ids(
    domain: str = "retail",
    split: str = "train",
    taubench_root: Optional[Path] = None,
) -> List[str]:
    domains = domain_list(domain)
    if len(domains) > 1:
        out: List[str] = []
        for dom in domains:
            out.extend(f"{dom}:{tid}" for tid in load_split_task_ids(dom, split, taubench_root))
        return out
    split_path, _ = _domain_paths(domain, taubench_root)
    if not split_path.is_file():
        return []
    with open(split_path, encoding="utf-8") as f:
        data = json.load(f)
    return [str(x) for x in data.get(split, [])]


def load_train_task_ids_retail() -> List[str]:
    """Backward-compatible retail helper. Prefer ``load_split_task_ids(domain, "train")``."""
    return load_split_task_ids("retail", "train")


def load_test_task_ids_retail() -> List[str]:
    """Backward-compatible retail helper. Prefer ``load_split_task_ids(domain, "test")``."""
    return load_split_task_ids("retail", "test")


def load_domain_tasks(
    domain: str = "retail",
    taubench_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    domains = domain_list(domain)
    if len(domains) > 1:
        out: List[Dict[str, Any]] = []
        for dom in domains:
            for task in load_domain_tasks(dom, taubench_root):
                row = dict(task)
                row["_domain"] = dom
                row["_combined_id"] = f"{dom}:{task.get('id')}"
                out.append(row)
        return out
    _, tasks_path = _domain_paths(domain, taubench_root)
    if not tasks_path.is_file():
        return []
    with open(tasks_path, encoding="utf-8") as f:
        return json.load(f)


def load_retail_tasks() -> List[Dict[str, Any]]:
    """Backward-compatible retail helper. Prefer ``load_domain_tasks(domain)``."""
    return load_domain_tasks("retail")


def _task_by_id(tasks: List[Dict[str, Any]], task_id: str) -> Optional[Dict[str, Any]]:
    for t in tasks:
        if str(t.get("_combined_id", t.get("id"))) == str(task_id):
            return t
    return None


def format_user_scenario_c(task: Dict[str, Any]) -> str:
    """Format task context like τ² ``UserScenario`` (persona + instructions) with clear section labels.
    """
    us = task.get("user_scenario") or {}
    persona = us.get("persona")
    instructions = us.get("instructions")
    lines: List[str] = []
    if persona is not None and str(persona).strip():
        lines.append("Base Persona:")
        lines.append(textwrap.indent(str(persona), "\t"))
    lines.append("Given Instructions:")
    if isinstance(instructions, dict):
        parts = []
        for k, v in sorted(instructions.items()):
            parts.append(f"{k}: {v}")
        lines.append(textwrap.indent("\n".join(parts), "\t"))
    else:
        lines.append(textwrap.indent(str(instructions), "\t"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Train / val split (within the official "train" set)
# ---------------------------------------------------------------------------

def split_train_val(
    seed: int = 42,
    val_fraction: float = 0.2,
    domain: str = "retail",
    taubench_root: Optional[Path] = None,
) -> Tuple[List[str], List[str]]:
    """Disjoint ``(train_ids, val_ids)`` slices of the official train split."""
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction!r}")
    ids = load_split_task_ids(domain, "train", taubench_root)
    if not ids:
        split_path, _ = _domain_paths(domain, taubench_root)
        raise RuntimeError(
            f"No official train task ids for domain={domain!r} "
            f"(check {split_path})."
        )
    random.Random(seed).shuffle(ids)
    n_val = min(max(1, int(len(ids) * val_fraction)), len(ids) - 1)
    return ids[n_val:], ids[:n_val]


def take_task_batch_sequential(
    task_ids: List[str],
    batch_size: int,
    start_index: int,
    domain: str = "retail",
    taubench_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Take ``batch_size`` full task dicts in **fixed ``task_ids`` order**, starting at
    ``start_index`` and wrapping with modulo (no shuffle). Used for evolution **train** batches.

    Use ``start_index = (iteration - 1) * batch_size`` (modulo ``len(task_ids)``) so each
    evaluate() call advances through the train pool.
    """
    tasks = load_domain_tasks(domain, taubench_root)
    if not task_ids or not tasks:
        return []
    n = len(task_ids)
    if n == 0:
        return []
    start = start_index % n
    out: List[Dict[str, Any]] = []
    for k in range(batch_size):
        tid = task_ids[(start + k) % n]
        task = _task_by_id(tasks, tid)
        if task is not None:
            out.append(task)
    return out


def sample_task_batch(
    task_ids: List[str],
    batch_size: int,
    rng: random.Random,
    domain: str = "retail",
    taubench_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Sample ``batch_size`` full task dicts from ``task_ids`` (shuffled). Used for **val**."""
    tasks = load_domain_tasks(domain, taubench_root)
    if not task_ids or not tasks:
        return []
    ids = list(task_ids)
    rng.shuffle(ids)
    chosen = ids[:batch_size]
    out: List[Dict[str, Any]] = []
    for tid in chosen:
        task = _task_by_id(tasks, tid)
        if task is not None:
            out.append(task)
    return out


def sample_context_batch(
    task_ids: List[str],
    batch_size: int,
    rng: random.Random,
    domain: str = "retail",
    taubench_root: Optional[Path] = None,
) -> List[str]:
    """Sample ``batch_size`` formatted **c** strings from ``task_ids``."""
    task_dicts = sample_task_batch(task_ids, batch_size, rng, domain, taubench_root)
    return [format_user_scenario_c(t) for t in task_dicts]


# ---------------------------------------------------------------------------
# Official τ² train / test (per-domain split_tasks.json)
# ---------------------------------------------------------------------------


def split_tasks_json_path(domain: str, taubench_root: Path) -> Path:
    """``.../tau2-bench/data/tau2/domains/<domain>/split_tasks.json``."""
    return Path(taubench_root) / "data" / "tau2" / "domains" / domain / "split_tasks.json"


def load_official_train_test_ids(
    domain: str,
    taubench_root: Path,
) -> tuple[set[str], set[str]]:
    """Load official ``train`` and ``test`` task id sets from τ² ``split_tasks.json``."""
    p = split_tasks_json_path(domain, taubench_root)
    if not p.is_file():
        raise FileNotFoundError(f"Missing split file: {p}")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    train = {str(x) for x in data.get("train", [])}
    test = {str(x) for x in data.get("test", [])}
    return train, test


def official_split_for_task_id(
    domain: str,
    task_id: str,
    taubench_root: Path,
) -> Optional[str]:
    """Return ``\"train\"``, ``\"test\"``, or ``None`` if the id is not in the split file."""
    train, test = load_official_train_test_ids(domain, taubench_root)
    tid = str(task_id)
    if tid in train:
        return "train"
    if tid in test:
        return "test"
    return None


def task_id_from_tau_human_instance_key(instance_key: str, domain: str) -> str:
    """Map ``tau_bench_human.json`` key to a task id: ``retail_12_ann2`` -> ``12``."""
    prefix = domain + "_"
    if not instance_key.startswith(prefix):
        return ""
    rest = instance_key[len(prefix) :]
    if "_ann" in rest:
        rest = rest.split("_ann")[0]
    return rest
