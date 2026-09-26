"""Incremental train/val metric plots under ``training/results/``.

Writes ``train_curve.jsonl`` (one line per successful fitness iteration) and
refreshes ``evolution_scores.png`` using the same plotting logic as
``analysis/plot_evolution_scores.py``. In ``val_curve.jsonl``, ``iteration``
is the trial that produced the validated elite, so val points align with
their producing trial on the x-axis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from persona_policies.config import PersonaPoliciesConfig


def append_train_curve_row(
    config: PersonaPoliciesConfig,
    *,
    iteration: int,
    epoch: int,
    batch_in_epoch: int,
    steps_per_epoch: int,
    program_path: str,
    train_metrics: Dict[str, Any],
    train_score: float,
) -> None:
    """Append one JSON line compatible with ``plot_scores`` train series."""
    hl = float(train_metrics.get("human_likeness", float("nan")))
    intra = train_metrics.get("intra_set_diversity")
    intra_f = float(intra) if intra is not None else float("nan")
    comb = float(train_metrics.get("combined_score", train_score))

    row: Dict[str, Any] = {
        "iteration": int(iteration),
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "steps_per_epoch": int(steps_per_epoch),
        "program": Path(program_path).name,
        "final_combined_score": float(train_score),
        "train_combined": comb,
        "train_human_likeness": hl,
        "train_intra_diversity": intra_f,
    }
    out = Path(config.training_results_dir) / "train_curve.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _load_train_curve_jsonl(path: Path) -> List[Dict[str, Any]]:
    """One JSON object per line, plus tolerate multiple objects glued on one line (crash/interrupt)."""
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    dec = json.JSONDecoder()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        i, n = 0, len(line)
        while i < n:
            while i < n and line[i].isspace():
                i += 1
            if i >= n:
                break
            try:
                obj, end = dec.raw_decode(line, i)
            except json.JSONDecodeError:
                break
            if isinstance(obj, dict):
                rows.append(obj)
            i = end
    rows.sort(key=lambda r: int(r.get("iteration", 0)))
    return rows


def refresh_evolution_plots(config: PersonaPoliciesConfig) -> None:
    """Rewrite ``training/results/evolution_scores.png`` from curve files."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        from persona_policies.analysis.plot_evolution_scores import (
            collect_val_rows,
            load_train_rows_for_plot,
            plot_scores,
        )
    except Exception as e:
        print(f"[metrics_plot] refresh skipped: {e}", flush=True)
        return

    root = Path(config.training_results_dir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    train_rows = load_train_rows_for_plot(config)
    val_rows = collect_val_rows(root / "val_curve.jsonl")
    if not train_rows and not val_rows:
        return
    plot_scores(
        train_rows,
        val_rows,
        root / "evolution_scores.png",
        title="Evolution: scores vs iteration",
        verbose=False,
        contiguous_x=False,
    )
