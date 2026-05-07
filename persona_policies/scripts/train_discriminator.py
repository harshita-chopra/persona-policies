"""
Train the behavioral discriminator.

Builds the human reference from **τ²-bench human** data (``tau_bench_human.json``)
filtered by ``config.taubench_domain`` (default ``retail``).

**Train / test protocol:** uses official ``split_tasks.json`` labels.
Human dialogues and baseline simulator fingerprints are split by task id into **train**
vs **test**. The RandomForest is fit on **train human + train sim** only; held-out
**test human + test sim** are used for ROC-AUC / accuracy. The saved
``human_fingerprints_<domain>.json`` still aggregates **all** domain dialogues for
the coverage reference (evaluator).

Requires (under ``persona_policies/outputs/reference_data/`` by default):
- Baseline sim fingerprints from ``collect_baseline.py`` with per-row
  ``{fingerprint, domain, task_id, split}`` (see ``baseline_*_fingerprints.json``).
- ``persona_policies/data/tau_bench_human.json`` (see ``config.tau_bench_human_path``)

Run:

  python persona_policies/scripts/train_discriminator.py
  python persona_policies/scripts/train_discriminator.py --domain airline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from persona_policies.config import PersonaPoliciesConfig
from persona_policies.config import canonical_domain_name, domain_list
from persona_policies.discriminator import BehavioralDiscriminator
from persona_policies.fingerprinting import (
    BehavioralFingerprint,
    BehavioralFingerprintExtractor,
    compute_human_distribution,
)
from persona_policies.tau_human_loader import load_domain_dialogues
from persona_policies.tau_train_context import (
    official_split_for_task_id,
    split_train_val,
    task_id_from_tau_human_instance_key,
)


def _qualified_task_key(domain_selector: str, row_domain: str | None, task_id: str | None) -> str:
    tid = str(task_id or "")
    if len(domain_list(domain_selector)) > 1:
        return f"{row_domain}:{tid}" if row_domain else tid
    return tid


def _human_fingerprints_tau_with_split(
    dialogues: list[dict],
    extractor: BehavioralFingerprintExtractor,
    domain: str,
    domain_selector: str,
    taubench_root: Path,
) -> tuple[list[BehavioralFingerprint], list[str | None], list[str]]:
    """Fingerprints aligned with per-dialogue official split and task key."""
    fps: list[BehavioralFingerprint] = []
    splits: list[str | None] = []
    task_keys: list[str] = []
    for dialogue in dialogues:
        trace = dialogue.get("conversation", dialogue.get("turns", []))
        if not trace:
            continue
        key = dialogue.get("instance_id", "")
        tid = task_id_from_tau_human_instance_key(str(key), domain)
        sp = official_split_for_task_id(domain, tid, taubench_root) if tid else None
        fps.append(
            extractor.compute_fingerprint(trace, user_role="user", agent_role="assistant"),
        )
        splits.append(sp)
        task_keys.append(_qualified_task_key(domain_selector, domain, tid))
    return fps, splits, task_keys


def _load_baseline_rows(path: str) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)
    rows: list[dict] = []
    for row in raw:
        if isinstance(row, dict) and "fingerprint" in row:
            rows.append(row)
        else:
            rows.append(
                {
                    "fingerprint": row,
                    "domain": None,
                    "task_id": None,
                    "split": None,
                }
            )
    return rows


def _rows_to_fps(rows: list[dict]) -> list[BehavioralFingerprint]:
    return [BehavioralFingerprint(features=dict(r["fingerprint"])) for r in rows]


def _filter_rows_split(rows: list[dict], split: str) -> list[dict]:
    return [r for r in rows if r.get("split") == split]


def _filter_rows_excluding_task_keys(
    rows: list[dict],
    excluded_task_keys: set[str],
    domain_selector: str,
) -> list[dict]:
    if not excluded_task_keys:
        return rows
    return [
        r for r in rows
        if _qualified_task_key(domain_selector, r.get("domain"), r.get("task_id"))
        not in excluded_task_keys
    ]


def main():
    parser = argparse.ArgumentParser(description="Train behavioral discriminator")
    parser.add_argument(
        "--domain",
        default=None,
        help="τ² domain to train on (default: config.taubench_domain, usually retail)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Exclude the same internal evolution val slice from discriminator training, e.g. 0.1.",
    )
    args = parser.parse_args()

    config = PersonaPoliciesConfig()
    extractor = BehavioralFingerprintExtractor()
    config.ensure_output_dirs()
    taubench_root = Path(config.taubench_root)

    domain = args.domain or config.taubench_domain
    if domain != config.taubench_domain:
        config.taubench_domain = domain
        config.refresh_domain_artifact_paths()
    if args.val_fraction is not None:
        config.val_fraction = float(args.val_fraction)
    src = config.tau_bench_human_path
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Missing {src}. Add tau_bench_human.json to persona_policies/data/.")

    print(f"Loading τ² human dialogues (domain={canonical_domain_name(domain)}) from {src}...")
    dialogues = []
    dialogue_domains: list[str] = []
    for one_dom in domain_list(domain):
        rows = load_domain_dialogues(src, one_dom)
        dialogues.extend(rows)
        dialogue_domains.extend([one_dom] * len(rows))
        print(f"  {one_dom}: {len(rows)} conversations")

    if not dialogues:
        raise ValueError(f"No '{domain}' dialogues found in {src}")

    print("Computing human fingerprints (one per conversation) + official split labels...")
    human_fingerprints = []
    human_splits = []
    human_task_keys = []
    for one_dom in domain_list(domain):
        dom_dialogues = [
            d for d, d_dom in zip(dialogues, dialogue_domains, strict=True)
            if d_dom == one_dom
        ]
        fps, splits, task_keys = _human_fingerprints_tau_with_split(
            dom_dialogues, extractor, one_dom, domain, taubench_root
        )
        human_fingerprints.extend(fps)
        human_splits.extend(splits)
        human_task_keys.extend(task_keys)
    print(f"  {len(human_fingerprints)} fingerprints")

    n_unlabeled = sum(1 for s in human_splits if s is None)
    if n_unlabeled:
        print(
            f"  Warning: {n_unlabeled} dialogues have task ids outside split_tasks.json (skipped for train/test)."
        )

    print("Computing human behavioral distribution (all dialogues, for coverage reference)...")
    human_dist = compute_human_distribution(
        dialogues,
        extractor,
        source="tau_bench_human",
        domain=canonical_domain_name(domain),
    )

    os.makedirs(os.path.dirname(config.human_fingerprints_cache) or ".", exist_ok=True)
    human_dist.save(config.human_fingerprints_cache)
    print(f"Saved human distribution to {config.human_fingerprints_cache}")

    print("Loading baseline simulator fingerprints...")
    baseline_path = config.baseline_fingerprints_path
    if not os.path.isfile(baseline_path):
        raise FileNotFoundError(
            f"Missing {baseline_path}. Run: python persona_policies/scripts/collect_baseline.py",
        )
    sim_rows = _load_baseline_rows(baseline_path)
    print(f"Loaded {len(sim_rows)} simulator fingerprint rows")

    if any(r.get("split") is not None for r in sim_rows):
        _, evolution_val_ids = split_train_val(
            seed=config.seed,
            val_fraction=config.val_fraction,
            domain=domain,
            taubench_root=taubench_root,
        )
        evolution_val_set = set(evolution_val_ids)
        if args.val_fraction is not None:
            print(
                f"Excluding evolution val from discriminator train: "
                f"{len(evolution_val_set)} task ids (val_fraction={config.val_fraction})"
            )
        h_train_idx = [i for i, s in enumerate(human_splits) if s == "train"]
        h_test_idx = [i for i, s in enumerate(human_splits) if s == "test"]
        sim_train = _filter_rows_split(sim_rows, "train")
        sim_test = _filter_rows_split(sim_rows, "test")

        if args.val_fraction is not None:
            h_train_idx = [
                i for i in h_train_idx
                if human_task_keys[i] not in evolution_val_set
            ]
            sim_train = _filter_rows_excluding_task_keys(
                sim_train,
                evolution_val_set,
                domain,
            )

        human_train = [human_fingerprints[i] for i in h_train_idx]
        human_test = [human_fingerprints[i] for i in h_test_idx]

        print(
            f"Split sizes — human: train={len(human_train)} test={len(human_test)}; "
            f"sim: train={len(sim_train)} test={len(sim_test)}"
        )
        if not human_train or not sim_train:
            raise ValueError(
                "Train split empty for human or sim. Check tau_bench_human keys and baseline_collect_split."
            )
        if not human_test or not sim_test:
            print(
                "Warning: empty test split for human or sim — metrics will be skipped. "
                "Collect baseline with --collect-split all and ensure fingerprint splits."
            )

        train_human = human_train
        train_sim = _rows_to_fps(sim_train)
        test_human = human_test
        test_sim = _rows_to_fps(sim_test)
    else:
        print(
            "Warning: baseline fingerprints lack per-row `split` metadata — "
            "training on all human + all sim (legacy mode)."
        )
        train_human = human_fingerprints
        train_sim = _rows_to_fps(sim_rows)
        test_human = []
        test_sim = []

    print("\nTraining discriminator (train split)...")
    discriminator = BehavioralDiscriminator()
    discriminator.train(train_human, train_sim, verbose=True)

    if test_human and test_sim:
        Xh = discriminator.fingerprints_to_matrix(test_human)
        Xs = discriminator.fingerprints_to_matrix(test_sim)
        X_te = np.vstack([Xh, Xs])
        y_te = np.array([1] * len(test_human) + [0] * len(test_sim))
        X_te_s = discriminator.scaler.transform(X_te)
        proba = discriminator.clf.predict_proba(X_te_s)[:, 1]
        pred = (proba >= 0.5).astype(int)
        try:
            auc = roc_auc_score(y_te, proba)
        except ValueError:
            auc = float("nan")
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred)
        print(f"\nHeld-out test (human + sim): ROC-AUC={auc:.4f}  accuracy={acc:.4f} f1={f1:.4f}")

    discriminator.save(config.discriminator_model_path)
    print(f"\nDiscriminator saved to {config.discriminator_model_path}")

    print("\n=== FEATURE IMPORTANCE (RandomForest) ===")
    imp = discriminator.clf.feature_importances_
    feature_names = extractor.feature_names()
    sorted_idx = np.argsort(imp)[::-1]
    print("Highest importance:")
    for i in sorted_idx[:12]:
        print(f"  {feature_names[i]:40s}: {imp[i]:.4f}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
