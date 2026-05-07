"""
Compute human behavioral reference distribution from tau_bench_human.json.

Filters conversations by domain (matching ``config.taubench_domain``, e.g.
``retail`` or ``airline``).  Strips the first setup turns
(``\\tau task_index:...`` + canvas instruction) which are platform
scaffolding, not real dialogue.

Run:
    python persona_policies/scripts/compute_human_reference.py
    python persona_policies/scripts/compute_human_reference.py --domain airline
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from persona_policies.config import PersonaPoliciesConfig
from persona_policies.config import canonical_domain_name, domain_list
from persona_policies.fingerprinting import (
    BehavioralFingerprintExtractor,
    compute_human_distribution,
)
from persona_policies.tau_human_loader import load_domain_dialogues


def compute_and_save(
    config: Optional[PersonaPoliciesConfig] = None,
    *,
    domain: Optional[str] = None,
) -> str:
    """Build human reference from τ² human JSON, save to ``human_fingerprints_cache``."""
    config = config or PersonaPoliciesConfig()
    config.ensure_output_dirs()
    extractor = BehavioralFingerprintExtractor()

    dom = domain or config.taubench_domain
    if dom != config.taubench_domain:
        config.taubench_domain = dom
        config.refresh_domain_artifact_paths()
    src = config.tau_bench_human_path
    if not os.path.isfile(src):
        raise FileNotFoundError(f"τ² human data not found: {src}")

    print(f"Loading '{canonical_domain_name(dom)}' dialogues from {src} ...")
    dialogues = []
    for one_dom in domain_list(dom):
        rows = load_domain_dialogues(src, one_dom)
        dialogues.extend(rows)
        print(f"  Found {len(rows)} {one_dom} conversations (setup turns stripped)")

    if not dialogues:
        raise ValueError(
            f"No '{dom}' dialogues in {src}. Check taubench_domain / file contents.",
        )

    human_dist = compute_human_distribution(
        dialogues,
        extractor,
        source="tau_bench_human",
        domain=canonical_domain_name(dom),
    )

    out_path = config.human_fingerprints_cache
    human_dist.save(out_path)
    print(f"\nSaved human behavioral distribution to {out_path}")
    print(f"  Domain:    {dom}")
    print(f"  Features:  {len(human_dist.feature_names)}")
    print(f"  Dialogues: {human_dist.n_dialogues}")

    print("\nPer-feature means:")
    for f in human_dist.feature_names:
        print(f"  {f:35s}: {human_dist.mean[f]:.4f}")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute human behavioral reference from tau_bench_human.json",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Domain to filter by (default: config.taubench_domain, usually retail)",
    )
    args = parser.parse_args()

    config = PersonaPoliciesConfig()
    try:
        compute_and_save(config, domain=args.domain)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
