#!/usr/bin/env python3
"""
Install a site-packages .pth file so every Python process (including OpenEvolve worker
processes) loads ``litellm_ensemble_patch`` before ``LLMEnsemble`` builds OpenAI clients.

Run once per virtualenv from the repo root:

  python persona_policies/evolution/install_openevolve_litellm_pth.py

Then ``openevolve-run`` and ``run_evolution.py`` can use LiteLLM routes such as
``gemini/...`` and ``bedrock/...`` without OPENAI_API_KEY for the mutation LLM.
"""

from __future__ import annotations

import site
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _site_packages() -> Path:
    try:
        sp = site.getsitepackages()
        if sp:
            return Path(sp[0])
    except Exception:
        pass
    try:
        usp = site.getusersitepackages()
        if usp:
            return Path(usp)
    except Exception:
        pass
    print("Could not resolve site-packages; set PYTHONPATH to repo root manually.", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    dest = _site_packages() / "persona_policies_openevolve_litellm.pth"
    lines = [
        str(_REPO_ROOT),
        "import persona_policies.evolution.litellm_ensemble_patch",
        "",
    ]
    dest.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {dest}")
    print("Restart your shell or open a new terminal so Python picks this up.")


if __name__ == "__main__":
    main()
