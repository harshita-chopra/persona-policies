"""
Configure import paths and data directory for τ²-bench (tau2) before importing ``tau2``.

Call ``ensure_taubench_importable()`` at the start of any script that uses
``persona_policies.injector`` or other tau2-backed modules.

If τ²-bench is installed via pip (``pip install -e /path/to/tau2-bench``), the
import path is already set up. This module additionally sets ``TAU2_DATA_DIR``
so τ² can locate its task data. Override by setting ``TAU2_DATA_DIR`` yourself
before calling any tau2-backed function, or pass ``taubench_root`` explicitly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TAU2_BENCH = _REPO_ROOT / "tau2-bench"
_DEFAULT_TAU2_SRC = _DEFAULT_TAU2_BENCH / "src"
_DEFAULT_TAU2_DATA = _DEFAULT_TAU2_BENCH / "data"


def ensure_taubench_importable(
    taubench_root: str | Path | None = None,
) -> Path:
    """
    Set ``TAU2_DATA_DIR`` and prepend tau2 ``src`` to ``sys.path``.

    Returns the resolved tau2-bench root directory.
    """
    root = Path(taubench_root) if taubench_root else _DEFAULT_TAU2_BENCH
    root = root.resolve()
    src = root / "src"
    data = root / "data"
    if data.is_dir():
        os.environ.setdefault("TAU2_DATA_DIR", str(data))
    if src.is_dir():
        s = str(src)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


def apply_tau2_bedrock_judge_overrides(
    nl_model: str = "bedrock/deepseek.v3-v1:0",
    env_model: str = "bedrock/deepseek.v3-v1:0",
) -> None:
    """
    τ² defaults NL-assertion and env-interface judges to OpenAI (``gpt-4.1``).
    Persona Policies strips ``OPENAI_API_KEY`` for Bedrock-only runs (see injector);
    without this patch, ``ALL_WITH_NL_ASSERTIONS`` intermittently fails with missing OpenAI key.

    Patches both ``tau2.config`` and modules that bind the default at import time.
    """
    ensure_taubench_importable()
    import tau2.config as tc

    tc.DEFAULT_LLM_NL_ASSERTIONS = nl_model
    tc.DEFAULT_LLM_ENV_INTERFACE = env_model
    try:
        import tau2.evaluator.evaluator_nl_assertions as nle

        nle.DEFAULT_LLM_NL_ASSERTIONS = nl_model
    except ImportError:
        pass
    try:
        import tau2.environment.utils.interface_agent as ia

        ia.DEFAULT_LLM_ENV_INTERFACE = env_model
    except ImportError:
        pass
