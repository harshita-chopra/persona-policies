#!/usr/bin/env bash
# =============================================================================
# Persona Policies — full pipeline from a clean tree (resume-friendly steps)
# =============================================================================
# From the repository root:
#   chmod +x persona_policies/run.sh && ./persona_policies/run.sh
# Or from this directory:
#   chmod +x run.sh && ./run.sh
#
# All durable artifacts go under: persona_policies/outputs/
#
# Human behavioral reference: built in step [5] from tau_bench_human.json (train_discriminator.py).
#
# LLM guidance (edit persona_policies/config.py and evolution/openevolve_config.yaml):
#   • Gemini routes through GEMINI_API_KEY (LiteLLM model ids: gemini/...).
#   • Gemma routes through OPENROUTER_API_KEY (LiteLLM model ids: openrouter/google/...).
#   • DeepSeek routes through AWS Bedrock (LiteLLM model ids: bedrock/...).
#   • τ² *agent* (taubench_agent_model): task success depends on it most.
#   • τ² *user simulator* (taubench_user_model): smaller/cheaper is fine (e.g. DeepSeek V3).
#   • Coherence / evolution_feedback_model: mid-tier OK.
#   • OpenEvolve *mutation* LLM: must route through LiteLLM — run step 3 once/venv.
#
# AWS: export AWS_REGION / credentials as you do for Bedrock today.
# Gemini: export GEMINI_API_KEY.
# OpenRouter Gemma: export OPENROUTER_API_KEY.
#
# Resume / keep artifacts: export SKIP_PIPELINE_CLEAN=1 before running so outputs/
# (baselines, checkpoints, OpenEvolve DB) are NOT deleted. To resume evolution only:
#   python persona_policies/evolution/run_evolution.py --iterations 400 --resume
# =============================================================================

set -euo pipefail

# Ctrl+C / SIGTERM: kill this script's entire process group so pip, Python, and
# openevolve-run (and typical children) cannot outlive the shell.
_on_pipeline_signal() {
  local code=130
  [[ "${1:-}" == TERM ]] && code=143
  echo "" >&2
  echo "==> Interrupted — stopping pipeline and child processes..." >&2
  # Negative PID = process group (same as this bash's PGID when running ./run.sh)
  kill -TERM -- -$$ 2>/dev/null || true
  sleep 0.5
  kill -KILL -- -$$ 2>/dev/null || true
  exit "${code}"
}
trap '_on_pipeline_signal INT' INT
trap '_on_pipeline_signal TERM' TERM

PP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PP_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# --- Wipe outputs unless SKIP_PIPELINE_CLEAN=1 (required for true resume of evolution) ---
if [ "${SKIP_PIPELINE_CLEAN:-}" = "1" ]; then
  echo "==> SKIP_PIPELINE_CLEAN=1 — keeping persona_policies/outputs/ and related dirs"
else
  echo "==> Cleaning old pipeline dirs (set SKIP_PIPELINE_CLEAN=1 to keep checkpoints / resume)"
  rm -rf \
    "${PP_DIR}/outputs" \
    "${REPO_ROOT}/tau2_outputs" \
    2>/dev/null || true
fi

echo "==> Creating output tree under persona_policies/outputs/"
python - <<'PY'
from persona_policies.config import PersonaPoliciesConfig
c = PersonaPoliciesConfig()
c.ensure_output_dirs()
print("OK:", c.outputs_root)
PY

echo "==> [1/6] Python dependencies"
python -m pip install -U pip
python -m pip install -r "${PP_DIR}/requirements.txt"
python -m pip install -U openevolve litellm

echo "==> [2/6] τ²-bench — install from the official repository if not already installed"
echo "    e.g.: pip install -e /path/to/tau2-bench"
echo "    Skipping automatic install; ensure tau2 is importable before proceeding."

echo "==> [3/6] OpenEvolve → LiteLLM / Bedrock hook (once per venv; idempotent)"
python "${PP_DIR}/evolution/install_openevolve_litellm_pth.py" || true

echo "==> [4/6] Baseline τ² episodes (train split, no persona)"
python "${PP_DIR}/scripts/collect_baseline.py" --n 100

echo "==> [5/6] Human distribution (τ² human, retail/airline per config) + discriminator (needs [4])"
python "${PP_DIR}/scripts/train_discriminator.py"

echo "==> [6/6] OpenEvolve persona evolution — G(c, D, N)"
echo "    Evolving Stage1 (joint archetype gen) + Stage2 (task expansion) prompts."
echo "    Each iteration: batch of val tasks → generate N personas → τ² rollouts → score."

# Fresh start. For resume after interruption: use --resume (requires existing checkpoints)
EVO_FLAGS=(--iterations 100)
if [ "${EVOLUTION_RESUME:-}" = "1" ]; then
  EVO_FLAGS+=(--resume)
  echo "    EVOLUTION_RESUME=1 — continuing from latest OpenEvolve checkpoint"
fi
python "${PP_DIR}/evolution/run_evolution.py" "${EVO_FLAGS[@]}"

echo ""
echo "Done. Key paths (under persona_policies/outputs/):"
echo "  reference_data/        — baselines, human FP, discriminator (fitness inputs)"
echo "  training/openevolve/   — OpenEvolve --output: checkpoints/, best/best_program.py, logs/"
echo "  training/simulations/iter_NNNN/ — τ² episode logs per fitness eval"
echo "  training/results/evolution_best.json — best metrics when run completes"
echo "  testing/personas/*.json — optional persona JSON for benchmark.py"
echo ""
echo "Benchmark (test split), after evolution:"
echo "  python persona_policies/benchmark.py"
echo ""
echo "Resume evolution:"
echo "  python persona_policies/evolution/run_evolution.py --iterations 100 --resume"
