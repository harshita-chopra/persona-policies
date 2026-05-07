import asyncio
import concurrent.futures
import logging
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)

_orig = None
_installed = False
_embedding_installed = False
_novelty_parse_installed = False


def _parse_novelty_verdict(content: str) -> Optional[bool]:
    """Return True for NOVEL, False for NOT NOVEL / NOT_NOVEL, None if unclear."""
    u = content.strip().upper()
    not_positions = [p for p in (u.find("NOT NOVEL"), u.find("NOT_NOVEL")) if p != -1]
    masked = u
    masked = masked.replace("NOT NOVEL", "\x00" * 9)
    masked = masked.replace("NOT_NOVEL", "\x00" * 9)
    p_novel = masked.find("NOVEL")
    if not not_positions and p_novel == -1:
        return None
    if not not_positions:
        return True
    if p_novel == -1:
        return False
    return p_novel < min(not_positions)


def _patched_llm_judge_novelty(self, program, similar_program) -> bool:
    """Patch OpenEvolve novelty parsing to treat NOT_NOVEL as negative."""
    from openevolve.novelty_judge import NOVELTY_SYSTEM_MSG, NOVELTY_USER_MSG

    user_msg = NOVELTY_USER_MSG.format(
        language=program.language,
        existing_code=similar_program.code,
        proposed_code=program.code,
    )
    try:
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.novelty_llm.generate_with_context(
                        system_message=NOVELTY_SYSTEM_MSG,
                        messages=[{"role": "user", "content": user_msg}],
                    ),
                )
                content: str = future.result()
        except RuntimeError:
            content = asyncio.run(
                self.novelty_llm.generate_with_context(
                    system_message=NOVELTY_SYSTEM_MSG,
                    messages=[{"role": "user", "content": user_msg}],
                )
            )

        if not content or not str(content).strip():
            logger.warning("Novelty LLM returned empty response")
            return True

        content = str(content).strip()
        verdict = _parse_novelty_verdict(content)
        if verdict is None:
            logger.warning(f"Unexpected novelty LLM response: {content[:500]!r}")
            return True
        return verdict

    except Exception as e:
        logger.error(f"Error in novelty LLM check: {e}")
    return True


def _install_novelty_verdict_parse() -> None:
    global _novelty_parse_installed
    if _novelty_parse_installed:
        return
    from openevolve.database import ProgramDatabase

    ProgramDatabase._llm_judge_novelty = _patched_llm_judge_novelty
    _novelty_parse_installed = True


def _run_iteration_worker(it: int, *a, **k):
    global _orig
    try:
        from pathlib import Path

        from openevolve.process_parallel import SerializableResult
        from persona_policies.config import PersonaPoliciesConfig

        cfg = PersonaPoliciesConfig()
        marker = Path(cfg.openevolve_output_dir) / "EARLY_STOP"
        if marker.is_file():
            return SerializableResult(error="early stop requested", iteration=it)
    except Exception:
        pass

    # Spawned process workers lazily construct LLMEnsemble inside OpenEvolve's
    # original worker function. Import the bridge in the worker immediately
    # before that happens, because the parent process import is not enough under
    # multiprocessing spawn.
    import persona_policies.evolution.litellm_ensemble_patch  # noqa: F401

    if _orig is None:
        import openevolve.process_parallel as p

        _orig = p._run_iteration_worker
    os.environ["OPENEVOLVE_ITERATION"] = str(it)
    try:
        return _orig(it, *a, **k)
    finally:
        os.environ.pop("OPENEVOLVE_ITERATION", None)


def _install() -> None:
    global _installed, _orig
    if _installed:
        return
    import openevolve.process_parallel as p

    _orig = p._run_iteration_worker
    p._run_iteration_worker = _run_iteration_worker
    _installed = True


class _OpenRouterEmbeddingClient:
    """OpenEvolve-compatible embedding client for OpenRouter embedding models."""

    def __init__(self, model_name: str):
        from openai import OpenAI

        if model_name.startswith("openrouter/"):
            model_name = model_name.removeprefix("openrouter/")
        self.model = model_name
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )

    def get_embedding(self, code):
        single_code = isinstance(code, str)
        inputs = [code] if single_code else list(code)
        response = self.client.embeddings.create(
            model=self.model,
            input=inputs,
            encoding_format="float",
            extra_headers={
                "HTTP-Referer": os.getenv(
                    "OPENROUTER_HTTP_REFERER", "https://github.com/persona-policies"
                ),
                "X-Title": os.getenv("OPENROUTER_X_TITLE", "persona-policies"),
            },
        )
        embeddings = [d.embedding for d in response.data]
        return embeddings[0] if single_code else embeddings


def _install_openrouter_embeddings() -> None:
    global _embedding_installed
    if _embedding_installed:
        return

    import openevolve.embedding as emb

    original = emb.EmbeddingClient

    class EmbeddingClient(original):
        def __new__(cls, model_name: str = "text-embedding-3-small"):
            if str(model_name).startswith("openrouter/") or str(model_name).startswith("qwen/"):
                return _OpenRouterEmbeddingClient(str(model_name))
            return super().__new__(cls)

        def __init__(self, model_name: str = "text-embedding-3-small"):
            if isinstance(self, _OpenRouterEmbeddingClient):
                return
            super().__init__(model_name)

    emb.EmbeddingClient = EmbeddingClient
    _embedding_installed = True


def main() -> int:
    # Ensure OpenEvolve routes gemini/... and bedrock/... model ids through
    # LiteLLM even when the optional site-packages .pth hook is not installed.
    try:
        import persona_policies.evolution.litellm_ensemble_patch  # noqa: F401
    except ModuleNotFoundError as e:
        if e.name != "litellm":
            raise
        print(
            "error: missing Python package in this environment: litellm\n"
            f"Install it with:\n  {sys.executable} -m pip install litellm",
            file=sys.stderr,
        )
        return 1

    _install()
    _install_openrouter_embeddings()
    _install_novelty_verdict_parse()
    from openevolve.cli import main as m

    return m()


if __name__ == "__main__":
    raise SystemExit(main())
