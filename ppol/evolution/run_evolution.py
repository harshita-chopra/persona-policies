"""Generic OpenEvolve launcher used by ``PPol.evolve()``.

Domain-agnostic: it spawns ``ppol.evolution.openevolve_entry`` as a subprocess
with the given initial program, evaluator (``fitness.py``), and config, and
manages the process tree / signals. The package only needs these primitives
so that ``PPol.evolve()`` works out of the box for any ``EpisodeRunner``.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
from pathlib import Path

# ppol/evolution/run_evolution.py -> parents[2] is the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _checkpoint_sort_key(p: Path) -> "tuple[int, int | float]":
    """Prefer numeric ``checkpoint_<N>`` ordering; fallback to mtime for others."""
    m = re.fullmatch(r"checkpoint_(\d+)", p.name)
    if m:
        return (1, int(m.group(1)))
    try:
        return (0, p.stat().st_mtime_ns)
    except OSError:
        return (0, -1)


def _popen_in_new_session(*args, **kwargs) -> subprocess.Popen:
    """Start a child in its own process group so we can kill the whole tree."""
    if sys.platform != "win32":
        if sys.version_info >= (3, 11):
            kwargs["start_new_session"] = True
        else:
            kwargs["preexec_fn"] = os.setsid
    return subprocess.Popen(*args, **kwargs)


def _terminate_process_tree(proc: subprocess.Popen, *, grace_sec: float = 1.0) -> None:
    """SIGTERM then SIGKILL the child and any OpenEvolve worker processes."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        proc.terminate()
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            proc.terminate()
    try:
        proc.wait(timeout=grace_sec)
        return
    except subprocess.TimeoutExpired:
        pass
    if sys.platform == "win32":
        proc.kill()
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            proc.kill()
    proc.wait()


def run_openevolve(
    initial_program_path: str,
    evaluator_path: str,
    config_path: str,
    output_dir: str,
    n_iterations: int,
    resume_checkpoint: "str | None",
    log_level: "str | None",
) -> bool:
    """Run OpenEvolve as a subprocess. Returns True on success/early-stop."""
    cmd = [
        sys.executable,
        "-m",
        "ppol.evolution.openevolve_entry",
        initial_program_path,
        evaluator_path,
        "--config",
        config_path,
        "--output",
        output_dir,
        "--iterations",
        str(n_iterations),
    ]
    if resume_checkpoint and os.path.isdir(resume_checkpoint):
        cmd.extend(["--checkpoint", resume_checkpoint])
    if log_level:
        cmd.extend(["--log-level", log_level])

    print(f"\n{'='*70}\nEVOLVING PERSONA GENERATOR\nCommand: {' '.join(cmd)}\n{'='*70}")

    (Path(output_dir) / "EARLY_STOP").unlink(missing_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    r = str(_REPO_ROOT)
    p = env.get("PYTHONPATH", "")
    if r not in p.split(os.pathsep):
        env["PYTHONPATH"] = f"{r}{os.pathsep}{p}" if p else r

    proc = _popen_in_new_session(cmd, env=env)
    interrupted = False

    def _on_signal(signum: int, _frame) -> None:
        nonlocal interrupted
        if interrupted:
            print("\n==> Force stop.", flush=True)
            _terminate_process_tree(proc, grace_sec=0.0)
            raise KeyboardInterrupt
        interrupted = True
        print("\n==> Interrupted — stopping OpenEvolve and worker processes...", flush=True)
        _terminate_process_tree(proc)

    old_handlers: dict = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            old_handlers[sig] = signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass

    try:
        rc = proc.wait()
    except KeyboardInterrupt:
        interrupted = True
        _terminate_process_tree(proc)
        rc = 130
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)

    if interrupted:
        return False
    return rc == 0 or (Path(output_dir) / "EARLY_STOP").is_file()
