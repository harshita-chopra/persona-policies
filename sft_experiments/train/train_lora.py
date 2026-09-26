from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SFT_DATA = ROOT / "data" / "sft"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve(cfg_path: Path) -> dict:
    cfg = load_yaml(cfg_path)
    parent = cfg.pop("_inherit", None)
    if parent:
        base = load_yaml(CONFIGS / parent)
        base.update(cfg)
        cfg = base
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    args = ap.parse_args()

    lf_dir = Path(os.environ.get("LLAMA_FACTORY_DIR", "")).expanduser()
    if not lf_dir.is_dir():
        sys.exit("Set LLAMA_FACTORY_DIR to your LLaMA-Factory checkout.")

    cfg = resolve(args.config)

    lf_data = lf_dir / "data"
    lf_data.mkdir(parents=True, exist_ok=True)

    info_path = lf_data / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    info.update(json.loads((CONFIGS / "dataset_info.json").read_text(encoding="utf-8")))
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    if SFT_DATA.is_dir():
        for sub in SFT_DATA.iterdir():
            if sub.is_dir():
                shutil.copytree(sub, lf_data / sub.name, dirs_exist_ok=True)

    resolved = lf_dir / "_resolved" / args.config.name
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    subprocess.check_call(["llamafactory-cli", "train", str(resolved)], cwd=lf_dir)


if __name__ == "__main__":
    main()
