from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

THINKING_RE = re.compile(r"<thinking>.*?</thinking>\s*", re.DOTALL)


def parse_json_maybe(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def inline_tool_json(content: str) -> str | None:
    decoder = json.JSONDecoder()
    for start in [i for i, c in enumerate(content) if c == "{"]:
        try:
            value, _ = decoder.raw_decode(content[start:])
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, dict)
            and isinstance(value.get("name"), str)
            and isinstance(value.get("arguments"), dict)
        ):
            return json.dumps(value, ensure_ascii=False)
    return None


def tool_call_content(msg: dict[str, Any]) -> str | None:
    calls = msg.get("tool_calls") or []
    if not calls:
        return None
    if len(calls) != 1:
        raise ValueError("expected one assistant tool call per turn after expansion")
    call = calls[0]
    fn = call.get("function") if isinstance(call, dict) else None
    name = call.get("name") or (fn or {}).get("name")
    args = parse_json_maybe(call.get("arguments") or (fn or {}).get("arguments") or {})
    if not isinstance(args, dict):
        raise ValueError("tool-call arguments must parse to an object")
    return json.dumps({"name": name, "arguments": args}, ensure_ascii=False)


def expand_parallel_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        calls = msg.get("tool_calls") or []
        if msg.get("role") == "assistant" and isinstance(calls, list) and len(calls) > 1:
            n = len(calls)
            for k in range(n):
                if i + 1 + k >= len(messages) or messages[i + 1 + k].get("role") != "tool":
                    raise ValueError("malformed parallel tool calls")
            for j in range(n):
                split = dict(msg)
                split["tool_calls"] = [calls[j]]
                out.append(split)
                out.append(messages[i + 1 + j])
            i += 1 + n
            continue
        out.append(msg)
        i += 1
    return out


def normalize(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]]]:
    system_parts: list[str] = []
    convo: list[dict[str, str]] = []
    saw_tool = False

    for msg in expand_parallel_tool_calls(list(messages)):
        role = msg.get("role")
        content = msg.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        content = THINKING_RE.sub("", content).strip()

        if role == "system":
            if content:
                system_parts.append(content)
        elif role == "user":
            if content:
                convo.append({"from": "human", "value": content})
        elif role == "assistant":
            tc = tool_call_content(msg) or inline_tool_json(content)
            if tc:
                content = tc
                saw_tool = True
            if content:
                convo.append({"from": "gpt", "value": content})
        elif role == "tool":
            name = msg.get("name")
            convo.append({"from": "observation", "value": (f"{name}: " if name else "") + content})

    while convo and convo[0]["from"] == "gpt":
        convo.pop(0)
    while convo:
        while convo and convo[-1]["from"] != "gpt":
            convo.pop()
        if not convo:
            break
        last = str(convo[-1]["value"])
        if '"name": "done"' in last or '"name":"done"' in last:
            convo.pop()
            continue
        break

    if not convo:
        raise ValueError("no trainable turns")
    if not saw_tool and not any(c["from"] == "observation" for c in convo):
        raise ValueError("no tool calls or observations")
    return "\n\n".join(system_parts), convo


def iter_episodes(paths: list[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path, required=True,
                    help="One or more raw trajectory JSONL files (will be merged + shuffled).")
    ap.add_argument("--domain", default="retail", choices=("retail", "airline", "retail_airline"))
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    allowed = {"retail", "airline"} if args.domain == "retail_airline" else {args.domain}

    rows: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    for ep in iter_episodes(args.inputs):
        d = str(ep.get("domain", ""))
        if d and d not in allowed:
            skipped["wrong_domain"] += 1
            continue
        if float(ep.get("reward", 0.0)) < 1.0:
            skipped["non_passing_reward"] += 1
            continue
        try:
            system, convo = normalize(ep["messages"])
        except Exception as exc:
            skipped[f"invalid:{type(exc).__name__}"] += 1
            continue
        rows.append({
            "system": system,
            "conversations": convo,
            "tools": ep.get("tools") or None,
            "source": {"task_id": ep.get("task_id"), "domain": d, "reward": ep.get("reward")},
        })

    random.Random(args.seed).shuffle(rows)
    n_val = int(round(len(rows) * args.val_fraction))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train_rows), ("val", val_rows)):
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "inputs": [str(p) for p in args.inputs],
        "domain": args.domain,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "skipped": dict(skipped),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
