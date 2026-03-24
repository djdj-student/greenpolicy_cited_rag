from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")


def read_jsonl_tail(path: Path, n: int = 50) -> list[dict[str, Any]]:
    if n <= 0:
        return []
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    return rows[-n:]


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    c = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c += 1
    return c
