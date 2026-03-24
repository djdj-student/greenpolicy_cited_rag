from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from policyhub.documents import list_versions


@dataclass(frozen=True)
class DiffSummary:
    added: int
    removed: int
    changed: int


def _read_text(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    # 保留行粒度，方便高亮
    return text.splitlines(keepends=False)


def unified_diff(old_path: Path, new_path: Path, context: int = 3) -> str:
    old_lines = _read_text(old_path)
    new_lines = _read_text(new_path)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=old_path.name,
        tofile=new_path.name,
        n=context,
        lineterm="",
    )
    return "\n".join(diff)


def diff_summary(old_path: Path, new_path: Path) -> DiffSummary:
    old_lines = _read_text(old_path)
    new_lines = _read_text(new_path)
    sm = difflib.SequenceMatcher(a=old_lines, b=new_lines)

    added = removed = changed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "insert":
            added += (j2 - j1)
        elif tag == "delete":
            removed += (i2 - i1)
        elif tag == "replace":
            changed += max(i2 - i1, j2 - j1)
    return DiffSummary(added=added, removed=removed, changed=changed)


_DATE_RE = re.compile(r"(19|20)\d{2}-\d{2}-\d{2}")


def _date_from_filename(path: Path) -> datetime | None:
    m = _DATE_RE.search(path.stem)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(0), "%Y-%m-%d")
    except ValueError:
        return None


def list_policies(raw_dir: Path) -> list[str]:
    if not raw_dir.exists():
        return []
    policy_ids: set[str] = set()
    for p in raw_dir.rglob("*"):
        if p.is_dir():
            continue
        if p.suffix.lower() not in {".txt", ".md"}:
            continue
        policy_ids.add(p.stem.split("_")[0])
    return sorted(policy_ids)


def list_policy_versions(raw_dir: Path, policy_id: str) -> list[Path]:
    versions = list_versions(raw_dir, policy_id)
    # 优先按日期排序
    versions.sort(key=lambda p: _date_from_filename(p) or datetime.min)
    return versions


def changes_after(raw_dir: Path, since: str) -> list[dict]:
    """汇总某日期后的变更（按同一 policy_id 的相邻版本对比）。

    since: YYYY-MM-DD
    """
    since_dt = datetime.strptime(since, "%Y-%m-%d")

    results: list[dict] = []
    for pid in list_policies(raw_dir):
        versions = list_policy_versions(raw_dir, pid)
        for old_p, new_p in zip(versions, versions[1:]):
            new_dt = _date_from_filename(new_p)
            if not new_dt or new_dt < since_dt:
                continue
            summ = diff_summary(old_p, new_p)
            results.append(
                {
                    "policy_id": pid,
                    "old": old_p.name,
                    "new": new_p.name,
                    "new_date": new_dt.strftime("%Y-%m-%d"),
                    "added": summ.added,
                    "removed": summ.removed,
                    "changed": summ.changed,
                }
            )
    # 按日期倒序
    results.sort(key=lambda x: x["new_date"], reverse=True)
    return results
