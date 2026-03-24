from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from llama_index.core import Document

_DATE_RE = re.compile(r"(19|20)\d{2}-\d{2}-\d{2}")
_HEADING_RE = re.compile(r"^[\ufeff\u200b\s]*(#{1,4})\s*(.+?)\s*$")
# e.g. "第一百三十七条" / "第二百零一条" / "第12条"
_ARTICLE_RE = re.compile(r"^(第[一二三四五六七八九十百千万亿零〇两0-9]+条)\s*[　 ]*(.*)$")
_PLAIN_HEADING_RE = re.compile(
    r"^(第[一二三四五六七八九十百千万亿零〇两0-9]+编|"
    r"第[一二三四五六七八九十百千万亿零〇两0-9]+篇|"
    r"第[一二三四五六七八九十百千万亿零〇两0-9]+分编|"
    r"第[一二三四五六七八九十百千万亿零〇两0-9]+章|"
    r"第[一二三四五六七八九十百千万亿零〇两0-9]+节)\s*[　 ]*(.*)$"
)


@dataclass(frozen=True)
class PolicyDocMeta:
    policy_id: str
    title: str
    source_path: str
    version_label: str | None
    version_date: str | None


def _infer_meta_from_path(path: Path) -> PolicyDocMeta:
    stem = path.stem
    parts = stem.split("_")

    # policy_id: 取第一个字段（不完美，但对批量命名最实用）
    policy_id = parts[0] if parts else stem

    # 日期：匹配 YYYY-MM-DD
    match = _DATE_RE.search(stem)
    version_date = match.group(0) if match else None

    # 标签：优先使用日期前一段，例如 xxx_通过稿_2026-06-30
    version_label = None
    if version_date:
        prefix = stem.split(version_date)[0].rstrip("_")
        prefix_parts = prefix.split("_")
        if len(prefix_parts) >= 2:
            version_label = prefix_parts[-1]

    title = stem

    return PolicyDocMeta(
        policy_id=policy_id,
        title=title,
        source_path=str(path.as_posix()),
        version_label=version_label,
        version_date=version_date,
    )


def load_policy_documents(raw_dir: Path) -> list[Document]:
    if not raw_dir.exists():
        return []

    docs: list[Document] = []
    for path in sorted(raw_dir.rglob("*")):
        if path.is_dir():
            continue
        if path.suffix.lower() not in {".txt", ".md"}:
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        # 防止演示占位文档污染真实索引
        if "占位示例文档" in text or "非真实法条" in text:
            continue
        meta = _infer_meta_from_path(path)

        # 若是法典/法规类 Markdown（存在大量“第X条”结构），按条切分更适合 RAG 引用与检索。
        # 这里按“行”统计命中次数，避免仅因出现“条例/条款”等字眼就误触发。
        article_hits = 0
        if path.suffix.lower() == ".md":
            for raw in text.splitlines():
                stripped = _normalize_line(_strip_md_prefix(raw))
                if not stripped:
                    continue
                if _ARTICLE_RE.match(stripped):
                    article_hits += 1
                    if article_hits >= 5:
                        break

        if path.suffix.lower() == ".md" and article_hits >= 5:
            split_docs = split_legal_markdown(
                text,
                base_metadata={
                    "policy_id": meta.policy_id,
                    "title": meta.title,
                    "source_path": meta.source_path,
                    "version_label": meta.version_label,
                    "version_date": meta.version_date,
                },
            )
            if split_docs:
                docs.extend(split_docs)
                continue

        # fallback：无法识别结构时按全文入库
        docs.append(
            Document(
                text=_normalize_line(text),
                metadata={
                    "policy_id": meta.policy_id,
                    "title": meta.title,
                    "source_path": meta.source_path,
                    "version_label": meta.version_label,
                    "version_date": meta.version_date,
                },
            )
        )

    return docs


def _normalize_line(s: str) -> str:
    # 去掉用于排版的“留空”标记，避免影响检索
    return (
        s.replace("\ufeff", "")
        .replace("\u200b", "")
        .replace("（留空）", "")
        .replace("(留空)", "")
        .strip()
    )


def _heading_text(line: str) -> tuple[int, str] | None:
    m = _HEADING_RE.match(line)
    if not m:
        return None
    level = len(m.group(1))
    title = _normalize_line(m.group(2))
    return level, title


def _strip_md_prefix(line: str) -> str:
    # remove leading markdown markers like '##### '
    return re.sub(r"^[\ufeff\u200b\s]*#{1,6}\s*", "", line).strip()


def split_legal_markdown(text: str, base_metadata: dict) -> list[Document]:
    """按“编/分编/章/节 + 第X条”切分法典/法规。

    返回：每条一个 Document（条文过长时仍交给下游 node_parser 再切分）。
    """

    lines = text.splitlines()

    ctx_part: str | None = None
    ctx_subpart: str | None = None
    ctx_chapter: str | None = None
    ctx_section: str | None = None

    docs: list[Document] = []

    current_article: str | None = None
    current_article_title: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_article, current_article_title, current_lines
        if not current_article:
            current_article = None
            current_article_title = None
            current_lines = []
            return

        heading_parts = [p for p in [ctx_part, ctx_subpart, ctx_chapter, ctx_section] if p]
        heading_path = " / ".join(heading_parts)

        # 只把“条文原文”写入 text，避免把 heading_path 这类非原文信息混入引用。
        # 结构信息保留在 metadata 里用于 UI 展示与核验。
        article_line = f"{current_article}{(' ' + current_article_title) if current_article_title else ''}".strip()
        chunk_text = "\n".join([article_line, *current_lines]).strip()

        md = dict(base_metadata)
        md.update(
            {
                "part": ctx_part,
                "subpart": ctx_subpart,
                "chapter": ctx_chapter,
                "section": ctx_section,
                "article": current_article,
                "article_title": current_article_title,
                "heading_path": heading_path,
            }
        )
        docs.append(Document(text=chunk_text, metadata=md))

        current_article = None
        current_article_title = None
        current_lines = []

    found_any_article = False

    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue

        # Normalize once for matching
        stripped = _normalize_line(_strip_md_prefix(line))
        if not stripped:
            continue

        # Article lines: "##### 第一百三十七条 ..." or "（留空）第二百零一条 ..."
        am = _ARTICLE_RE.match(stripped)
        if am:
            found_any_article = True
            flush()
            current_article = am.group(1)
            current_article_title = _normalize_line(am.group(2)) or None
            current_lines = []
            continue

        # Headings (markdown)
        hm = _heading_text(_normalize_line(line))
        if hm:
            # 标题行不属于任何条文；若当前正在收集条文内容，先结束它
            flush()
            level, title = hm
            # reset lower levels when entering a higher-level heading
            if level == 1:
                ctx_part = title
                ctx_subpart = None
                ctx_chapter = None
                ctx_section = None
            elif level == 2:
                ctx_subpart = title
                ctx_chapter = None
                ctx_section = None
            elif level == 3:
                ctx_chapter = title
                ctx_section = None
            elif level == 4:
                ctx_section = title
            continue

        # Headings (plain text, common in法规/法典的转存文本)
        ph = _PLAIN_HEADING_RE.match(stripped)
        if ph:
            flush()
            head_kind = _normalize_line(ph.group(1))
            head_title = _normalize_line(ph.group(2))
            full = (head_kind + ("　" + head_title if head_title else "")).strip()

            if head_kind.endswith("编") or head_kind.endswith("篇"):
                ctx_part = full
                ctx_subpart = None
                ctx_chapter = None
                ctx_section = None
            elif head_kind.endswith("分编"):
                ctx_subpart = full
                ctx_chapter = None
                ctx_section = None
            elif head_kind.endswith("章"):
                ctx_chapter = full
                ctx_section = None
            elif head_kind.endswith("节"):
                ctx_section = full
            continue

        # Content lines
        if current_article:
            current_lines.append(stripped)

    flush()

    # If we didn't find any structured articles, return empty to let caller fallback
    if not found_any_article:
        return []
    return docs


def list_versions(raw_dir: Path, policy_id: str) -> list[Path]:
    if not raw_dir.exists():
        return []
    candidates = []
    for path in sorted(raw_dir.rglob("*")):
        if path.is_dir():
            continue
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        if path.stem.split("_")[0] == policy_id:
            candidates.append(path)
    return candidates
