from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from llama_index.core import Settings

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from policyhub.change_tracking import changes_after, list_policy_versions, unified_diff
from policyhub.rag import QAResult, ask
from policyhub.records import append_jsonl, count_jsonl_lines


class AgentState(BaseModel):
    user_input: str
    mode: str | None = None  # qa | change | compliance
    result_text: str | None = None
    citations: list[dict] | None = None
    diagnostics: dict | None = None
    run_id: str | None = None
    run_record: dict | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _make_run_id(question: str, mode: str) -> str:
    h = hashlib.sha256(f"{mode}|{question}".encode("utf-8")).hexdigest()[:10]
    t = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{t}_{mode}_{h}"


def _extract_citation_refs(text: str) -> set[int]:
    # 解析形如 [1] [2] 的引用标注
    refs: set[int] = set()
    if not text:
        return refs
    i = 0
    while i < len(text):
        if text[i] != "[":
            i += 1
            continue
        j = text.find("]", i + 1)
        if j == -1:
            i += 1
            continue
        inner = text[i + 1 : j].strip()
        if inner.isdigit():
            try:
                refs.add(int(inner))
            except Exception:
                pass
        i = j + 1
    return refs


def _llm_complete(prompt: str) -> str:
    llm = Settings.llm
    if llm is None:
        raise RuntimeError("LLM 未配置，请先调用 configure_llm(settings)")

    # llama-index 的不同 LLM 可能返回 CompletionResponse 或字符串
    try:
        resp = llm.complete(prompt)
        text = getattr(resp, "text", None)
        return (text if text is not None else str(resp)).strip()
    except Exception:
        # 兜底：有些模型只支持 chat
        try:
            from llama_index.core.llms import ChatMessage

            resp = llm.chat([ChatMessage(role="user", content=prompt)])
            message = getattr(resp, "message", None)
            content = getattr(message, "content", None)
            return (content if content is not None else str(resp)).strip()
        except Exception as e:
            raise RuntimeError(f"LLM 调用失败：{type(e).__name__}: {e}")


def _build_numbered_context(citations: list[dict]) -> str:
    lines: list[str] = []
    for c in citations:
        i = c.get("i")
        quote = (c.get("quote") or "").strip()
        if not quote:
            continue

        meta_parts = [
            c.get("title"),
            c.get("heading_path"),
            c.get("article"),
            c.get("version_date"),
            c.get("source_path"),
        ]
        meta = " | ".join([p for p in meta_parts if p])
        if meta:
            lines.append(f"【{i}】{meta}")
        else:
            lines.append(f"【{i}】")
        lines.append(quote)
        lines.append("")
    return "\n".join(lines).strip()


def _agent_answer_with_diagnostics(index, question: str, task_hint: str, data_dir: Path) -> tuple[str, list[dict], dict, dict]:
    trace: list[dict] = []

    def _retrieve(top_k: int) -> tuple[list[dict], list[float]]:
        t0 = time.perf_counter()
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(question)
        t1 = time.perf_counter()

        citations: list[dict] = []
        scores: list[float] = []
        for i, sn in enumerate(nodes or []):
            node = getattr(sn, "node", None)
            meta = (getattr(node, "metadata", None) or {}) if node is not None else {}
            score = float(getattr(sn, "score", 0.0) or 0.0)
            scores.append(score)
            quote = ""
            try:
                quote = (node.get_content() or "") if node is not None else ""
            except Exception:
                quote = ""

            citations.append(
                {
                    "i": i + 1,
                    "score": score,
                    "source_path": meta.get("source_path"),
                    "title": meta.get("title"),
                    "policy_id": meta.get("policy_id"),
                    "version_label": meta.get("version_label"),
                    "version_date": meta.get("version_date"),
                    "heading_path": meta.get("heading_path"),
                    "article": meta.get("article"),
                    "quote": (quote or "")[:900],
                }
            )

        trace.append({"step": "retrieve", "top_k": top_k, "t_s": round(t1 - t0, 4), "n": len(citations)})
        return citations, scores

    def _generate(citations: list[dict], strong: bool) -> str:
        context = _build_numbered_context(citations)
        rules = [
            "你是环保政策知识库的合规/政策解读 Agent。",
            "只能基于【参考资料摘录】回答；若资料不足，请明确输出：资料不足。",
            "必须使用方括号引用标注来源，例如：[1] 或 [2][5]。",
            "每个关键结论都要跟随至少一个引用标注；不要输出没有引用的条款编号/数值/处罚幅度。",
            "严禁编造未出现在参考资料摘录中的条文、编号、日期。",
        ]
        if strong:
            rules.append("如果无法为某个结论找到引用，请删除该结论或改写为‘资料不足’。")

        prompt = "\n".join(
            [
                "\n".join(rules),
                "",
                f"任务：{task_hint}",
                "",
                "【参考资料摘录】",
                context or "（无）",
                "",
                "【问题】",
                question,
                "",
                "【回答（必须带 [n] 引用）】",
            ]
        )

        t0 = time.perf_counter()
        answer = _llm_complete(prompt)
        t1 = time.perf_counter()
        trace.append({"step": "generate", "strong": strong, "t_s": round(t1 - t0, 4), "answer_len": len(answer)})
        return answer

    # 1) 首次召回 + 生成
    top_k_1 = 8
    citations, scores = _retrieve(top_k_1)
    answer = _generate(citations, strong=False)
    refs = _extract_citation_refs(answer)

    # 2) 轻量自检：引用标注不足则扩召回重试（体现 Agent 的“优化过程”）
    need_retry = (len(refs) < 2) or (not refs)
    if need_retry:
        trace.append({"step": "self_check", "ok": False, "reason": "citations_too_few", "refs": sorted(list(refs))})
        top_k_2 = 12
        citations2, scores2 = _retrieve(top_k_2)
        answer2 = _generate(citations2, strong=True)
        refs2 = _extract_citation_refs(answer2)
        # 选择引用覆盖更好的结果
        if len(refs2) >= len(refs):
            citations, scores, answer, refs = citations2, scores2, answer2, refs2
        trace.append({"step": "retry_done", "refs": sorted(list(refs))})
    else:
        trace.append({"step": "self_check", "ok": True, "refs": sorted(list(refs))})

    # 诊断指标（全部可由程序计算，不伪造）
    n_ctx = len(citations)
    unique_sources = len({(c.get("source_path") or c.get("title") or "") for c in citations if (c.get("source_path") or c.get("title"))})
    unique_articles = len({c.get("article") for c in citations if c.get("article")})

    score_max = max(scores) if scores else 0.0
    score_min = min(scores) if scores else 0.0
    score_mean = mean(scores) if scores else 0.0

    ref_unique = len(refs)
    ref_coverage = (ref_unique / n_ctx) if n_ctx else 0.0
    unreferenced = sorted([c.get("i") for c in citations if c.get("i") and (int(c["i"]) not in refs)])

    diagnostics = {
        "retrieval": {
            "n_context": n_ctx,
            "unique_sources": unique_sources,
            "unique_articles": unique_articles,
            "score_max": round(score_max, 4),
            "score_mean": round(score_mean, 4),
            "score_min": round(score_min, 4),
        },
        "answer": {
            "answer_len": len(answer or ""),
            "citation_refs_unique": ref_unique,
            "citation_ref_coverage": round(ref_coverage, 4),
            "unreferenced_context_indices": unreferenced,
        },
        "trace": trace,
    }

    run_id = _make_run_id(question=question, mode="agent")
    run_path = data_dir / "runs" / "agent_runs.jsonl"
    record = {
        "run_id": run_id,
        "ts_utc": _utc_now_iso(),
        "question": question,
        "task_hint": task_hint,
        "diagnostics": diagnostics,
        "citations": citations,
        "answer": answer,
    }
    append_jsonl(run_path, record)

    # 顺便给 UI 一个“累计记录数”，体现过程沉淀
    diagnostics["run_log"] = {"path": str(run_path), "count": count_jsonl_lines(run_path)}

    return answer, citations, diagnostics, record


def _route(state: AgentState) -> AgentState:
    q = (state.user_input or "").strip()
    # 简单可控的规则路由：可后续替换为 LLM Router
    change_kw = ["变更", "对比", "diff", "新增", "删除", "修改", "草案", "通过稿", "修订", "版本", "2026年后", "汇总"]
    compliance_kw = ["合规", "风险", "处罚", "违法", "行政", "执法", "检查", "整改", "豁免"]

    if any(k in q for k in change_kw):
        state.mode = "change"
    elif any(k in q for k in compliance_kw):
        state.mode = "compliance"
    else:
        state.mode = "qa"
    return state


def _qa_node(index, state: AgentState) -> AgentState:
    data_dir = Path("./data").resolve()
    answer, citations, diagnostics, record = _agent_answer_with_diagnostics(
        index=index,
        question=state.user_input,
        task_hint="政策问答：给出条款要点 + 适用条件/例外 + 执行主体。",
        data_dir=data_dir,
    )
    state.result_text = answer
    state.citations = citations
    state.diagnostics = diagnostics
    state.run_id = record.get("run_id")
    state.run_record = record
    return state


def _compliance_node(index, state: AgentState) -> AgentState:
    data_dir = Path("./data").resolve()
    answer, citations, diagnostics, record = _agent_answer_with_diagnostics(
        index=index,
        question=state.user_input,
        task_hint="合规分析：输出风险点、触发条件、建议动作；所有结论都必须带 [n] 引用。",
        data_dir=data_dir,
    )
    state.result_text = answer
    state.citations = citations
    state.diagnostics = diagnostics
    state.run_id = record.get("run_id")
    state.run_record = record
    return state


def _change_node(raw_dir: Path, state: AgentState) -> AgentState:
    q = (state.user_input or "").strip()

    # 支持两类：
    # 1) “2026年后政策变更汇总”
    if "汇总" in q or "2026" in q:
        rows = changes_after(raw_dir, since="2026-01-01")
        if not rows:
            state.result_text = "未在 RAW_DOCS_DIR 中找到可用于汇总的版本对比（请先放入带日期命名的政策文本）。"
            state.citations = []
            return state
        lines = ["2026-01-01 之后的版本变更汇总（相邻版本对比）：", ""]
        for r in rows[:50]:
            lines.append(
                f"- {r['new_date']} | {r['policy_id']} | {r['old']} -> {r['new']} | +{r['added']} -{r['removed']} ~{r['changed']}"
            )
        state.result_text = "\n".join(lines)
        state.citations = []
        return state

    # 2) “对比 某政策 两个版本文件” —— 简化：如果用户输入包含两个文件名，则直接 diff
    # 更复杂的解析（提取政策名/日期）可以后续迭代
    tokens = [t for t in q.replace("，", " ").replace(",", " ").split() if t]
    files = [t for t in tokens if t.endswith(".txt") or t.endswith(".md")]
    if len(files) >= 2:
        old_path = raw_dir / files[0]
        new_path = raw_dir / files[1]
        if not old_path.exists() or not new_path.exists():
            state.result_text = "未找到你指定的版本文件。请确认文件在 data/raw/ 下，并把文件名完整贴出来。"
            state.citations = []
            return state
        state.result_text = unified_diff(old_path, new_path)
        state.citations = []
        return state

    # 兜底：引导用户选择政策与版本
    state.result_text = (
        "我可以做政策版本对比，但目前问题里没有明确两个版本。\n"
        "请用以下任一方式再问一次：\n"
        "1) 直接贴两个文件名：对比 A.txt B.txt\n"
        "2) 说‘2026年后政策变更汇总’（需要文件名含 YYYY-MM-DD）"
    )
    state.citations = []
    return state


def build_agent(index, raw_dir: Path):
    g = StateGraph(AgentState)
    g.add_node("route", lambda s: _route(s))
    g.add_node("qa", lambda s: _qa_node(index, s))
    g.add_node("compliance", lambda s: _compliance_node(index, s))
    g.add_node("change", lambda s: _change_node(raw_dir, s))

    def _branch(state: AgentState) -> str:
        return state.mode or "qa"

    g.set_entry_point("route")
    g.add_conditional_edges("route", _branch, {"qa": "qa", "change": "change", "compliance": "compliance"})
    g.add_edge("qa", END)
    g.add_edge("change", END)
    g.add_edge("compliance", END)
    return g.compile()
