from __future__ import annotations

import base64
import html
from pathlib import Path

import streamlit as st

from policyhub.indexing import build_or_load_index
from policyhub.llm import configure_llm
from policyhub.settings import get_settings


@st.cache_data(show_spinner=False)
def _image_b64(image_path: str) -> str:
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


@st.cache_resource(show_spinner=False)
def _get_index_cached():
    # 只在首次提问时初始化（embedding/索引加载较慢）。
    settings = get_settings()
    configure_llm(settings)
    return build_or_load_index(settings)


def _inject_background_image(image_path: Path) -> None:
    bg_line = "background-image: none;"
    if not image_path.exists():
        st.warning(f"未找到背景图：{image_path.as_posix()}（将继续运行，但不显示背景图）")
    else:
        try:
            b64 = _image_b64(image_path.as_posix())
            # Streamlit 的静态资源默认不会自动暴露；用 data URL 最稳妥。
            bg_line = f"background-image: url(\"data:image/jpeg;base64,{b64}\");"
        except Exception as e:
            st.warning(f"背景图加载失败（将继续运行）：{type(e).__name__}: {e}")

    st.markdown(
        f"""
<style>
.stApp {{
  {bg_line}
  background-size: cover;
  background-repeat: no-repeat;
  background-attachment: fixed;
  background-position: center;
}}

/* 隐藏侧边栏（本 Demo 仅保留问答页） */
section[data-testid="stSidebar"] {{
    display: none;
}}

/* 顶部条栏透明化，让背景图覆盖到顶部 */
header[data-testid="stHeader"],
.stAppHeader {{
    background: transparent !important;
}}
div[data-testid="stToolbar"] {{
    background: transparent !important;
}}

/* 主内容容器更贴顶，避免形成“白条”观感 */
section.main > div.block-container {{
    padding-top: 1.2rem;
}}

/* 标题：黑色粗体居中 */
.gh-title {{
    color: #000;
    font-weight: 800;
    text-align: center;
    margin: 0.2rem 0 1.0rem 0;
}}

/* 中部白框（提问/答案） */
.gh-card {{
    max-width: 640px;
    width: min(640px, 92vw);
    margin: 0 auto 1rem auto;
    background: #fff;
    border-radius: 14px;
    padding: 14px 16px;
}}
.gh-card * {{
    color: #000;
}}

/* 回答/引用：白色玻璃卡片（不改变背景图，只提升可读性） */
.gh-glass {{
    max-width: 640px;
    width: min(640px, 92vw);
    margin: 0 auto 1rem auto;
    background: rgba(255, 255, 255, 0.70);
    border: 1px solid rgba(0, 0, 0, 0.06);
    border-radius: 14px;
    padding: 14px 16px;
}}
.gh-glass * {{
    color: #000;
}}

.gh-card-title {{
    font-weight: 800;
    margin: 0 0 0.5rem 0;
}}
.gh-pre {{
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.6;
}}

/* 输入框背景统一为纯白（避免默认灰底导致与答案白框不一致） */
.stTextArea textarea,
.stTextInput input {{
    background-color: #fff !important;
    color: #000 !important;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def _render_citations(citations: list[dict], title: str = "引用原文") -> None:
    if not citations:
        return
    st.subheader(title)
    for c in citations:
        header = f"[{c.get('i')}] {c.get('title') or ''}"
        meta = " | ".join(
            [
                x
                for x in [
                    c.get("policy_id"),
                    c.get("version_label"),
                    c.get("version_date"),
                    c.get("heading_path"),
                    c.get("article"),
                    c.get("source_path"),
                    f"score={c.get('score'):.3f}" if c.get("score") is not None else None,
                ]
                if x
            ]
        )
        with st.expander(header):
            if meta:
                st.caption(meta)
            st.code(c.get("quote") or "", language="text")


def _render_answer_card(title: str, text: str, *, glass: bool = True) -> None:
        klass = "gh-glass" if glass else "gh-card"
        safe_title = html.escape(title or "")
        safe_text = html.escape(text or "")
        st.markdown(
                f"""
<div class="{klass}">
    <div class="gh-card-title">{safe_title}</div>
    <div class="gh-pre">{safe_text}</div>
</div>
""",
                unsafe_allow_html=True,
        )


def _render_agent_process(diagnostics: dict | None) -> None:
    diagnostics = diagnostics or {}
    retrieval = diagnostics.get("retrieval") or {}
    answer_diag = diagnostics.get("answer") or {}
    trace = diagnostics.get("trace") or []

    def _esc(v) -> str:
        return html.escape("" if v is None else str(v))

    # 过程轨迹
    trace_items: list[str] = []
    for t in trace or []:
        step = t.get("step")
        if step == "retrieve":
            trace_items.append(
                f"<li>检索：top_k={_esc(t.get('top_k'))} 命中={_esc(t.get('n'))} 用时={_esc(t.get('t_s'))}s</li>"
            )
        elif step == "generate":
            trace_items.append(
                f"<li>生成：strong={_esc(t.get('strong'))} 字数={_esc(t.get('answer_len'))} 用时={_esc(t.get('t_s'))}s</li>"
            )
        elif step == "self_check":
            ok = t.get("ok")
            if ok:
                trace_items.append(f"<li>自检：通过 refs={_esc(t.get('refs'))}</li>")
            else:
                trace_items.append(
                    f"<li>自检：未通过 reason={_esc(t.get('reason'))} refs={_esc(t.get('refs'))}</li>"
                )
        elif step == "retry_done":
            trace_items.append(f"<li>重试：完成 refs={_esc(t.get('refs'))}</li>")
        else:
            trace_items.append(f"<li>{_esc(step)}: {_esc(t)}</li>")

    trace_html = "<ul>" + "".join(trace_items) + "</ul>" if trace_items else "<div class='gh-pre'>（无 trace）</div>"

    st.markdown(
        f"""
<div class="gh-glass">
  <div class="gh-card-title">Agent 思考过程（可复盘）</div>
  <ul>
    <li>输出可审计：答案强制使用编号引用标注 [1][2]...</li>
    <li>自检与重试：引用不足时会自动扩召回并重写一次</li>
    <li>量化诊断：展示召回规模/分数统计/引用覆盖率/耗时轨迹</li>
  </ul>

  <div class="gh-card-title" style="font-weight:700;">量化诊断（程序计算）</div>
  <ul>
    <li>context 片段数：{_esc(retrieval.get('n_context'))}</li>
    <li>命中来源数：{_esc(retrieval.get('unique_sources'))}</li>
    <li>命中文章数：{_esc(retrieval.get('unique_articles'))}</li>
    <li>相似度：max={_esc(retrieval.get('score_max'))} mean={_esc(retrieval.get('score_mean'))} min={_esc(retrieval.get('score_min'))}</li>
    <li>引用覆盖率：{_esc(answer_diag.get('citation_ref_coverage'))}</li>
  </ul>

  <div class="gh-card-title" style="font-weight:700;">过程轨迹（trace）</div>
  {trace_html}
</div>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="GreenPolicyHub", layout="centered", initial_sidebar_state="collapsed")
    _inject_background_image(Path(__file__).parent / "static" / "bg.jpg")
    st.markdown('<h1 class="gh-title">GreenPolicyHub — 环保政策知识库</h1>', unsafe_allow_html=True)
    # 注意：索引/embedding 初始化会慢（尤其首次下载 bge-m3），这里延迟到用户点击“提问”再做。

    settings = get_settings()

    st.markdown('<div class="gh-card">', unsafe_allow_html=True)
    q = st.text_area("输入问题", placeholder="例如：排污许可证有效期一般是多久？", height=120)
    c1, c2 = st.columns(2)
    with c1:
        ask_rag_clicked = st.button("直接 RAG 提问", key="ask_rag", use_container_width=True)
    with c2:
        ask_agent_clicked = st.button("Agent 提问", key="ask_agent", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if ask_rag_clicked or ask_agent_clicked:
        if not (q or "").strip():
            st.warning("请输入问题")
            return

        index = st.session_state.get("index")
        if index is None:
            try:
                with st.spinner("首次初始化索引/Embedding（可能需要 30-180 秒）..."):
                    index = _get_index_cached()
                    st.session_state["index"] = index
            except Exception as e:
                st.error("索引/Embedding 初始化失败。")
                st.caption(f"错误详情：{type(e).__name__}: {e}")
                return

        if ask_rag_clicked:
            try:
                from policyhub.rag import ask

                with st.spinner("检索与生成中（直接 RAG）..."):
                    res = ask(index=index, question=q)

                _render_answer_card("直接 RAG", res.answer, glass=True)
                _render_citations(res.citations, title="引用原文（直接 RAG）")
            except Exception as e:
                st.error(
                    "直接 RAG 失败（常见原因：Ollama 冷启动或生成太慢导致超时）。\n"
                    "你可以：1) 稍等 10-30 秒再试；2) 在 .env 提高 OLLAMA_REQUEST_TIMEOUT；"
                    "3) 切换 POLICYHUB_LLM_PROVIDER=deepseek。"
                )
                st.caption(f"错误详情：{type(e).__name__}: {e}")

        if ask_agent_clicked:
            try:
                from policyhub.agent import AgentState, build_agent
                from policyhub.records import append_jsonl

                agent = build_agent(index=index, raw_dir=settings.raw_docs_dir)
                with st.spinner("Agent 推理中（自动路由）..."):
                    out = agent.invoke(AgentState(user_input=q))

                if isinstance(out, dict):
                    result_text = out.get("result_text")
                    citations = out.get("citations") or []
                    diagnostics = out.get("diagnostics") or {}
                    run_id = out.get("run_id")
                    run_record = out.get("run_record") or {}
                else:
                    result_text = getattr(out, "result_text", None)
                    citations = getattr(out, "citations", None) or []
                    diagnostics = getattr(out, "diagnostics", None) or {}
                    run_id = getattr(out, "run_id", None)
                    run_record = getattr(out, "run_record", None) or {}

                # 先展示 Agent 的全过程，再展示答案
                _render_agent_process(diagnostics)

                _render_answer_card("Agent 答案（带编号引用）", result_text or "", glass=True)
                _render_citations(citations, title="引用原文（Agent）")

                with st.expander("记录与导出（bad case / 完整诊断 JSON）", expanded=False):
                    if run_id:
                        st.caption(f"run_id: {run_id}")

                    run_log = (diagnostics or {}).get("run_log") or {}
                    if run_log:
                        st.caption(f"Agent 运行日志：{run_log.get('path')}（累计 {run_log.get('count')} 条）")

                    st.markdown("**完整 diagnostics（便于你复盘/对比）**")
                    st.json(diagnostics)

                    note = st.text_input("bad case 备注（可选）", key=f"badcase_note_{run_id or 'na'}")
                    if st.button("标记为 bad case（写入 data/badcases/badcases.jsonl）", key=f"mark_badcase_{run_id or 'na'}"):
                        badcase_path = Path("./data/badcases/badcases.jsonl").resolve()
                        payload = {
                            "ts_utc": __import__("datetime").datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                            "run_id": run_id,
                            "note": note,
                            "run_record": run_record,
                        }
                        append_jsonl(badcase_path, payload)
                        st.success(f"已记录 bad case：{badcase_path}")
                        st.caption("建议：后续你调整 top_k / prompt / 切分策略后，用同一问题再跑一次，对照 trace 里是否减少了重试/是否提升引用覆盖率。")
            except Exception as e:
                st.error(
                    "Agent 调用失败（常见原因：Ollama 冷启动或生成太慢导致超时）。\n"
                    "你可以：1) 稍等 10-30 秒再试；2) 在 .env 提高 OLLAMA_REQUEST_TIMEOUT；"
                    "3) 切换 POLICYHUB_LLM_PROVIDER=deepseek。"
                )
                st.caption(f"错误详情：{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
