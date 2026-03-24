from __future__ import annotations

from dataclasses import dataclass

from llama_index.core import PromptTemplate
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine


_QA_PROMPT = PromptTemplate(
    """你是环保政策知识库助手。只能基于【参考资料】回答；若资料不足，请明确说“资料不足”。

要求：
1) 回答必须包含可核验的引用片段（来自参考资料原文），不得编造条款或编号。
2) 尽量给出条款要点 + 适用条件/例外 + 执行主体。

【参考资料】
{context_str}

【问题】
{query_str}

【回答】
"""
)


@dataclass(frozen=True)
class QAResult:
    answer: str
    citations: list[dict]


def make_query_engine(index: VectorStoreIndex, top_k: int = 6) -> RetrieverQueryEngine:
    retriever = index.as_retriever(similarity_top_k=top_k)
    synthesizer = CompactAndRefine(text_qa_template=_QA_PROMPT)
    return RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesizer)


def ask(index: VectorStoreIndex, question: str, top_k: int = 6) -> QAResult:
    engine = make_query_engine(index=index, top_k=top_k)
    response = engine.query(question)

    citations: list[dict] = []
    for i, sn in enumerate(getattr(response, "source_nodes", []) or []):
        node = sn.node
        meta = node.metadata or {}
        citations.append(
            {
                "i": i + 1,
                "score": float(getattr(sn, "score", 0.0) or 0.0),
                "source_path": meta.get("source_path"),
                "title": meta.get("title"),
                "policy_id": meta.get("policy_id"),
                "version_label": meta.get("version_label"),
                "version_date": meta.get("version_date"),
                "heading_path": meta.get("heading_path"),
                "article": meta.get("article"),
                "quote": (node.get_content() or "")[:800],
            }
        )

    return QAResult(answer=str(response), citations=citations)
