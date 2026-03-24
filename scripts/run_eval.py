from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset

# Allow running as: python scripts/run_eval.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policyhub.indexing import build_or_load_index
from policyhub.llm import configure_llm
from policyhub.rag import ask
from policyhub.settings import get_settings


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS eval on LvZhengTong")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to JSONL. Each line: {question, ground_truth(optional)}",
    )
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    settings = get_settings()
    configure_llm(settings)
    index = build_or_load_index(settings, rebuild=False)

    rows = load_jsonl(Path(args.dataset))

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[list[str]] = []

    for r in rows:
        q = str(r["question"]).strip()
        gt = r.get("ground_truth")

        res = ask(index=index, question=q, top_k=args.top_k)

        questions.append(q)
        answers.append(res.answer)
        contexts.append([c.get("quote") or "" for c in res.citations])
        if gt is None:
            ground_truths.append([])
        else:
            ground_truths.append([str(gt)])

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truths": ground_truths,
        }
    )

    # 延迟导入：避免未安装 eval 依赖时影响其他功能
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
    print(result)


if __name__ == "__main__":
    main()
