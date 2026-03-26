from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter

# Allow running as: python scripts/build_index.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policyhub.indexing import build_or_load_index
from policyhub.llm import configure_llm
from policyhub.documents import load_policy_documents
from policyhub.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/refresh GreenPolicyHub index")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear the Chroma collection and rebuild embeddings from RAW_DOCS_DIR",
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Print chunking + embedding stats and previews (useful for screenshots).",
    )
    parser.add_argument(
        "--preview-n",
        type=int,
        default=3,
        help="How many chunk/node previews to print when --show-stats is enabled.",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_llm(settings)

    if bool(args.show_stats):
        # Resolve embedding device the same way as policyhub.indexing
        device = (settings.embedding_device or "auto").strip().lower()
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        print("=== Build/Index Stats ===")
        print(f"RAW_DOCS_DIR: {settings.raw_docs_dir}")
        print(f"CHROMA_PERSIST_DIR: {settings.chroma_persist_dir}")
        print(f"CHROMA_COLLECTION: {settings.chroma_collection}")
        print(f"EMBEDDING_MODEL_NAME: {settings.embedding_model_name}")
        print(f"EMBEDDING_DEVICE: {device} (configured={settings.embedding_device})")
        print(f"REBUILD: {bool(args.rebuild)}")

        docs = load_policy_documents(settings.raw_docs_dir)
        print(f"Loaded documents (pre-node chunking): {len(docs)}")

        preview_n = max(0, int(args.preview_n))
        if preview_n > 0 and docs:
            print("\n--- Chunk previews (Document-level) ---")
            for i, d in enumerate(docs[:preview_n], start=1):
                md = d.metadata or {}
                heading_path = md.get("heading_path")
                article = md.get("article")
                title = md.get("title")
                print(f"[{i}] title={title} article={article} heading_path={heading_path}")
                txt = (d.text or "").strip().replace("\n", " ")
                print(txt[:260] + ("..." if len(txt) > 260 else ""))

        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=120)
        nodes = splitter.get_nodes_from_documents(docs) if docs else []
        print(f"\nFinal nodes after SentenceSplitter: {len(nodes)}")

        if preview_n > 0 and nodes:
            print("\n--- Node previews (post-split) ---")
            for i, n in enumerate(nodes[:preview_n], start=1):
                md = n.metadata or {}
                heading_path = md.get("heading_path")
                article = md.get("article")
                title = md.get("title")
                print(f"[{i}] title={title} article={article} heading_path={heading_path}")
                txt = (n.get_content() or "").strip().replace("\n", " ")
                print(txt[:260] + ("..." if len(txt) > 260 else ""))
        print("=== End Stats ===\n")

    _ = build_or_load_index(settings, rebuild=bool(args.rebuild))
    print("OK: index built/loaded.")


if __name__ == "__main__":
    main()
