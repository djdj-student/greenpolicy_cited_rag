from __future__ import annotations

from pathlib import Path

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from policyhub.documents import load_policy_documents
from policyhub.settings import PolicyHubSettings


def build_or_load_index(settings: PolicyHubSettings, rebuild: bool = False) -> VectorStoreIndex:
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)

    # Embedding
    device = (settings.embedding_device or "auto").strip().lower()
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    Settings.embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model_name, device=device)

    # Chunking
    # 说明：文档侧已按“第X条”切分；这里主要负责处理超长条文的二次切分。
    Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=120)

    # Vector store (Chroma)
    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    if rebuild:
        try:
            chroma_client.delete_collection(settings.chroma_collection)
        except Exception:
            # ignore if collection does not exist
            pass
    chroma_collection = chroma_client.get_or_create_collection(settings.chroma_collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 快路径：已有向量且不要求重建时，直接从向量库加载（避免每次启动都重新 embedding 全量文档）
    try:
        existing = int(chroma_collection.count())
    except Exception:
        existing = 0

    if (not rebuild) and existing > 0:
        return VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)

    # 慢路径：清库后或首次建库 → 读取原文并生成 embeddings
    docs = load_policy_documents(settings.raw_docs_dir)
    if not docs:
        return VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)

    return VectorStoreIndex.from_documents(docs, storage_context=storage_context, show_progress=True)
