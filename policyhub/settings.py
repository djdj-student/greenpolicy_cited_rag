from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path:
    # policyhub/ is a top-level package under the repo root
    return Path(__file__).resolve().parents[1]


def _resolve_dir(value: str, default_rel: str) -> Path:
    raw = (value or default_rel).strip()
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    return (_project_root() / p).resolve()


@dataclass(frozen=True)
class PolicyHubSettings:
    llm_provider: str

    openai_api_key: str | None
    openai_model: str
    openai_base_url: str | None

    ollama_base_url: str
    ollama_model: str
    ollama_request_timeout: float
    ollama_keep_alive: str | float | None

    deepseek_api_key: str | None
    deepseek_base_url: str
    deepseek_model: str

    embedding_model_name: str
    embedding_device: str

    chroma_persist_dir: Path
    chroma_collection: str

    raw_docs_dir: Path
    index_dir: Path


def get_settings() -> PolicyHubSettings:
    load_dotenv(override=False)

    llm_provider = os.getenv("POLICYHUB_LLM_PROVIDER", "ollama").strip().lower()

    openai_api_key = os.getenv("OPENAI_API_KEY") or None
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_base_url = openai_base_url.strip() if openai_base_url else None

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    # 默认选择更容易在低内存机器上跑通的模型；用户可在 .env 中自行覆盖。
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct").strip()

    # Ollama 默认 request_timeout=30s 在冷启动/长回答时容易超时
    try:
        ollama_request_timeout = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "120").strip())
    except ValueError:
        ollama_request_timeout = 120.0

    # 让模型在一段时间内保持常驻，减少下一次请求冷启动
    keep_alive_raw = (os.getenv("OLLAMA_KEEP_ALIVE") or "5m").strip()
    if keep_alive_raw.lower() in {"none", "null", ""}:
        ollama_keep_alive = None
    else:
        try:
            ollama_keep_alive = float(keep_alive_raw)
        except ValueError:
            ollama_keep_alive = keep_alive_raw

    # DeepSeek（OpenAI 兼容接口）
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or None
    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip()
    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()

    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3").strip()
    embedding_device = os.getenv("EMBEDDING_DEVICE", "auto").strip().lower()

    chroma_persist_dir = _resolve_dir(os.getenv("CHROMA_PERSIST_DIR", ""), "./data/chroma")
    chroma_collection = os.getenv("CHROMA_COLLECTION", "green_policyhub").strip()

    raw_docs_dir = _resolve_dir(os.getenv("RAW_DOCS_DIR", ""), "./data/raw")
    index_dir = _resolve_dir(os.getenv("INDEX_DIR", ""), "./data/index")

    return PolicyHubSettings(
        llm_provider=llm_provider,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        ollama_request_timeout=ollama_request_timeout,
        ollama_keep_alive=ollama_keep_alive,

        deepseek_api_key=deepseek_api_key,
        deepseek_base_url=deepseek_base_url,
        deepseek_model=deepseek_model,
        embedding_model_name=embedding_model_name,
        embedding_device=embedding_device,
        chroma_persist_dir=chroma_persist_dir,
        chroma_collection=chroma_collection,
        raw_docs_dir=raw_docs_dir,
        index_dir=index_dir,
    )
