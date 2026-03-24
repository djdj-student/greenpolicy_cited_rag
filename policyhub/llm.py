from __future__ import annotations

from llama_index.core import Settings

from policyhub.settings import PolicyHubSettings


def configure_llm(settings: PolicyHubSettings) -> None:
    provider = settings.llm_provider

    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("POLICYHUB_LLM_PROVIDER=openai 但未设置 OPENAI_API_KEY")
        from llama_index.llms.openai import OpenAI

        Settings.llm = OpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            api_base=settings.openai_base_url,
        )
        return

    if provider == "deepseek":
        if not settings.deepseek_api_key:
            raise RuntimeError("POLICYHUB_LLM_PROVIDER=deepseek 但未设置 DEEPSEEK_API_KEY")
        from llama_index.llms.openai import OpenAI

        Settings.llm = OpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            api_base=settings.deepseek_base_url,
        )
        return

    if provider == "ollama":
        from llama_index.llms.ollama import Ollama

        Settings.llm = Ollama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            request_timeout=settings.ollama_request_timeout,
            keep_alive=settings.ollama_keep_alive,
        )
        return

    raise ValueError(f"未知 LLM Provider: {provider}. 仅支持 openai/ollama/deepseek")
