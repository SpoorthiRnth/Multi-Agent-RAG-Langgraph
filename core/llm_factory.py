"""
This is mainly for enabling a switchable LLM provider.

Supports:
  Claude (Anthropic API) via langchain-anthropic
  Ollama (local) via langchain-ollama

Switch via LLM_PROVIDER env var: "claude" and "ollama"
"""

import logging
from functools import lru_cache
from langchain_core.language_models import BaseChatModel
from core.config import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    #Returns a LangChain compatible chat model based on LLM_PROVIDER config.

    provider = config.llm.provider.lower()
    logger.info(f"Initializing LLM provider: {provider}")

    if provider == "claude":
        return _build_claude()
    elif provider == "ollama":
        return _build_ollama()
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            "Valid options: 'claude', 'ollama'"
        )


def _build_claude() -> BaseChatModel:
    # Anthropic Claude via API.
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError("Run: pip install langchain-anthropic")

    if not config.llm.anthropic_api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file or switch to LLM_PROVIDER=ollama"
        )

    logger.info(f"Using Claude model: {config.llm.claude_model}")
    return ChatAnthropic(
        model=config.llm.claude_model,
        api_key=config.llm.anthropic_api_key,
        temperature=0.0,
        max_tokens=2048,
    )


def _build_ollama() -> BaseChatModel:
    # Local Ollama inference.
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError("Run: pip install langchain-ollama")

    logger.info(
        f"Using Ollama model: {config.llm.ollama_model} "
        f"at {config.llm.ollama_base_url}"
    )
    return ChatOllama(
        model=config.llm.ollama_model,
        base_url=config.llm.ollama_base_url,
        temperature=0.0,
    )


def get_provider_name() -> str:
    # Human-readable provider name for display.
    provider = config.llm.provider.lower()
    if provider == "claude":
        return f"Claude ({config.llm.claude_model})"
    elif provider == "ollama":
        return f"Ollama / {config.llm.ollama_model}"
    return provider
