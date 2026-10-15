"""Chat model factory. Returns ``None`` for the offline (extractive, no-LLM) mode."""

from __future__ import annotations

from typing import Optional

from .config import Settings
from .privacy import enforce_privacy_mode

_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
    "ollama": "llama3.1",
}

_INSTALL_HINTS = {
    "openai": "pip install langchain-openai  (and set OPENAI_API_KEY)",
    "anthropic": "pip install langchain-anthropic  (and set ANTHROPIC_API_KEY)",
    "ollama": "pip install langchain-ollama  (and run an Ollama server)",
}


class ChatLLM:
    """Thin wrapper giving every LangChain chat model one ``complete(system, user)`` method."""

    def __init__(self, model, name: str) -> None:
        self._model = model
        self.name = name

    def complete(self, system: str, user: str) -> str:
        response = self._model.invoke([("system", system), ("human", user)])
        content = response.content
        if isinstance(content, list):  # some providers return content blocks
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        return str(content).strip()


def get_llm(settings: Settings) -> Optional[ChatLLM]:
    provider = settings.llm_provider.lower().strip()
    if provider in {"", "none", "offline"}:
        return None
    enforce_privacy_mode(settings.privacy_mode, provider)
    model_name = settings.llm_model or _DEFAULT_MODELS.get(provider, "")
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(model=model_name, temperature=0)
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            model = ChatAnthropic(model=model_name, temperature=0)
        elif provider == "ollama":
            from langchain_ollama import ChatOllama

            model = ChatOllama(model=model_name, temperature=0)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")
    except ImportError as exc:
        raise RuntimeError(f"Provider '{provider}' needs an extra package: {_INSTALL_HINTS[provider]}") from exc
    return ChatLLM(model, f"{provider}:{model_name}")
