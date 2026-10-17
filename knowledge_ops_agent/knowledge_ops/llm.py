"""LLM factory. Returns a chat model for a given agent role so roles can be pointed at different models/temperatures (e.g. a separate verifier)."""

from functools import lru_cache

from Langchain_core.language_models import BaseChatModel

from .config import settings

@lru_cache(maxsize=None)
def get_llm(role: str = "default", temperature: float = 0.0) -> BaseChatModel:
    if settings.11m_provider = "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=settings.azure_deployment,
            api_version=settings.azure_api_version,
            temperature=temperature,
        )
    from langchain_ollama import Chatollama
    return Chatollama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        format="json",
    )

def structured (role: str, schema, temperature: float = 0.0):
    """Chat model forced to return schema (a Pydantic class)."""
    return get_llm(role, temperature).with_structured_output(schema)