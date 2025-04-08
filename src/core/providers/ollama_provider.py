from .base_provider import BaseProvider
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from config.settings import config


class OllamaProvider(BaseProvider):
    """Provider for Ollama models."""

    @classmethod
    def supports(cls, model_name: str) -> bool:
        return model_name.startswith("llama")

    def create_llm(self, **kwargs) -> BaseLanguageModel:
        return ChatOllama(
            model=config.models.llm_model,
            temperature=config.models.temperature,
            **kwargs
        )

    def create_embeddings(self, **kwargs) -> Embeddings:
        return OllamaEmbeddings(model=config.models.embeddings_model, **kwargs)
