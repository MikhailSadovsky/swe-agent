from abc import ABC, abstractmethod
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings


class BaseProvider(ABC):
    """Base class for LLM providers."""

    @classmethod
    @abstractmethod
    def supports(cls, model_name: str) -> bool:
        """Check if this provider supports the given model name."""
        pass

    @abstractmethod
    def create_llm(self, **kwargs) -> BaseLanguageModel:
        """Create the LLM instance."""
        pass

    @abstractmethod
    def create_embeddings(self, **kwargs) -> Embeddings:
        """Create the embeddings instance."""
        pass
