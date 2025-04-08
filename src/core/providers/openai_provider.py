from .base_provider import BaseProvider
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config.settings import config


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI models."""

    @classmethod
    def supports(cls, model_name: str) -> bool:
        return model_name.startswith("gpt-")

    def create_llm(self, **kwargs) -> BaseLanguageModel:
        return ChatOpenAI(
            model=config.models.llm_model,
            temperature=config.models.temperature,
            openai_api_key=config.openai_api_key.get_secret_value(),
            **kwargs
        )

    def create_embeddings(self, **kwargs) -> Embeddings:
        return OpenAIEmbeddings(
            model=config.models.embeddings_model,
            openai_api_key=config.openai_api_key.get_secret_value(),
            **kwargs
        )
