from .base_provider import BaseProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider


class ProviderFactory:
    """Factory to get the appropriate provider."""

    _providers = [OpenAIProvider, OllamaProvider]

    @classmethod
    def get_provider(cls, model_name: str) -> BaseProvider:
        """Get the first provider that supports the model name."""
        for provider_cls in cls._providers:
            if provider_cls.supports(model_name):
                return provider_cls()
        raise ValueError(f"No provider found for model: {model_name}")
