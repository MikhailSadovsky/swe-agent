import tiktoken
import re
from config.settings import config


class CommonUtils:
    _encoders = {}

    @classmethod
    def get_encoder(cls, model_name: str = None):
        model_name = model_name or config.models.llm_model
        if model_name not in cls._encoders:
            try:
                if "deepseek" in model_name.lower():
                    cls._encoders[model_name] = tiktoken.encoding_for_model("gpt-4")
                else:
                    cls._encoders[model_name] = tiktoken.encoding_for_model(model_name)
            except KeyError:
                cls._encoders[model_name] = tiktoken.get_encoding("cl100k_base")
        return cls._encoders[model_name]

    @classmethod
    def truncate_text(
        cls, text: str, max_tokens: int, max_chars: int = None, model_name: str = None
    ) -> str:
        if not text:
            return text
        encoder = cls.get_encoder(model_name)
        tokens = encoder.encode(text)
        truncated_tokens = tokens[:max_tokens]
        truncated_text = encoder.decode(truncated_tokens)

        # Apply character-level truncation if needed
        if max_chars is not None:
            if len(truncated_text) > max_chars:
                truncated_chars = truncated_text[:max_chars]
                valid_tokens = encoder.encode(truncated_chars)
                truncated_text = encoder.decode(valid_tokens)

        if len(tokens) > max_tokens or (max_chars and len(truncated_text) < len(text)):
            truncated_text = truncated_text.rstrip() + "..."

        return truncated_text

    @classmethod
    def calculate_tokens(cls, text: str, model_name: str = None) -> int:
        return len(cls.get_encoder(model_name).encode(text))

    @classmethod
    def classify_problem(cls, problem_stmt: str) -> str:
        problem_stmt = problem_stmt.lower()
        if any(kw in problem_stmt for kw in ["bug", "error", "fix"]):
            return "bug_fix"
        if any(kw in problem_stmt for kw in ["feature", "implement", "new"]):
            return "feature_add"
        return "code_quality"

    @classmethod
    def validate_diff_structure(cls, patch: str) -> bool:
        return all(
            [
                re.search(r"^diff --git", patch, re.MULTILINE),
                re.search(r"^--- ", patch, re.MULTILINE),
                re.search(r"^\+\+\+ ", patch, re.MULTILINE),
                re.search(r"^@@ -\d+,\d+ \+\d+,\d+ @@", patch, re.MULTILINE),
            ]
        )
