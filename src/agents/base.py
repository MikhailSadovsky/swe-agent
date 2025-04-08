from abc import ABC, abstractmethod
from config.settings import config
from core.constants import TaskType
from utils.common_utils import CommonUtils
from langchain_core.language_models import BaseLanguageModel
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.common_utils = CommonUtils

    @abstractmethod
    def execute(self, state: dict) -> dict:
        pass

    def _handle_error(self, state: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        logger.error(error_msg)
        return {
            **state,
            "current_task": TaskType.SOFTWARE_ENGINEER,
            "failure_reason": error_msg,
            "token_count": state.get("token_count", 0) + 500,
        }

    def _update_token_count(self, state: Dict[str, Any], new_content: str) -> int:
        return state.get("token_count", 0) + self.common_utils.calculate_tokens(
            new_content
        )

    def _format_code_context(self, docs: list) -> str:
        """Format retrieved docs for analysis context"""
        context = (
            "\n".join(
                [
                    f"### {doc.metadata.get('source', 'Unknown')}\n"
                    f"{self.common_utils.truncate_text(doc.page_content, 500, 2000)}"
                    for doc in docs
                ]
            )
            if docs
            else "No relevant code context found"
        )

        return self.common_utils.truncate_text(
            context,
            config.workflow.max_context_tokens // 2,
            config.workflow.max_content_length // 4,
        )
