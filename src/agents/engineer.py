from .base import BaseAgent
from config.settings import config
from core.prompts import prompt_manager
from core.retriever import HybridRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class SoftwareEngineerAgent(BaseAgent):
    def __init__(self, llm: BaseLanguageModel, retriever: HybridRetriever):
        super().__init__(llm)
        self.retriever = retriever
        self.decision_prompt_template = prompt_manager.get_prompt(
            "engineer", "decision_prompt"
        )

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            retrieved_docs = self.retriever.retrieve(
                state["problem_stmt"], state.get("review_feedback", "")
            )

            messages = [
                SystemMessage(
                    content=self.common_utils.truncate_text(
                        self._build_prompt(state),
                        config.workflow.max_context_tokens,
                        config.workflow.max_content_length,
                    )
                )
            ]
            response = self.llm.invoke(messages).content
            next_task = self._parse_response(response)

            return {
                **state,
                "retrieved_docs": retrieved_docs,
                "current_task": next_task,
                "token_count": self._update_token_count(state, response),
            }
        except Exception as e:
            return self._handle_error(state, f"Decision failed: {str(e)}")

    def _build_prompt(self, state: Dict[str, Any]) -> str:
        return self.decision_prompt_template.format(
            problem_stmt=self.common_utils.truncate_text(state["problem_stmt"], 200),
            analysis_attempts=state["analysis_attempts"],
            last_analysis=state.get("review_feedback", ""),
            max_attempts=config.workflow.max_analysis_attempts,
            analysis_summary=self._summarize_analysis(state),
            review_feedback=self.common_utils.truncate_text(
                state.get("review_feedback", ""), 200
            ),
            docs_summary=self._summarize_docs(state["retrieved_docs"]),
        )

    def _summarize_analysis(self, state: Dict[str, Any]) -> str:
        return (
            "\n".join(
                [
                    f"Analysis {i+1}: {self.common_utils.truncate_text(a, 200)}"
                    for i, a in enumerate(state["analysis_history"][-3:])
                ]
            )
            if state["analysis_history"]
            else "No previous analysis"
        )

    def _summarize_docs(self, docs: list) -> str:
        return (
            "\n".join(
                [
                    f"• {d.metadata.get('source', 'unknown')}: {self.common_utils.truncate_text(d.page_content, 500)}"
                    for d in docs
                ]
            )
            if docs
            else "No relevant code found"
        )

    def _parse_response(self, response: str) -> str:
        if response.upper() == "ANALYZE":
            return "code_analysis"
        return "editing"
