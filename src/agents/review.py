from langchain_core.language_models import BaseLanguageModel
from .base import BaseAgent
from core.prompts import prompt_manager
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import config
from typing import Dict, Any, Literal
import re
import logging
import tiktoken

logger = logging.getLogger(__name__)


class ReviewAgent(BaseAgent):
    def __init__(self, llm: BaseLanguageModel):
        super().__init__(llm)

        self.encoder = tiktoken.encoding_for_model(config.models.llm_model)

        self.system_prompt_template = prompt_manager.get_prompt(
            "review", "validation_system"
        )
        self.human_prompt_template = prompt_manager.get_prompt(
            "review", "validation_human"
        )
        self.validation_pattern = re.compile(
            r"STATUS:\s*(APPROVED|REJECTED)", re.IGNORECASE
        )

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive code review with quality gates"""
        try:
            messages = [
                SystemMessage(content=self._build_system_prompt(state)),
                HumanMessage(content=self._build_human_prompt(state)),
            ]

            feedback = self.llm.invoke(messages).content
            status = self._determine_status(feedback)

            return self._update_state(state, feedback, status)
        except Exception as e:
            logger.error(f"Review failed: {str(e)}")
            return self._handle_error(state, "Review process failed")

    def _build_system_prompt(self, state: Dict[str, Any]) -> str:
        """Construct system prompt with validation criteria"""
        return self.system_prompt_template.format(
            max_review_attempts=config.workflow.max_review_attempts,
            problem_type=self.common_utils.classify_problem(state["problem_stmt"]),
        )

    def _build_human_prompt(self, state: Dict[str, Any]) -> str:
        """Build human prompt with review context"""
        return self.human_prompt_template.format(
            problem_stmt=self.common_utils.truncate_text(state["problem_stmt"], 200),
            patch_content=state["generated_patch"],
            previous_feedback=self._format_previous_feedback(state),
            analysis_summary=self._summarize_analysis(state["analysis"]),
            attempt_count=state["review_retry_count"],
            max_review_attempts=config.workflow.max_review_attempts,
            problem_type=self.common_utils.classify_problem(state["problem_stmt"]),
        )

    def _determine_status(self, feedback: str) -> Literal["approved", "rejected"]:
        """Parse review decision from LLM response"""
        if match := self.validation_pattern.search(feedback):
            return match.group(1).lower()
        return "rejected"

    def _update_state(
        self, state: Dict[str, Any], feedback: str, status: str
    ) -> Dict[str, Any]:
        """Update state with review results"""
        return {
            **state,
            "review_feedback": feedback,
            "review_retry_count": state["review_retry_count"]
            + (0 if status == "approved" else 1),
            "current_task": self._determine_next_step(status, state),
            "token_count": state["token_count"] + self._calculate_token_usage(feedback),
        }

    def _calculate_token_usage(self, feedback: str) -> int:
        """Calculate precise token usage"""
        return len(self.encoder.encode(feedback))

    def _format_previous_feedback(self, state: Dict[str, Any]) -> str:
        """Format feedback history"""
        if not state.get("review_feedback"):
            return "No previous feedback"

        return "\n".join(
            [
                f"Attempt {i+1}: {self.common_utils.truncate_text(fb, 200)}"
                for i, fb in enumerate(state["review_feedback"][-2:])
            ]
        )

    def _summarize_analysis(self, analysis: str) -> str:
        """Extract key analysis points"""
        return "\n".join(
            re.findall(
                r"#+\s(Key Issues|Solution Strategy):?\s*(.+?)(?=\n#|$)", analysis
            )
        )

    def _determine_next_step(self, status: str, state: Dict[str, Any]) -> str:
        """Determine workflow progression"""
        if status == "approved":
            return "complete"
        if state["review_retry_count"] >= config.workflow.max_review_attempts:
            return "failed"
        return "software_engineer"
