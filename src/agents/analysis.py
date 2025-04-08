from langchain_core.language_models import BaseLanguageModel
from .base import BaseAgent
from core.prompts import prompt_manager
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import config
from typing import Dict, Any
import logging
import re

logger = logging.getLogger(__name__)


class CodeAnalyzerAgent(BaseAgent):
    def __init__(self, llm: BaseLanguageModel):
        super().__init__(llm)
        self.encoder = self.common_utils.get_encoder()

        self.analysis_prompt_template = prompt_manager.get_prompt(
            "analysis", "deep_analysis"
        )
        self.validation_prompt_template = prompt_manager.get_prompt(
            "analysis", "analysis_validation"
        )

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code analysis with quality control gates"""
        try:
            # Build and validate analysis
            analysis = self._generate_analysis(state)

            if not self._validate_analysis(analysis):
                raise ValueError("Analysis validation failed")

            return self._update_state(state, analysis)

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return self._handle_error(state, "Analysis generation failed")

    def _generate_analysis(self, state: Dict[str, Any]) -> str:
        """Generate technical analysis using structured prompt"""
        prompt = self.analysis_prompt_template.format(
            problem_stmt=self.common_utils.truncate_text(
                state["problem_stmt"], 200, config.workflow.max_content_length // 8
            ),
            code_context=self._format_code_context(state["retrieved_docs"]),
            previous_analysis=self._summarize_previous_analysis(state),
            review_feedback=self.common_utils.truncate_text(
                state.get("review_feedback", ""),
                200,
            ),
            attempts_left=config.workflow.max_analysis_attempts
            - state["analysis_attempts"],
        )

        messages = [SystemMessage(content=prompt)]
        return self.llm.invoke(messages).content

    def _validate_analysis(self, analysis: str) -> bool:
        """Ensure analysis meets quality standards"""
        messages = [
            SystemMessage(content=self.validation_prompt_template),
            HumanMessage(content=analysis),
        ]
        response = self.llm.invoke(messages).content
        return "VALID" in response.upper()

    def _update_state(self, state: Dict[str, Any], analysis: str) -> Dict[str, Any]:
        """Update workflow state with new analysis"""
        return {
            **state,
            "analysis": analysis,
            "analysis_history": state["analysis_history"] + [analysis],
            "analysis_attempts": state["analysis_attempts"] + 1,
            "current_task": self._determine_next_step(state),
            "token_count": state["token_count"] + self._calculate_token_usage(analysis),
        }

    def _summarize_previous_analysis(self, state: Dict[str, Any]) -> str:
        """Create condensed summary of analysis history"""
        if not state["analysis_history"]:
            return "No previous analysis available"

        return "\n".join(
            [
                f"Attempt {i+1}: {self._extract_key_points(a)}"
                for i, a in enumerate(state["analysis_history"][-2:])
            ]
        )

    def _extract_key_points(self, analysis: str) -> str:
        """Extract main conclusions from analysis text"""
        points = re.findall(
            r"(Critical Issue|Proposed Solution|Implementation Step):\s*(.+?)(?=\n\w+:|$)",
            analysis,
        )
        return (
            ", ".join([f"{k}: {v}" for k, v in points])
            if points
            else "No key points extracted"
        )

    def _determine_next_step(self, state: Dict[str, Any]) -> str:
        """Determine next workflow step"""
        if state["analysis_attempts"] >= config.workflow.max_analysis_attempts:
            return "editing"
        return "software_engineer"

    def _calculate_token_usage(self, analysis: str) -> int:
        """Calculate exact token usage"""
        return len(self.encoder.encode(analysis))
