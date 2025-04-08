from core.constants import TaskType
from .base import BaseAgent
from core.prompts import prompt_manager
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseLanguageModel
from config.settings import config
from typing import Dict, Any
import re
import logging

logger = logging.getLogger(__name__)


class EditorAgent(BaseAgent):
    def __init__(self, llm: BaseLanguageModel):
        super().__init__(llm)

        self.system_prompt_template = prompt_manager.get_prompt("editing", "system")
        self.human_prompt_template = prompt_manager.get_prompt("editing", "human")
        self.validation_prompt_template = prompt_manager.get_prompt(
            "editing", "validation"
        )

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate and validate code patches with quality control"""
        try:
            context = self._assemble_context(state)

            messages = [
                SystemMessage(content=self._build_system_prompt(state)),
                HumanMessage(content=self._build_human_prompt(state, context)),
            ]

            raw_patch = self.llm.invoke(messages).content
            validated_patch = self._validate_patch(raw_patch, state)

            return self._update_state(state, validated_patch)

        except Exception as e:
            return self._handle_error(state, f"Patch generation failed: {str(e)}")

    def _assemble_context(self, state: Dict[str, Any]) -> str:
        """Build code context with smart token allocation"""
        token_budget = config.workflow.max_context_tokens - state["token_count"]
        context_parts = []

        problem_stmt = self.common_utils.truncate_text(
            state["problem_stmt"],
            token_budget // 4,
            config.workflow.max_content_length // 10,
        )
        context_parts.append(f"## Problem Statement\n{problem_stmt}")
        remaining_tokens = token_budget - self.common_utils.calculate_tokens(
            problem_stmt
        )

        code_context = []
        for doc in state["retrieved_docs"]:
            content = f"### {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            tokens_needed = self.common_utils.calculate_tokens(content)

            if tokens_needed <= remaining_tokens:
                code_context.append(content)
                remaining_tokens -= tokens_needed
            else:
                truncated = self.common_utils.truncate_text(content, remaining_tokens)
                code_context.append(truncated)
                break

        if code_context:
            context_parts.append("## Relevant Code Context\n" + "\n".join(code_context))

        return "\n\n".join(context_parts)

    def _build_system_prompt(self, state: Dict[str, Any]) -> str:
        """Construct system-level instructions"""
        return self.system_prompt_template.format(
            problem_type=self.common_utils.classify_problem(state["problem_stmt"]),
            max_files=config.workflow.max_files_per_patch,
        )

    def _build_human_prompt(self, state: Dict[str, Any], context: str) -> str:
        """Build task-specific prompt content"""
        return self.human_prompt_template.format(
            problem_stmt=self.common_utils.truncate_text(
                state["problem_stmt"], 200, config.workflow.max_content_length // 10
            ),
            analysis_summary=self._summarize_analysis(state["analysis"]),
            code_context=self.common_utils.truncate_text(
                context, 3000, config.workflow.max_content_length // 2
            ),
            previous_attempts=self._format_attempts(state["edit_history"]),
            review_feedback=self.common_utils.truncate_text(
                state.get("review_feedback", ""), 200, 500
            ),
        )

    def _validate_patch(self, patch: str, state: Dict[str, Any]) -> str:
        """Multi-stage patch validation"""
        if not self._validate_patch_structure(patch):
            return "INVALID: Malformed diff structure"

        if not self._validate_patch_content(patch, state):
            return "INVALID: Does not address problem"

        return patch

    def _validate_patch_structure(self, patch: str) -> bool:
        """Validate basic diff syntax"""
        return self.common_utils.validate_diff_structure(patch)

    def _validate_patch_content(self, patch: str, state: Dict[str, Any]) -> bool:
        """Semantic validation using LLM"""
        messages = [
            SystemMessage(content=self.validation_prompt_template),
            HumanMessage(content=f"Problem: {state['problem_stmt']}\nPatch:\n{patch}"),
        ]
        response = self.llm.invoke(messages).content
        return "VALID" in response.upper()

    def _update_state(self, state: Dict[str, Any], patch: str) -> Dict[str, Any]:
        """Update workflow state with token tracking"""
        return {
            **state,
            "generated_patch": patch,
            "edit_history": state["edit_history"] + [patch],
            "current_task": TaskType.REVIEW
            if "INVALID" not in patch
            else TaskType.SOFTWARE_ENGINEER,
            "token_count": state["token_count"]
            + self.common_utils.calculate_tokens(patch),
        }

    def _summarize_analysis(self, analysis: str) -> str:
        """Extract key analysis points"""
        return "\n".join(
            re.findall(
                r"#+\s(Implementation Plan|Critical Issues):?\s*(.+?)(?=\n#|$)",
                analysis,
                re.DOTALL,
            )
        )

    def _format_attempts(self, attempts: list) -> str:
        """Format edit history for context"""
        return (
            "\n".join(
                [
                    f"Attempt {i+1}: {self.common_utils.truncate_text(a, 200)}"
                    for i, a in enumerate(attempts[-2:])
                ]
            )
            or "No previous attempts"
        )
