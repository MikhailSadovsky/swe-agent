from config.settings import config
from core.data_models import InstanceItem
from core.retriever import HybridRetriever
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from core.state import WorkflowState
from workflows.graph import build_workflow
from utils.git_utils import setup_repository
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class SWEBenchAgent:
    def __init__(
        self, instance: InstanceItem, llm: BaseLanguageModel, embeddings: Embeddings
    ):
        self.instance = instance
        self.llm = llm
        self.embeddings = embeddings
        self.repo_path = setup_repository(
            repo_url=instance.repo, commit_hash=instance.base_commit
        )
        self.retriever = HybridRetriever(self.repo_path, self.llm, self.embeddings)
        self.workflow = build_workflow(self.llm, self.retriever)

    def run_workflow(self) -> Dict[str, Any]:
        initial_state: WorkflowState = {
            "instance_id": self.instance.instance_id,
            "problem_stmt": self.instance.problem_statement,
            "repo_path": self.repo_path,
            "current_task": "software_engineer",
            "retrieved_docs": [],
            "analysis": "",
            "analysis_history": [],
            "generated_patch": "",
            "analysis_attempts": 0,
            "review_retry_count": 0,
            "review_feedback": "",
            "token_count": 0,
            "edit_history": [],
            "failure_reason": "",
        }
        app = self.workflow.compile(checkpointer=MemorySaver())
        return app.invoke(
            initial_state,
            {
                "configurable": {"thread_id": config.workflow.thread_id},
                "recursion_limit": config.workflow.max_analysis_attempts
                + config.workflow.max_review_attempts
                + config.workflow.recursion_additional_limit,
            },
        )
