from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import END, StateGraph
from core.state import WorkflowState
from core.retriever import HybridRetriever
from agents import CodeAnalyzerAgent, SoftwareEngineerAgent, EditorAgent, ReviewAgent
from config.settings import config
import re


def build_workflow(llm: BaseLanguageModel, retriever: HybridRetriever) -> StateGraph:
    workflow = StateGraph(WorkflowState)

    engineer = SoftwareEngineerAgent(llm=llm, retriever=retriever)
    analyzer = CodeAnalyzerAgent(llm=llm)
    editor = EditorAgent(llm=llm)
    reviewer = ReviewAgent(llm=llm)

    workflow.add_node("software_engineer", engineer.execute)
    workflow.add_node("code_analysis", analyzer.execute)
    workflow.add_node("editing", editor.execute)
    workflow.add_node("review", reviewer.execute)
    workflow.add_node("complete", lambda state: state)
    workflow.add_node("failed", lambda state: state)

    workflow.add_edge("software_engineer", "code_analysis")

    workflow.add_conditional_edges(
        "code_analysis",
        lambda s: "editing" if s["current_task"] == "editing" else "software_engineer",
        {"editing": "editing", "software_engineer": "software_engineer"},
    )

    workflow.add_conditional_edges(
        "software_engineer",
        lambda s: "failed" if _detect_stagnation(s) else "code_analysis",
        {"code_analysis": "code_analysis", "failed": "failed"},
    )

    workflow.add_conditional_edges(
        "editing",
        lambda s: "review" if s["generated_patch"] else "failed",
        {"review": "review", "failed": "failed"},
    )

    workflow.add_conditional_edges(
        "review",
        lambda s: _determine_review_next_step(s),
        {
            "complete": "complete",
            "software_engineer": "software_engineer",
            "failed": "failed",
        },
    )

    workflow.add_edge("complete", END)
    workflow.add_edge("failed", END)

    workflow.set_entry_point("software_engineer")
    return workflow


def _detect_stagnation(state: WorkflowState) -> bool:
    """Check for lack of progress"""
    max_analysis = config.workflow.max_analysis_attempts
    max_review = config.workflow.max_review_attempts

    if state["analysis_attempts"] > max_analysis * 2:
        return True
    if len(state["edit_history"]) > max_review * 3:
        return True
    return False


def _determine_review_next_step(state: WorkflowState) -> str:
    """Determine next step after review"""
    if state["current_task"] == "complete":
        return "complete"
    if state["review_retry_count"] >= config.workflow.max_review_attempts:
        return "failed"
    return "software_engineer"
