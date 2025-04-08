from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import END, StateGraph
from core.constants import TaskType
from core.state import WorkflowState
from core.retriever import HybridRetriever
from agents import CodeAnalyzerAgent, SoftwareEngineerAgent, EditorAgent, ReviewAgent
from config.settings import config


def build_workflow(llm: BaseLanguageModel, retriever: HybridRetriever) -> StateGraph:
    workflow = StateGraph(WorkflowState)

    engineer = SoftwareEngineerAgent(llm=llm, retriever=retriever)
    analyzer = CodeAnalyzerAgent(llm=llm)
    editor = EditorAgent(llm=llm)
    reviewer = ReviewAgent(llm=llm)

    workflow.add_node(TaskType.SOFTWARE_ENGINEER, engineer.execute)
    workflow.add_node(TaskType.CODE_ANALYSIS, analyzer.execute)
    workflow.add_node(TaskType.EDITING, editor.execute)
    workflow.add_node(TaskType.REVIEW, reviewer.execute)
    workflow.add_node(TaskType.COMPLETE, lambda state: state)
    workflow.add_node(TaskType.FAILED, lambda state: state)

    workflow.add_edge(TaskType.SOFTWARE_ENGINEER, TaskType.CODE_ANALYSIS)

    workflow.add_conditional_edges(
        TaskType.CODE_ANALYSIS,
        lambda s: TaskType.EDITING
        if s["current_task"] == TaskType.EDITING
        else TaskType.SOFTWARE_ENGINEER,
        {
            TaskType.EDITING: TaskType.EDITING,
            TaskType.SOFTWARE_ENGINEER: TaskType.SOFTWARE_ENGINEER,
        },
    )

    workflow.add_conditional_edges(
        TaskType.SOFTWARE_ENGINEER,
        lambda s: TaskType.FAILED if _detect_stagnation(s) else TaskType.CODE_ANALYSIS,
        {
            TaskType.CODE_ANALYSIS: TaskType.CODE_ANALYSIS,
            TaskType.FAILED: TaskType.FAILED,
        },
    )

    workflow.add_conditional_edges(
        TaskType.EDITING,
        lambda s: TaskType.REVIEW if s["generated_patch"] else TaskType.FAILED,
        {TaskType.REVIEW: TaskType.REVIEW, TaskType.FAILED: TaskType.FAILED},
    )

    workflow.add_conditional_edges(
        TaskType.REVIEW,
        lambda s: _determine_review_next_step(s),
        {
            TaskType.COMPLETE: TaskType.COMPLETE,
            TaskType.SOFTWARE_ENGINEER: TaskType.SOFTWARE_ENGINEER,
            TaskType.FAILED: TaskType.FAILED,
        },
    )

    workflow.add_edge(TaskType.COMPLETE, END)
    workflow.add_edge(TaskType.FAILED, END)

    workflow.set_entry_point(TaskType.SOFTWARE_ENGINEER)
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
    if state["current_task"] == TaskType.COMPLETE:
        return TaskType.COMPLETE
    if state["review_retry_count"] >= config.workflow.max_review_attempts:
        return TaskType.FAILED
    return TaskType.SOFTWARE_ENGINEER
