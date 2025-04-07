import operator
from typing import TypedDict, Literal
from langchain_core.documents import Document
from typing import Annotated, List


def overwrite_reducer(old_value, new_value):
    return new_value


def append_reducer(old_value, new_value):
    if isinstance(old_value, list):
        old_value.append(new_value)
        return old_value
    return [old_value, new_value]


def increment_reducer(old_value, new_value):
    return old_value + new_value if isinstance(old_value, int) else new_value


class WorkflowState(TypedDict):
    instance_id: Annotated[str, overwrite_reducer]
    problem_stmt: Annotated[str, overwrite_reducer]
    repo_path: Annotated[str, overwrite_reducer]
    current_task: Annotated[
        Literal["software_engineer", "code_analysis", "editing", "review", "complete"],
        overwrite_reducer,
    ]
    retrieved_docs: Annotated[List[Document], operator.add]
    analysis: Annotated[str, overwrite_reducer]
    analysis_history: Annotated[List[str], operator.add]
    generated_patch: Annotated[str, overwrite_reducer]
    analysis_attempts: Annotated[int, increment_reducer]
    review_retry_count: Annotated[int, increment_reducer]
    review_feedback: Annotated[str, overwrite_reducer]
    token_count: Annotated[int, increment_reducer]
    edit_history: Annotated[List[dict], operator.add]
    failure_reason: Annotated[str, overwrite_reducer]
