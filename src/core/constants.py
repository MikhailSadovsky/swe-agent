from enum import Enum


class TaskType(str, Enum):
    SOFTWARE_ENGINEER = "software_engineer"
    CODE_ANALYSIS = "code_analysis"
    EDITING = "editing"
    REVIEW = "review"
    COMPLETE = "complete"
    FAILED = "failed"


class ReviewStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
