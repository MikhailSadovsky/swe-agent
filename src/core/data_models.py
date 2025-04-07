from pydantic import BaseModel
from pathlib import Path
from typing import Optional


class InstanceItem(BaseModel):
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    repo_path: Optional[Path] = None

    @classmethod
    def from_huggingface(cls, dataset_item: dict) -> "InstanceItem":
        return cls(
            instance_id=dataset_item["instance_id"],
            repo=dataset_item["repo"],
            base_commit=dataset_item["base_commit"],
            problem_statement=dataset_item["problem_statement"],
        )

    @property
    def repo_name(self) -> str:
        return self.repo.replace("/", "__")
