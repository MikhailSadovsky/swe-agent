from pydantic import BaseModel, Field, model_validator, SecretStr
from pathlib import Path
from enum import Enum

from pydantic_settings import BaseSettings, SettingsConfigDict


class FileExtensions(Enum):
    RELEVANT = (".py",)


class ModelSettings(BaseModel):
    embeddings_model: str = Field(default="text-embedding-3-small")
    llm_model: str = Field(default="gpt-4-turbo")
    temperature: float = Field(default=0.0)
    deepseek_base_url: str = Field(default="https://api.deepseek.com/v1")


class RetrievalSettings(BaseModel):
    chunk_size: int = Field(default=1200)
    chunk_overlap: int = Field(default=300)
    parser_threshold: int = Field(default=500)
    relevant_extensions: tuple = Field(default=FileExtensions.RELEVANT.value)
    vector_store_path: str = Field(default="faiss_index")


class WorkflowSettings(BaseModel):
    max_context_tokens: int = Field(default=120000)
    max_content_length: int = Field(default=396000)
    max_analysis_attempts: int = Field(default=3)
    max_review_attempts: int = Field(default=3)
    max_files_per_patch: int = Field(default=20)
    recursion_additional_limit: int = Field(default=50)
    thread_id: int = Field(default=1)


class EvaluationSettings(BaseModel):
    dataset_name: str = Field(default="princeton-nlp/SWE-bench_Verified")
    predictions_path: Path = Field(default=Path("results/predictions.json"))
    max_workers: int = Field(default=4)
    timeout: int = Field(default=1800)


class Settings(BaseSettings):
    models: ModelSettings = ModelSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    workflow: WorkflowSettings = WorkflowSettings()
    evaluation: EvaluationSettings = EvaluationSettings()
    openai_api_key: SecretStr
    deepseek_api_key: SecretStr
    repo_clone_path: str = Field(default="repos")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @model_validator(mode="after")
    def create_paths(self) -> "Settings":
        self.evaluation.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        return self


config = Settings()
