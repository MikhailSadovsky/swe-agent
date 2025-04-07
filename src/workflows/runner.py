import logging

from typing import List
from datasets import load_dataset
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from core.data_models import InstanceItem
from agents.swe_agent import SWEBenchAgent
from evaluation.storage import PredictionStore
from config.settings import config

logger = logging.getLogger(__name__)


class WorkflowRunner:
    def __init__(self):
        self.llm = self._initialize_llm()
        self.embeddings = self._initialize_embeddings()
        self.prediction_store = PredictionStore()
        self.processed_instances = set()

    def _initialize_llm(self) -> BaseLanguageModel:
        """Create configured LLM instance once"""
        return ChatOpenAI(
            model=config.models.llm_model,
            temperature=config.models.temperature,
            openai_api_key=config.openai_api_key.get_secret_value(),
        )

    def _initialize_embeddings(self) -> Embeddings:
        """Create configured embeddings once"""
        return OpenAIEmbeddings(
            model=config.models.embeddings_model,
            openai_api_key=config.openai_api_key.get_secret_value(),
        )

    def process_instances(self, instance_ids: List[str]):
        """Process list of SWE-bench instances"""
        logger.info(f"Starting processing for {len(instance_ids)} instances")

        dataset = load_dataset(config.evaluation.dataset_name, split="test")
        valid_instances = [
            InstanceItem.from_huggingface(item)
            for item in dataset
            if item["instance_id"] in instance_ids
            and item["instance_id"] not in self.processed_instances
        ]

        for instance in valid_instances:
            try:
                result = self._process_single_instance(instance)
                if result:
                    self.prediction_store.add_prediction(result)
                    self.processed_instances.add(instance.instance_id)
            except Exception as e:
                logger.error(f"Failed processing {instance.instance_id}: {str(e)}")
                continue

        self.prediction_store.save()
        logger.info(
            f"Completed processing. Total successful: {len(self.prediction_store.get_predictions())}"
        )

    def _process_single_instance(self, instance: InstanceItem) -> dict:
        """Process a single SWE-bench instance"""
        logger.info(f"Processing instance: {instance.instance_id}")

        try:
            agent = SWEBenchAgent(instance, self.llm, self.embeddings)

            result = agent.run_workflow()

            return self._format_result(instance, result)
        except Exception as e:
            logger.error(f"Error processing {instance.instance_id}: {str(e)}")
            raise

    def _format_result(self, instance: InstanceItem, workflow_result: dict) -> dict:
        """Format workflow result into prediction format"""
        return {
            "instance_id": instance.instance_id,
            "model_name_or_path": instance.instance_id,
            "model_patch": workflow_result.get("generated_patch", ""),
            "status": self._determine_status(workflow_result),
        }

    def _determine_status(self, result: dict) -> str:
        """Determine final status from workflow result"""
        if result.get("failure_reason"):
            return "failed"
        if result["current_task"] == "complete":
            return "success"
        return "partial"

    def get_processed_ids(self) -> List[str]:
        """Get list of already processed instance IDs"""
        return list(self.processed_instances)

    def load_existing_predictions(self):
        """Load previous predictions from storage"""
        existing = self.prediction_store.load()
        self.processed_instances = {p["instance_id"] for p in existing}
        logger.info(f"Loaded {len(existing)} existing predictions")
