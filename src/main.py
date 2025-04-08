import argparse
from config.settings import config
from evaluation.evaluator import EvaluationManager
from pathlib import Path
from workflows.runner import WorkflowRunner


def update_config_from_args(args):
    """Update configuration from command line arguments"""
    config.evaluation.dataset_name = args.dataset
    config.evaluation.predictions_path = Path(args.predictions_path)
    config.models.llm_model = args.llm_model
    config.models.embeddings_model = args.embeddings_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWE-bench Agentic Solver")
    parser.add_argument(
        "--instances",
        nargs="+",
        required=True,
        help="Specify one or more test instances. Separate multiple instances with spaces.",
    )
    parser.add_argument(
        "--dataset",
        default=config.evaluation.dataset_name,
        help="Name of the dataset to use for evaluation.",
    )
    parser.add_argument(
        "--predictions-path",
        default=config.evaluation.predictions_path,
        help="Path where predictions are stored.",
    )
    parser.add_argument(
        "--llm-model",
        default=config.models.llm_model,
        help="Specify the LLM model to use (currently supported gpt-*, llama*).",
    )
    parser.add_argument(
        "--embeddings-model",
        default=config.models.embeddings_model,
        help="Specify the model used for embeddings generation.",
    )

    args = parser.parse_args()
    update_config_from_args(args)

    runner = WorkflowRunner()
    runner.process_instances(args.instances)

    evaluator = EvaluationManager()
    evaluator.run_evaluation(args.instances)
