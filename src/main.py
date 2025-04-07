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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWE-bench Agentic Solver")
    parser.add_argument("--instances", nargs="+", required=True)
    parser.add_argument("--dataset", default=config.evaluation.dataset_name)
    parser.add_argument(
        "--predictions-path", default=config.evaluation.predictions_path
    )
    parser.add_argument("--llm-model", default=config.models.llm_model)

    args = parser.parse_args()
    update_config_from_args(args)

    runner = WorkflowRunner()
    runner.process_instances(args.instances)

    evaluator = EvaluationManager()
    evaluator.run_evaluation(args.instances)
