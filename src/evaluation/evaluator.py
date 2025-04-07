from config.settings import config
import logging
from swebench.harness.run_evaluation import main as run_evaluation

logger = logging.getLogger(__name__)


class EvaluationManager:
    def run_evaluation(
        self,
        instances,
        split="test",
        max_workers=4,
        force_rebuild=False,
        cache_level="env",
        clean=False,
        open_file_limit=4096,
        run_id="1",
        timeout=1800,
        rewrite_reports=False,
        modal=False,
        namespace=None,
    ):
        logger.info("Started SWE Bench evaluation")
        run_evaluation(
            dataset_name=config.evaluation.dataset_name,
            split=split,
            instance_ids=instances,
            predictions_path=str(config.evaluation.predictions_path),
            max_workers=max_workers,
            force_rebuild=force_rebuild,
            cache_level=cache_level,
            clean=clean,
            open_file_limit=open_file_limit,
            run_id=run_id,
            timeout=timeout,
            rewrite_reports=rewrite_reports,
            modal=modal,
            namespace=namespace,
        )
