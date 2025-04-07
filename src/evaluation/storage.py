import json
from pathlib import Path
from typing import List, Dict, Optional
from config.settings import config
import logging

logger = logging.getLogger(__name__)


class PredictionStore:
    """Handles storage and retrieval of prediction results"""

    def __init__(self, file_path: Optional[Path] = None):
        self.file_path = file_path or config.evaluation.predictions_path
        self._predictions: List[Dict] = []
        self._loaded_ids = set()
        self._initialize_storage()

    def _initialize_storage(self):
        """Load existing predictions and prepare storage directory"""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if self.file_path.exists():
            try:
                with open(self.file_path, "r") as f:
                    existing = json.load(f)
                    self._predictions = existing
                    self._loaded_ids = {p["instance_id"] for p in existing}
                    logger.info(f"Loaded {len(existing)} existing predictions")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed loading predictions: {str(e)}")
                self._predictions = []

    def add_prediction(self, prediction: Dict):
        """Add a new prediction to the store"""
        if not self._is_valid_prediction(prediction):
            logger.error(
                f"Invalid prediction format for prediction {prediction}, skipping"
            )
            return

        if prediction["instance_id"] in self._loaded_ids:
            logger.warning(f"Duplicate prediction for {prediction['instance_id']}")
            return

        self._predictions.append(prediction)
        self._loaded_ids.add(prediction["instance_id"])
        logger.debug(f"Added prediction for {prediction['instance_id']}")

    def save(self):
        """Persist predictions to disk"""
        try:
            with open(self.file_path, "w") as f:
                json.dump(self._predictions, f, indent=2)
                logger.info(
                    f"Saved {len(self._predictions)} predictions to {self.file_path}"
                )
        except IOError as e:
            logger.error(f"Failed saving predictions: {str(e)}")
            raise

    def get_predictions(self) -> List[Dict]:
        """Return all stored predictions"""
        return self._predictions.copy()

    def get_processed_ids(self) -> set:
        """Return set of already processed instance IDs"""
        return self._loaded_ids.copy()

    def _is_valid_prediction(self, prediction: Dict) -> bool:
        """Validate prediction structure"""
        required_keys = {
            "instance_id",
            "model_name_or_path",
            "model_patch",
        }
        return all(key in prediction for key in required_keys)

    def clear(self):
        """Clear all stored predictions (for testing)"""
        self._predictions.clear()
        self._loaded_ids.clear()
        logger.warning("Cleared all predictions from store")
