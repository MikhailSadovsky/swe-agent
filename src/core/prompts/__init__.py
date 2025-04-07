from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


class PromptManager:
    def __init__(self):
        self.prompts = {}
        self._load_prompts()

    def _load_prompts(self):
        prompt_dir = Path(__file__).parent
        if not prompt_dir.exists():
            raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")

        for prompt_file in prompt_dir.glob("*.yaml"):
            try:
                with open(prompt_file, "r") as f:
                    category = prompt_file.stem
                    self.prompts[category] = yaml.safe_load(f)
                    logger.info(
                        f"Loaded {len(self.prompts[category])} prompts from {category}"
                    )
            except Exception as e:
                logger.error(f"Error loading {prompt_file}: {str(e)}")
                raise

    def get_prompt(self, category: str, name: str, **kwargs) -> str:
        try:
            template = self.prompts[category][name]
            return template
        except KeyError:
            logger.error(f"Prompt not found: {category}/{name}")
            raise
        except Exception as e:
            logger.error(f"Error formatting prompt {category}/{name}: {str(e)}")
            raise


prompt_manager = PromptManager()
