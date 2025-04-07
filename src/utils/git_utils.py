from pathlib import Path
from git import Repo
from config.settings import config
import logging

logger = logging.getLogger(__name__)


def setup_repository(repo_url: str, commit_hash: str) -> str:
    """Clone repository and checkout specific commit"""
    repo_name = repo_url.replace("/", "__")
    repo_dir = Path(config.repo_clone_path) / repo_name

    try:
        if not repo_dir.exists():
            logger.info(f"Cloning {repo_url} to {repo_dir}")
            Repo.clone_from(f"https://github.com/{repo_url}.git", str(repo_dir))

        repo = Repo(str(repo_dir))
        if repo.head.commit.hexsha != commit_hash:
            logger.info(f"Checking out commit {commit_hash}")
            repo.git.checkout(commit_hash)

        return str(repo_dir)

    except Exception as e:
        logger.error(f"Repository setup failed: {str(e)}")
        raise
