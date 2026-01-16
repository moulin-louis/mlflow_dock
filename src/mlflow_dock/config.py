import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    mlflow_webhook_secret: str
    docker_registry: str
    docker_username: str
    docker_registry_password: str
    max_timestamp_age: int
    port: int

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables.

        Raises:
            KeyError: If required environment variables are missing
        """
        load_dotenv()

        return cls(
            mlflow_webhook_secret=os.environ["MLFLOW_WEBHOOK_SECRET"],
            docker_registry=os.getenv("DOCKER_REGISTRY", "docker.io"),
            docker_username=os.environ["DOCKER_USERNAME"],
            docker_registry_password=os.environ["DOCKER_REGISTRY_PASSWORD"],
            max_timestamp_age=int(os.getenv("MAX_TIMESTAMP_AGE", "300")),
            port=int(os.getenv("PORT", "8000")),
        )
