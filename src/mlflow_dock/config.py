import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    webhook_secret: str
    docker_registry: str
    docker_username: str
    docker_password: str
    max_timestamp_age: int
    port: int

    def __repr__(self) -> str:
        """Custom repr that masks sensitive information."""
        return (
            f"Settings("
            f"webhook_secret='***', "
            f"docker_registry='{self.docker_registry}', "
            f"docker_username='{self.docker_username}', "
            f"docker_password='***', "
            f"max_timestamp_age={self.max_timestamp_age}, "
            f"port={self.port})"
        )

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables.

        Raises:
            KeyError: If required environment variables are missing
        """
        load_dotenv()

        return cls(
            webhook_secret=os.environ["WEBHOOK_SECRET"],
            docker_registry=os.getenv("DOCKER_REGISTRY", "docker.io"),
            docker_username=os.environ["DOCKER_USERNAME"],
            docker_password=os.environ["DOCKER_PASSWORD"],
            max_timestamp_age=int(os.getenv("MAX_TIMESTAMP_AGE", "300")),
            port=int(os.getenv("PORT", "8000")),
        )


# Global settings instance - initialized lazily
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance, initializing if needed."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None
