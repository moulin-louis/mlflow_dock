import asyncio
import logging

import docker
import docker.errors
import mlflow
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class DockerBuildError(Exception):
    """Raised when Docker image build fails."""

    pass


class DockerPushError(Exception):
    """Raised when Docker image push fails."""

    pass


def _build_docker_image(model_uri: str, image_name: str) -> str:
    """Build Docker image using MLflow.

    Args:
        model_uri: MLflow model URI (e.g., "models:/model_name/1")
        image_name: Full image name including registry and tag

    Returns:
        Build result from MLflow

    Raises:
        DockerBuildError: If build fails
    """
    try:
        logger.info(f"Starting Docker build for {image_name}")
        result = mlflow.models.build_docker(
            model_uri=model_uri,
            name=image_name,
        )
        logger.info(f"Docker build complete: {result}")
        return result
    except Exception as e:
        logger.error(f"Docker build failed: {e}")
        raise DockerBuildError(f"Failed to build image {image_name}: {e}") from e


@retry(
    retry=retry_if_exception_type((docker.errors.APIError, DockerPushError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _push_docker_image(
    image_name: str,
    auth_config: dict[str, str] | None = None,
) -> None:
    """Push Docker image to registry with retry logic.

    Args:
        image_name: Full image name including registry and tag
        auth_config: Optional dict with 'username' and 'password' for registry auth

    Raises:
        DockerPushError: If push fails after retries
        docker.errors.APIError: If Docker API fails after retries
    """
    logger.info(f"Pushing {image_name} to registry")
    client = docker.from_env()

    push_kwargs: dict = {"stream": True, "decode": True}
    if auth_config:
        push_kwargs["auth_config"] = auth_config
        logger.info("Using provided registry credentials for push")

    for line in client.images.push(image_name, **push_kwargs):
        if "status" in line:
            logger.info(f"Push status: {line['status']}")
        if "error" in line:
            error_msg = line["error"]
            logger.error(f"Push error: {error_msg}")
            raise DockerPushError(error_msg)

    logger.info(f"Successfully pushed {image_name} to registry")


def build_and_push_docker(
    model_uri: str,
    model_name: str,
    version: str,
    docker_registry: str,
    docker_username: str,
    docker_registry_password: str,
) -> None:
    """Build and push Docker image for an MLflow model.

    Args:
        model_uri: MLflow model URI
        model_name: Name of the model
        version: Model version
        docker_registry: Docker registry URL
        docker_username: Docker registry username
        docker_registry_password: Registry password for authentication

    Raises:
        DockerBuildError: If build fails
        DockerPushError: If push fails after retries
    """
    image_name = f"{docker_registry}/{model_name}:{version}"
    auth_config = {"username": docker_username, "password": docker_registry_password}

    _build_docker_image(model_uri, image_name)
    _push_docker_image(image_name, auth_config=auth_config)


async def build_and_push_docker_async(
    model_uri: str,
    model_name: str,
    version: str,
    docker_registry: str,
    docker_username: str,
    docker_registry_password: str,
) -> None:
    """Async wrapper that runs the blocking build/push in a thread pool.

    Args:
        model_uri: MLflow model URI
        model_name: Name of the model
        version: Model version
        docker_registry: Docker registry URL
        docker_username: Docker registry username
        docker_registry_password: Registry password for authentication
    """
    image_name = f"{docker_registry}/{docker_username}/{model_name}:{version}"

    try:
        await asyncio.to_thread(
            build_and_push_docker,
            model_uri,
            model_name,
            version,
            docker_registry,
            docker_username,
            docker_registry_password,
        )
    except Exception as e:
        logger.error(f"Build and push failed for {image_name}: {e}")
