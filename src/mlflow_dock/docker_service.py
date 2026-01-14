import asyncio
import logging

import docker
import docker.errors
import mlflow
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
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
def _push_docker_image(image_name: str) -> None:
    """Push Docker image to registry with retry logic.

    Args:
        image_name: Full image name including registry and tag

    Raises:
        DockerPushError: If push fails after retries
        docker.errors.APIError: If Docker API fails after retries
    """
    logger.info(f"Pushing {image_name} to registry")
    client = docker.from_env()

    for line in client.images.push(image_name, stream=True, decode=True):
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
) -> None:
    """Build and push Docker image for an MLflow model.

    Args:
        model_uri: MLflow model URI
        model_name: Name of the model
        version: Model version
        docker_registry: Docker registry URL
        docker_username: Docker registry username

    Raises:
        DockerBuildError: If build fails
        DockerPushError: If push fails after retries
    """
    image_name = f"{docker_registry}/{docker_username}/{model_name}:{version}"

    _build_docker_image(model_uri, image_name)
    _push_docker_image(image_name)


async def build_and_push_docker_async(
    model_uri: str,
    model_name: str,
    version: str,
    docker_registry: str,
    docker_username: str,
) -> None:
    """Async wrapper that runs the blocking build/push in a thread pool.

    Args:
        model_uri: MLflow model URI
        model_name: Name of the model
        version: Model version
        docker_registry: Docker registry URL
        docker_username: Docker registry username
    """
    await asyncio.to_thread(
        build_and_push_docker,
        model_uri,
        model_name,
        version,
        docker_registry,
        docker_username,
    )
