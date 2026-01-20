import asyncio
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

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

BUILD_LOG_DIR = Path("/var/log/mlflow-dock")


def _get_build_log_path(model_name: str, version: str) -> Path:
    """Generate a log file path for a specific build.

    Args:
        model_name: Name of the model being built
        version: Version or alias of the model

    Returns:
        Path to the log file
    """
    BUILD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    safe_version = version.replace("/", "_").replace(":", "_")
    return BUILD_LOG_DIR / f"{safe_model_name}_{safe_version}_{timestamp}.log"


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
        return mlflow.models.build_docker(
            model_uri=model_uri,
            name=image_name,
        )
    except Exception as e:
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
    log_path = _get_build_log_path(model_name, version)

    # Save original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)

    try:
        with open(log_path, "w") as log_file:
            # Redirect stdout and stderr to log file at fd level
            os.dup2(log_file.fileno(), stdout_fd)
            os.dup2(log_file.fileno(), stderr_fd)

            try:
                _build_docker_image(model_uri, image_name)
                _push_docker_image(image_name, auth_config=auth_config)
            finally:
                # Restore original file descriptors
                os.dup2(saved_stdout_fd, stdout_fd)
                os.dup2(saved_stderr_fd, stderr_fd)
    except Exception as e:
        # Append error to log file
        with open(log_path, "a") as log_file:
            log_file.write(f"\n\nFAILED: {e}\n")
        raise
    finally:
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


async def build_and_push_docker_async(
    model_uri: str,
    model_name: str,
    version: str,
    docker_registry: str,
    docker_username: str,
    docker_registry_password: str,
) -> None:
    """Async wrapper that runs the blocking build/push in a subprocess.

    Args:
        model_uri: MLflow model URI
        model_name: Name of the model
        version: Model version
        docker_registry: Docker registry URL
        docker_username: Docker registry username
        docker_registry_password: Registry password for authentication
    """
    loop = asyncio.get_running_loop()

    with ProcessPoolExecutor(max_workers=1) as executor:
        await loop.run_in_executor(
            executor,
            build_and_push_docker,
            model_uri,
            model_name,
            version,
            docker_registry,
            docker_username,
            docker_registry_password,
        )
