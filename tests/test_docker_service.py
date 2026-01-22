from unittest.mock import MagicMock, patch

import docker.errors
import pytest

from mlflow_dock.docker_service import (
    DockerBuildError,
    DockerPushError,
    _build_docker_image,
    _push_docker_image,
    build_and_push_docker,
)


class TestBuildDockerImage:
    """Tests for Docker image building."""

    @patch("mlflow_dock.docker_service.mlflow")
    def test_successful_build(self, mock_mlflow):
        """Successful build should return result."""
        mock_mlflow.models.build_docker.return_value = "build-result"

        result = _build_docker_image("models:/test/1", "registry/user/test:1")

        assert result == "build-result"
        mock_mlflow.models.build_docker.assert_called_once_with(
            model_uri="models:/test/1",
            name="registry/user/test:1",
        )

    @patch("mlflow_dock.docker_service.mlflow")
    def test_build_failure_raises_error(self, mock_mlflow):
        """Build failure should raise DockerBuildError."""
        mock_mlflow.models.build_docker.side_effect = Exception("Build failed")

        with pytest.raises(DockerBuildError) as exc_info:
            _build_docker_image("models:/test/1", "registry/user/test:1")

        assert "Build failed" in str(exc_info.value)


class TestPushDockerImage:
    """Tests for Docker image pushing with retry logic."""

    @patch("mlflow_dock.docker_service.docker")
    def test_successful_push(self, mock_docker):
        """Successful push should complete without error."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.images.push.return_value = [
            {"status": "Pushing"},
            {"status": "Pushed"},
        ]

        _push_docker_image("registry/user/test:1")

        mock_client.images.push.assert_called_once_with(
            "registry/user/test:1", stream=True, decode=True
        )

    @patch("mlflow_dock.docker_service.docker")
    def test_push_with_auth_config(self, mock_docker):
        """Push with auth_config should pass credentials to docker client."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.images.push.return_value = [{"status": "Pushed"}]

        auth_config = {"username": "testuser", "password": "testpass"}
        _push_docker_image("registry/user/test:1", auth_config=auth_config)

        mock_client.images.push.assert_called_once_with(
            "registry/user/test:1",
            stream=True,
            decode=True,
            auth_config=auth_config,
        )

    @patch("mlflow_dock.docker_service.docker")
    def test_push_without_auth_config(self, mock_docker):
        """Push without auth_config should not include auth in request."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.images.push.return_value = [{"status": "Pushed"}]

        _push_docker_image("registry/user/test:1", auth_config=None)

        mock_client.images.push.assert_called_once_with(
            "registry/user/test:1", stream=True, decode=True
        )

    @patch("mlflow_dock.docker_service.docker")
    def test_push_error_in_response(self, mock_docker):
        """Push error in response should raise DockerPushError."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.images.push.return_value = [
            {"status": "Pushing"},
            {"error": "Access denied"},
        ]

        with pytest.raises(DockerPushError) as exc_info:
            _push_docker_image("registry/user/test:1")

        assert "Access denied" in str(exc_info.value)

    @patch("mlflow_dock.docker_service.docker")
    def test_api_error_triggers_retry(self, mock_docker):
        """Docker API error should trigger retry."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client

        # First call fails, second succeeds
        mock_client.images.push.side_effect = [
            docker.errors.APIError("Connection refused"),
            [{"status": "Pushed"}],
        ]

        # This should succeed after retry
        _push_docker_image("registry/user/test:1")

        assert mock_client.images.push.call_count == 2

    @patch("mlflow_dock.docker_service.docker")
    def test_max_retries_exceeded(self, mock_docker):
        """Should raise after max retries exceeded."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.images.push.side_effect = docker.errors.APIError("Always fails")

        with pytest.raises(docker.errors.APIError):
            _push_docker_image("registry/user/test:1")

        # Should have attempted 3 times (initial + 2 retries)
        assert mock_client.images.push.call_count == 3


class TestBuildAndPushDocker:
    """Tests for combined build and push workflow."""

    @patch("mlflow_dock.docker_service._get_build_log_path")
    @patch("mlflow_dock.docker_service._push_docker_image")
    @patch("mlflow_dock.docker_service._build_docker_image")
    def test_full_workflow_success(
        self, mock_build, mock_push, mock_log_path, tmp_path
    ):
        """Full workflow should build then push."""
        mock_build.return_value = "build-result"
        mock_log_path.return_value = tmp_path / "test.log"

        build_and_push_docker(
            model_uri="models:/test/1",
            model_name="test",
            version="1",
            docker_registry="registry.io",
            docker_username="user",
            docker_registry_password="secret",
        )

        mock_build.assert_called_once_with("models:/test/1", "registry.io/test:1")
        mock_push.assert_called_once_with(
            "registry.io/test:1",
            auth_config={"username": "user", "password": "secret"},
        )

    @patch("mlflow_dock.docker_service._get_build_log_path")
    @patch("mlflow_dock.docker_service._push_docker_image")
    @patch("mlflow_dock.docker_service._build_docker_image")
    def test_build_failure_skips_push(
        self, mock_build, mock_push, mock_log_path, tmp_path
    ):
        """Build failure should prevent push."""
        mock_build.side_effect = DockerBuildError("Build failed")
        mock_log_path.return_value = tmp_path / "test.log"

        with pytest.raises(DockerBuildError):
            build_and_push_docker(
                model_uri="models:/test/1",
                model_name="test",
                version="1",
                docker_registry="registry.io",
                docker_username="user",
                docker_registry_password="secret",
            )

        mock_push.assert_not_called()

    @patch("mlflow_dock.docker_service._get_build_log_path")
    @patch("mlflow_dock.docker_service._push_docker_image")
    @patch("mlflow_dock.docker_service._build_docker_image")
    def test_image_name_format(self, mock_build, mock_push, mock_log_path, tmp_path):
        """Image name should be formatted correctly."""
        mock_log_path.return_value = tmp_path / "test.log"

        build_and_push_docker(
            model_uri="models:/my-model/5",
            model_name="my-model",
            version="5",
            docker_registry="ghcr.io",
            docker_username="myorg",
            docker_registry_password="secret",
        )

        expected_image = "ghcr.io/my-model:5"
        mock_build.assert_called_once_with("models:/my-model/5", expected_image)
        mock_push.assert_called_once_with(
            expected_image,
            auth_config={"username": "myorg", "password": "secret"},
        )

    @patch("mlflow_dock.docker_service._get_build_log_path")
    @patch("mlflow_dock.docker_service._push_docker_image")
    @patch("mlflow_dock.docker_service._build_docker_image")
    def test_workflow_with_registry_password(
        self, mock_build, mock_push, mock_log_path, tmp_path
    ):
        """Workflow with registry password should pass auth config to push."""
        mock_log_path.return_value = tmp_path / "test.log"

        build_and_push_docker(
            model_uri="models:/test/1",
            model_name="test",
            version="1",
            docker_registry="registry.io",
            docker_username="user",
            docker_registry_password="secret123",
        )

        mock_push.assert_called_once_with(
            "registry.io/test:1",
            auth_config={"username": "user", "password": "secret123"},
        )
