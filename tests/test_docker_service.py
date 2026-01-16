from unittest.mock import MagicMock, patch

import docker.errors
import pytest

from mlflow_dock.docker_service import (
    DockerAuthError,
    DockerBuildError,
    DockerPushError,
    _authenticate_docker,
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


class TestAuthenticateDocker:
    """Tests for Docker authentication."""

    def test_successful_authentication(self):
        """Successful authentication should complete without error."""
        mock_client = MagicMock()
        mock_client.login.return_value = {"Status": "Login Succeeded"}

        _authenticate_docker(
            mock_client, "registry.io", "testuser", "testpassword"
        )

        mock_client.login.assert_called_once_with(
            username="testuser", password="testpassword", registry="registry.io"
        )

    def test_authentication_failure(self):
        """Failed authentication should raise DockerAuthError."""
        mock_client = MagicMock()
        # Create a proper Docker APIError
        api_error = docker.errors.APIError("Invalid credentials")
        mock_client.login.side_effect = api_error

        with pytest.raises(DockerAuthError) as exc_info:
            _authenticate_docker(
                mock_client, "registry.io", "testuser", "wrongpassword"
            )

        assert "Invalid credentials" in str(exc_info.value)
        assert "registry.io" in str(exc_info.value)


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

    @patch("mlflow_dock.docker_service._authenticate_docker")
    @patch("mlflow_dock.docker_service.docker")
    def test_push_with_authentication(self, mock_docker, mock_auth):
        """Push with credentials should authenticate first."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.images.push.return_value = [
            {"status": "Pushing"},
            {"status": "Pushed"},
        ]

        _push_docker_image(
            "registry/user/test:1",
            registry="registry.io",
            username="testuser",
            password="testpassword",
        )

        mock_auth.assert_called_once_with(
            mock_client, "registry.io", "testuser", "testpassword"
        )
        mock_client.images.push.assert_called_once()

    @patch("mlflow_dock.docker_service._authenticate_docker")
    @patch("mlflow_dock.docker_service.docker")
    def test_push_without_authentication(self, mock_docker, mock_auth):
        """Push without credentials should not authenticate."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.images.push.return_value = [
            {"status": "Pushing"},
            {"status": "Pushed"},
        ]

        _push_docker_image("registry/user/test:1")

        mock_auth.assert_not_called()
        mock_client.images.push.assert_called_once()

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

    @patch("mlflow_dock.docker_service._push_docker_image")
    @patch("mlflow_dock.docker_service._build_docker_image")
    def test_full_workflow_success(self, mock_build, mock_push):
        """Full workflow should build then push."""
        mock_build.return_value = "build-result"

        build_and_push_docker(
            model_uri="models:/test/1",
            model_name="test",
            version="1",
            docker_registry="registry.io",
            docker_username="user",
        )

        mock_build.assert_called_once_with("models:/test/1", "registry.io/user/test:1")
        mock_push.assert_called_once_with("registry.io/user/test:1", "registry.io", "user", None)

    @patch("mlflow_dock.docker_service._push_docker_image")
    @patch("mlflow_dock.docker_service._build_docker_image")
    def test_build_failure_skips_push(self, mock_build, mock_push):
        """Build failure should prevent push."""
        mock_build.side_effect = DockerBuildError("Build failed")

        with pytest.raises(DockerBuildError):
            build_and_push_docker(
                model_uri="models:/test/1",
                model_name="test",
                version="1",
                docker_registry="registry.io",
                docker_username="user",
            )

        mock_push.assert_not_called()

    @patch("mlflow_dock.docker_service._push_docker_image")
    @patch("mlflow_dock.docker_service._build_docker_image")
    def test_image_name_format(self, mock_build, mock_push):
        """Image name should be formatted correctly."""
        build_and_push_docker(
            model_uri="models:/my-model/5",
            model_name="my-model",
            version="5",
            docker_registry="ghcr.io",
            docker_username="myorg",
        )

        expected_image = "ghcr.io/myorg/my-model:5"
        mock_build.assert_called_once_with("models:/my-model/5", expected_image)
        mock_push.assert_called_once_with(expected_image, "ghcr.io", "myorg", None)

    @patch("mlflow_dock.docker_service._push_docker_image")
    @patch("mlflow_dock.docker_service._build_docker_image")
    def test_workflow_with_authentication(self, mock_build, mock_push):
        """Workflow with password should pass credentials to push."""
        mock_build.return_value = "build-result"

        build_and_push_docker(
            model_uri="models:/test/1",
            model_name="test",
            version="1",
            docker_registry="registry.io",
            docker_username="user",
            docker_password="secure-password",
        )

        mock_build.assert_called_once_with("models:/test/1", "registry.io/user/test:1")
        mock_push.assert_called_once_with(
            "registry.io/user/test:1", "registry.io", "user", "secure-password"
        )
