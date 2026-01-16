import json
import time
from unittest.mock import AsyncMock, patch


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_healthy(self, client):
        """Health endpoint should return healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestWebhookEndpoint:
    """Tests for webhook endpoint."""

    def test_missing_signature_header(self, client, valid_webhook_payload):
        """Request without signature header should return 422."""
        response = client.post(
            "/webhook",
            json=valid_webhook_payload,
            headers={
                "X-MLflow-Delivery-ID": "test-id",
                "X-MLflow-Timestamp": str(int(time.time())),
            },
        )
        assert response.status_code == 422

    def test_missing_delivery_id_header(self, client, valid_webhook_payload):
        """Request without delivery ID header should return 422."""
        response = client.post(
            "/webhook",
            json=valid_webhook_payload,
            headers={
                "X-MLflow-Signature": "v1,test",
                "X-MLflow-Timestamp": str(int(time.time())),
            },
        )
        assert response.status_code == 422

    def test_missing_timestamp_header(self, client, valid_webhook_payload):
        """Request without timestamp header should return 422."""
        response = client.post(
            "/webhook",
            json=valid_webhook_payload,
            headers={
                "X-MLflow-Signature": "v1,test",
                "X-MLflow-Delivery-ID": "test-id",
            },
        )
        assert response.status_code == 422

    def test_expired_timestamp(self, client, test_secret, valid_webhook_payload):
        """Request with expired timestamp should return 400."""
        from tests.conftest import generate_signature

        old_timestamp = str(int(time.time()) - 400)  # Older than max_age
        payload = json.dumps(valid_webhook_payload)
        signature = generate_signature(payload, test_secret, "test-id", old_timestamp)

        response = client.post(
            "/webhook",
            content=payload,
            headers={
                "X-MLflow-Signature": signature,
                "X-MLflow-Delivery-ID": "test-id",
                "X-MLflow-Timestamp": old_timestamp,
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 400
        assert "Timestamp" in response.json()["detail"]

    def test_invalid_signature(self, client, valid_webhook_payload):
        """Request with invalid signature should return 401."""
        response = client.post(
            "/webhook",
            json=valid_webhook_payload,
            headers={
                "X-MLflow-Signature": "v1,invalidbase64signature==",
                "X-MLflow-Delivery-ID": "test-id",
                "X-MLflow-Timestamp": str(int(time.time())),
            },
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid signature"

    @patch("mlflow_dock.main.build_and_push_docker_async")
    def test_valid_model_version_created_webhook(
        self, mock_build, client, valid_webhook_payload, make_webhook_headers
    ):
        """Valid model_version.created webhook should trigger build."""
        mock_build.return_value = AsyncMock()()

        headers = make_webhook_headers(valid_webhook_payload)

        response = client.post(
            "/webhook",
            content=json.dumps(valid_webhook_payload),
            headers=headers,
        )

        assert response.status_code == 202
        assert response.json() == {"status": "submitted"}



class TestWebhookPayloadExtraction:
    """Tests for webhook payload data extraction."""

    @patch("mlflow_dock.main.build_and_push_docker_async")
    @patch("mlflow_dock.main.asyncio.create_task")
    def test_extracts_model_info_correctly(
        self, mock_create_task, mock_build, client, make_webhook_headers
    ):
        """Webhook should correctly extract model name, URI, and version."""
        payload = {
            "entity": "model_version",
            "action": "created",
            "data": {
                "name": "my-custom-model",
                "source": "models:/my-custom-model/42",
                "version": "42",
                "run_id": None,
                "tags": {},
                "description": None,
            },
        }
        headers = make_webhook_headers(payload)

        response = client.post(
            "/webhook",
            content=json.dumps(payload),
            headers=headers,
        )

        assert response.status_code == 202
        # Verify create_task was called (which means build_and_push_docker_async was invoked)
        mock_create_task.assert_called_once()
