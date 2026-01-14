import base64
import hashlib
import hmac
import json
import time

import pytest
from fastapi.testclient import TestClient

from mlflow_dock.config import Settings, reset_settings


@pytest.fixture
def test_secret():
    """Test webhook secret."""
    return "test-webhook-secret-key"


@pytest.fixture
def test_settings(test_secret, monkeypatch):
    """Configure test environment variables and return settings."""
    monkeypatch.setenv("WEBHOOK_SECRET", test_secret)
    monkeypatch.setenv("DOCKER_REGISTRY", "test-registry.io")
    monkeypatch.setenv("DOCKER_USERNAME", "testuser")
    monkeypatch.setenv("MAX_TIMESTAMP_AGE", "300")
    monkeypatch.setenv("PORT", "8000")

    # Reset settings to pick up new env vars
    reset_settings()

    yield Settings.from_env()

    # Reset after test
    reset_settings()


@pytest.fixture
def client(test_settings):
    """Create FastAPI test client with test settings."""
    from mlflow_dock.main import app

    return TestClient(app)


@pytest.fixture
def valid_webhook_payload():
    """Sample valid webhook payload for model_version.created event."""
    return {
        "entity": "model_version",
        "action": "created",
        "data": {
            "name": "test-model",
            "source": "models:/test-model/1",
            "version": "1",
        },
    }


@pytest.fixture
def tag_webhook_payload():
    """Sample webhook payload for model_version_tag.set event."""
    return {
        "entity": "model_version_tag",
        "action": "set",
        "data": {
            "name": "test-model",
            "version": "1",
            "key": "alias",
            "value": "production",
        },
    }


def generate_signature(
    payload: str, secret: str, delivery_id: str, timestamp: str
) -> str:
    """Generate valid MLflow webhook signature."""
    signed_content = f"{delivery_id}.{timestamp}.{payload}"
    signature = hmac.new(
        secret.encode("utf-8"), signed_content.encode("utf-8"), hashlib.sha256
    ).digest()
    signature_b64 = base64.b64encode(signature).decode("utf-8")
    return f"v1,{signature_b64}"


@pytest.fixture
def webhook_headers(test_secret, valid_webhook_payload):
    """Generate valid webhook headers with signature."""
    delivery_id = "test-delivery-123"
    timestamp = str(int(time.time()))
    payload = json.dumps(valid_webhook_payload)
    signature = generate_signature(payload, test_secret, delivery_id, timestamp)

    return {
        "X-MLflow-Signature": signature,
        "X-MLflow-Delivery-ID": delivery_id,
        "X-MLflow-Timestamp": timestamp,
        "Content-Type": "application/json",
    }


@pytest.fixture
def make_webhook_headers(test_secret):
    """Factory fixture to generate headers for any payload."""

    def _make_headers(payload_dict: dict) -> dict:
        delivery_id = "test-delivery-123"
        timestamp = str(int(time.time()))
        payload = json.dumps(payload_dict)
        signature = generate_signature(payload, test_secret, delivery_id, timestamp)

        return {
            "X-MLflow-Signature": signature,
            "X-MLflow-Delivery-ID": delivery_id,
            "X-MLflow-Timestamp": timestamp,
            "Content-Type": "application/json",
        }

    return _make_headers
