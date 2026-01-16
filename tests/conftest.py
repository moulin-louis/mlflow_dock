import base64
import hashlib
import hmac
import json
import os
import time

import pytest
from fastapi.testclient import TestClient

# Set test env vars before importing app
os.environ.setdefault("MLFLOW_WEBHOOK_SECRET", "test-webhook-secret-key")
os.environ.setdefault("DOCKER_REGISTRY", "test-registry.io")
os.environ.setdefault("DOCKER_USERNAME", "testuser")
os.environ.setdefault("DOCKER_REGISTRY_PASSWORD", "testpassword")
os.environ.setdefault("MAX_TIMESTAMP_AGE", "300")
os.environ.setdefault("PORT", "8000")


@pytest.fixture
def test_secret():
    """Test webhook secret."""
    return "test-webhook-secret-key"


@pytest.fixture
def client():
    from mlflow_dock.main import app

    """Create FastAPI test client."""
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
            "run_id": None,
            "tags": {},
            "description": None,
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
