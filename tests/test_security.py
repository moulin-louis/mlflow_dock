import base64
import hashlib
import hmac
import time

from mlflow_dock.security import verify_mlflow_signature, verify_timestamp_freshness


class TestVerifyTimestampFreshness:
    """Tests for timestamp freshness verification."""

    def test_valid_recent_timestamp(self):
        """Recent timestamp should be valid."""
        timestamp = str(int(time.time()))
        assert verify_timestamp_freshness(timestamp) is True

    def test_valid_timestamp_at_max_age(self):
        """Timestamp at exactly max_age should be valid."""
        timestamp = str(int(time.time()) - 300)
        assert verify_timestamp_freshness(timestamp, max_age=300) is True

    def test_expired_timestamp(self):
        """Timestamp older than max_age should be invalid."""
        timestamp = str(int(time.time()) - 301)
        assert verify_timestamp_freshness(timestamp, max_age=300) is False

    def test_very_old_timestamp(self):
        """Very old timestamp should be invalid."""
        timestamp = str(int(time.time()) - 3600)
        assert verify_timestamp_freshness(timestamp) is False

    def test_future_timestamp(self):
        """Future timestamp should be invalid."""
        timestamp = str(int(time.time()) + 10)
        assert verify_timestamp_freshness(timestamp) is False

    def test_invalid_timestamp_string(self):
        """Non-numeric timestamp should be invalid."""
        assert verify_timestamp_freshness("not-a-number") is False

    def test_empty_timestamp(self):
        """Empty string should be invalid."""
        assert verify_timestamp_freshness("") is False

    def test_none_timestamp(self):
        """None should be invalid."""
        assert verify_timestamp_freshness(None) is False

    def test_custom_max_age(self):
        """Custom max_age should be respected."""
        timestamp = str(int(time.time()) - 60)
        assert verify_timestamp_freshness(timestamp, max_age=120) is True
        assert verify_timestamp_freshness(timestamp, max_age=30) is False


class TestVerifyMlflowSignature:
    """Tests for HMAC signature verification."""

    def _generate_signature(
        self, payload: str, secret: str, delivery_id: str, timestamp: str
    ) -> str:
        """Helper to generate valid signature."""
        signed_content = f"{delivery_id}.{timestamp}.{payload}"
        signature = hmac.new(
            secret.encode("utf-8"), signed_content.encode("utf-8"), hashlib.sha256
        ).digest()
        return f"v1,{base64.b64encode(signature).decode('utf-8')}"

    def test_valid_signature(self):
        """Valid signature should pass verification."""
        payload = '{"entity": "model_version", "action": "created"}'
        secret = "test-secret"
        delivery_id = "delivery-123"
        timestamp = "1234567890"

        signature = self._generate_signature(payload, secret, delivery_id, timestamp)

        assert (
            verify_mlflow_signature(payload, signature, secret, delivery_id, timestamp)
            is True
        )

    def test_invalid_signature_wrong_secret(self):
        """Signature with wrong secret should fail."""
        payload = '{"entity": "model_version"}'
        correct_secret = "correct-secret"
        wrong_secret = "wrong-secret"
        delivery_id = "delivery-123"
        timestamp = "1234567890"

        signature = self._generate_signature(
            payload, correct_secret, delivery_id, timestamp
        )

        assert (
            verify_mlflow_signature(
                payload, signature, wrong_secret, delivery_id, timestamp
            )
            is False
        )

    def test_invalid_signature_modified_payload(self):
        """Signature should fail if payload was modified."""
        original_payload = '{"entity": "model_version"}'
        modified_payload = '{"entity": "model_version", "extra": "data"}'
        secret = "test-secret"
        delivery_id = "delivery-123"
        timestamp = "1234567890"

        signature = self._generate_signature(
            original_payload, secret, delivery_id, timestamp
        )

        assert (
            verify_mlflow_signature(
                modified_payload, signature, secret, delivery_id, timestamp
            )
            is False
        )

    def test_invalid_signature_wrong_delivery_id(self):
        """Signature should fail if delivery_id doesn't match."""
        payload = '{"entity": "model_version"}'
        secret = "test-secret"
        original_delivery_id = "delivery-123"
        wrong_delivery_id = "delivery-456"
        timestamp = "1234567890"

        signature = self._generate_signature(
            payload, secret, original_delivery_id, timestamp
        )

        assert (
            verify_mlflow_signature(
                payload, signature, secret, wrong_delivery_id, timestamp
            )
            is False
        )

    def test_invalid_signature_wrong_timestamp(self):
        """Signature should fail if timestamp doesn't match."""
        payload = '{"entity": "model_version"}'
        secret = "test-secret"
        delivery_id = "delivery-123"
        original_timestamp = "1234567890"
        wrong_timestamp = "1234567891"

        signature = self._generate_signature(
            payload, secret, delivery_id, original_timestamp
        )

        assert (
            verify_mlflow_signature(
                payload, signature, secret, delivery_id, wrong_timestamp
            )
            is False
        )

    def test_invalid_signature_format_no_prefix(self):
        """Signature without v1, prefix should fail."""
        payload = '{"entity": "model_version"}'
        secret = "test-secret"
        delivery_id = "delivery-123"
        timestamp = "1234567890"

        # Generate valid signature but remove prefix
        valid_sig = self._generate_signature(payload, secret, delivery_id, timestamp)
        invalid_sig = valid_sig.removeprefix("v1,")

        assert (
            verify_mlflow_signature(
                payload, invalid_sig, secret, delivery_id, timestamp
            )
            is False
        )

    def test_invalid_signature_wrong_prefix(self):
        """Signature with wrong prefix should fail."""
        payload = '{"entity": "model_version"}'
        secret = "test-secret"
        delivery_id = "delivery-123"
        timestamp = "1234567890"

        valid_sig = self._generate_signature(payload, secret, delivery_id, timestamp)
        invalid_sig = "v2," + valid_sig.removeprefix("v1,")

        assert (
            verify_mlflow_signature(
                payload, invalid_sig, secret, delivery_id, timestamp
            )
            is False
        )

    def test_invalid_signature_garbage(self):
        """Garbage signature should fail."""
        assert (
            verify_mlflow_signature(
                '{"test": true}', "v1,not-valid-base64!", "secret", "id", "ts"
            )
            is False
        )

    def test_empty_payload(self):
        """Empty payload should work with valid signature."""
        payload = ""
        secret = "test-secret"
        delivery_id = "delivery-123"
        timestamp = "1234567890"

        signature = self._generate_signature(payload, secret, delivery_id, timestamp)

        assert (
            verify_mlflow_signature(payload, signature, secret, delivery_id, timestamp)
            is True
        )

    def test_unicode_payload(self):
        """Unicode payload should work correctly."""
        payload = '{"name": "模型名称", "description": "テスト"}'
        secret = "test-secret"
        delivery_id = "delivery-123"
        timestamp = "1234567890"

        signature = self._generate_signature(payload, secret, delivery_id, timestamp)

        assert (
            verify_mlflow_signature(payload, signature, secret, delivery_id, timestamp)
            is True
        )
