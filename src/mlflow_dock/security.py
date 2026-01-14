import base64
import hashlib
import hmac
import time


def verify_timestamp_freshness(timestamp_str: str, max_age: int = 300) -> bool:
    """Verify that the webhook timestamp is recent enough to prevent replay attacks.

    Args:
        timestamp_str: Unix timestamp as string from webhook header
        max_age: Maximum allowed age in seconds (default: 300)

    Returns:
        True if timestamp is valid and within max_age, False otherwise
    """
    try:
        webhook_timestamp = int(timestamp_str)
        current_timestamp = int(time.time())
        age = current_timestamp - webhook_timestamp
        return 0 <= age <= max_age
    except (ValueError, TypeError):
        return False


def verify_mlflow_signature(
    payload: str, signature: str, secret: str, delivery_id: str, timestamp: str
) -> bool:
    """Verify the HMAC-SHA256 signature from MLflow webhook.

    Args:
        payload: Raw request body as string
        signature: Signature from X-MLflow-Signature header (format: "v1,<base64>")
        secret: Webhook secret for HMAC verification
        delivery_id: Unique delivery ID from X-MLflow-Delivery-ID header
        timestamp: Unix timestamp from X-MLflow-Timestamp header

    Returns:
        True if signature is valid, False otherwise
    """
    if not signature.startswith("v1,"):
        return False

    signature_b64 = signature.removeprefix("v1,")
    signed_content = f"{delivery_id}.{timestamp}.{payload}"
    expected_signature = hmac.new(
        secret.encode("utf-8"), signed_content.encode("utf-8"), hashlib.sha256
    ).digest()
    expected_signature_b64 = base64.b64encode(expected_signature).decode("utf-8")
    return hmac.compare_digest(signature_b64, expected_signature_b64)
