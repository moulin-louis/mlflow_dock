import asyncio
import logging

from fastapi import FastAPI, Header, HTTPException, Request

from mlflow_dock.config import get_settings
from mlflow_dock.docker_service import build_and_push_docker_async
from mlflow_dock.security import verify_mlflow_signature, verify_timestamp_freshness

app = FastAPI()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@app.post("/webhook")
async def handle_webhook(
    request: Request,
    x_mlflow_signature: str = Header(),
    x_mlflow_delivery_id: str = Header(),
    x_mlflow_timestamp: str = Header(),
):
    """Handle webhook with HMAC signature verification."""
    settings = get_settings()

    payload_bytes = await request.body()
    payload = payload_bytes.decode("utf-8")

    if not x_mlflow_signature:
        raise HTTPException(status_code=400, detail="Missing signature header")
    if not x_mlflow_delivery_id:
        raise HTTPException(status_code=400, detail="Missing delivery ID header")
    if not x_mlflow_timestamp:
        raise HTTPException(status_code=400, detail="Missing timestamp header")

    if not verify_timestamp_freshness(x_mlflow_timestamp, settings.max_timestamp_age):
        raise HTTPException(
            status_code=400,
            detail="Timestamp is too old or invalid (possible replay attack)",
        )

    if not verify_mlflow_signature(
        payload,
        x_mlflow_signature,
        settings.webhook_secret,
        x_mlflow_delivery_id,
        x_mlflow_timestamp,
    ):
        raise HTTPException(status_code=401, detail="Invalid signature")

    webhook_data = await request.json()

    entity = webhook_data.get("entity")
    action = webhook_data.get("action")
    payload_data = webhook_data.get("data", {})

    logger.info(f"Received webhook: {entity}.{action}")
    logger.info(f"Payload: {payload_data}")

    if entity == "model_version" and action == "created":
        model_name = payload_data.get("name")
        model_uri = payload_data.get("source")
        version = payload_data.get("version")

        logger.info(f"model_uri = {model_uri}")

        asyncio.create_task(
            build_and_push_docker_async(
                model_uri=model_uri,
                model_name=model_name,
                version=version,
                docker_registry=settings.docker_registry,
                docker_username=settings.docker_username,
            )
        )
        logger.info(f"Queued Docker build and push for {model_name}:{version}")
    elif entity == "model_version_tag" and action == "set":
        model_name = payload_data.get("name")
        version = payload_data.get("version")
        tag_key = payload_data.get("key")
        tag_value = payload_data.get("value")
        logger.info(f"Tag set on {model_name} v{version}: {tag_key}={tag_value}")

    return {"status": "success"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


def main():
    """Main entry point for running the FastAPI server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
