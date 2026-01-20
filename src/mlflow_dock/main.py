import asyncio
import logging
import os
from typing import Literal

import mlflow
from fastapi import FastAPI, Header, HTTPException, Request
from mlflow.webhooks.types import (
    ModelVersionAliasCreatedPayload,
    ModelVersionCreatedPayload,
)
from pydantic import BaseModel

from mlflow_dock.config import Settings
from mlflow_dock.docker_service import build_and_push_docker_async
from mlflow_dock.security import verify_mlflow_signature, verify_timestamp_freshness

app = FastAPI()
settings = Settings.from_env()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

exp = mlflow.search_experiments()
print(f"found {len(exp)}")


class ModelVersionCreatedEvent(BaseModel):
    entity: Literal["model_version"]
    action: Literal["created"]
    data: ModelVersionCreatedPayload


class ModelVersionAliasCreatedEvent(BaseModel):
    entity: Literal["model_version_alias"]
    action: Literal["created"]
    data: ModelVersionAliasCreatedPayload


WebhookEvent = ModelVersionCreatedEvent | ModelVersionAliasCreatedEvent


@app.post("/webhook", status_code=202)
async def handle_webhook(
    request: Request,
    event: WebhookEvent,
    x_mlflow_signature: str = Header(),
    x_mlflow_delivery_id: str = Header(),
    x_mlflow_timestamp: str = Header(),
):
    """Handle webhook with HMAC signature verification."""

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
        settings.mlflow_webhook_secret,
        x_mlflow_delivery_id,
        x_mlflow_timestamp,
    ):
        raise HTTPException(status_code=401, detail="Invalid signature")

    logger.info(f"Received webhook: {event.entity}.{event.action}")

    match event:
        case ModelVersionCreatedEvent(data=data):
            asyncio.create_task(
                build_and_push_docker_async(
                    model_uri=data["source"],
                    model_name=data["name"],
                    version=data["version"],
                    docker_registry=settings.docker_registry,
                    docker_username=settings.docker_username,
                    docker_registry_password=settings.docker_registry_password,
                )
            )
            logger.info(
                f"Queued Docker build and push for {data['name']}:{data['version']}"
            )
        case ModelVersionAliasCreatedEvent(data=data):
            asyncio.create_task(
                build_and_push_docker_async(
                    model_uri=f"models:/{data['name']}@{data['alias']}",
                    model_name=data["name"],
                    version=data["alias"],
                    docker_registry=settings.docker_registry,
                    docker_username=settings.docker_username,
                    docker_registry_password=settings.docker_registry_password,
                )
            )
            logger.info(
                f"Queued Docker build and push for {data['name']}:{data['alias']}"
            )

    return {"status": "submitted"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


def main():
    """Main entry point for running the FastAPI server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
