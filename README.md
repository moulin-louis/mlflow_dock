# MLflow Docker Webhooks

A FastAPI-based webhook receiver for MLflow that automatically builds and pushes Docker images when new model versions are created.

## Features

- **Automatic Docker Image Building**: Listens for MLflow model version creation events and automatically builds Docker images
- **Background Processing**: Uses FastAPI background tasks for non-blocking image builds and pushes

## Prerequisites

- Python 3.12 or higher
- Docker installed and running

## Configuration

Create a `.env` file with the following required variables:

```bash
# Required
WEBHOOK_SECRET=your-webhook-secret-here
DOCKER_USERNAME=your-docker-username

# Optional
DOCKER_REGISTRY=docker.io
MAX_TIMESTAMP_AGE=300
PORT=8000

# AWS credentials (if using S3 for model storage) (used by MLFlow Client)
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
```

## Usage

### Running the Server

```bash
docker run --env-file .env -p 8000:8000 --rm -d ghcr.io/moulin-louis/mlflow_dock:latest 
```

The server will start on `http://0.0.0.0:8000` (or the port specified in your `.env` file).

### Setting Up MLflow Webhook

Register using the MLflow Python client:

```python
from mlflow import MlflowClient

client = MlflowClient("http://your-mlflow-server:5000")
webhook = client.create_webhook(
    name="mlflow_dock_webhook",
    url="https://your-webhook-server:8000/webhook",
    events=[
        "model_version.created",
        "model_version_alias.created",
    ],
    secret="your-webhook-secret",
)
```

### Docker Login

If your using a private registry (Docker.io/ Gitlab/ etc), Make sure you're logged into your Docker registry:

```bash
docker login
```

## How It Works

1. MLflow triggers a webhook when a new model version or alias is created/updated
2. The mlflow-dock verifies the signature and timestamp
3. Upon successful verification, a background task is queued to:
   - Build a Docker image using MLflow's `build_docker` function
   - Push the image to the configured Docker registry
4. The image is tagged as `{DOCKER_REGISTRY}/{DOCKER_USERNAME}/{model_name}:{version}` or `{DOCKER_REGISTRY}/{DOCKER_USERNAME}/{model_name}:{tag}` 

## Security

- **HMAC Signature Verification**: All webhook requests are verified using HMAC-SHA256
- **Timestamp Validation**: Prevents replay attacks by rejecting old requests (default: 5 minutes)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/)
- [MLflow](https://mlflow.org/)
- [Docker Python SDK](https://docker-py.readthedocs.io/)
