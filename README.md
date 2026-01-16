# MLflow Docker Webhooks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A FastAPI-based webhook receiver for MLflow that automatically builds and pushes Docker images when new model versions are created.

## Features

- **Automatic Docker Image Building**: Listens for MLflow model version creation events and automatically builds Docker images
- **Background Processing**: Uses FastAPI background tasks for non-blocking image builds and pushes
- **Secure**: HMAC signature verification and timestamp validation to prevent replay attacks
- **Multi-Registry Support**: Works with Docker Hub, GitHub Container Registry, AWS ECR, and more

## Supported Events

| Event | Description |
|-------|-------------|
| `model_version.created` | Triggered when a new model version is registered |
| `model_version_alias.created` | Triggered when an alias is assigned to a model version |

Other MLflow webhook events are not supported and will return a `422` validation error.

## Prerequisites

- Python 3.12 or higher
- Docker installed and running
- MLflow server with webhook support

## Configuration

Create a `.env` file with the following variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MLFLOW_WEBHOOK_SECRET` | Yes | - | Secret key for HMAC signature verification |
| `DOCKER_USERNAME` | Yes | - | Username for Docker registry authentication |
| `DOCKER_REGISTRY_PASSWORD` | Yes | - | Password or token for Docker registry authentication |
| `DOCKER_REGISTRY` | No | `docker.io` | Docker registry URL |
| `MAX_TIMESTAMP_AGE` | No | `300` | Maximum age (seconds) for webhook timestamps |
| `PORT` | No | `8000` | Server port |

### Example `.env` file

```bash
MLFLOW_WEBHOOK_SECRET=your-webhook-secret-here
DOCKER_USERNAME=your-docker-username
DOCKER_REGISTRY_PASSWORD=your-registry-password
DOCKER_REGISTRY=docker.io
```

## Usage

### Running the Server

```bash
docker run --env-file .env -p 8000:8000 --rm -d ghcr.io/moulin-louis/mlflow_dock:latest
```

The server will start on `http://0.0.0.0:8000` (or the port specified in your `.env` file).

### Setting Up MLflow Webhook

Register the webhook using the MLflow Python client:

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

## Registry Configuration Examples

### Docker Hub

```bash
DOCKER_REGISTRY=docker.io
DOCKER_USERNAME=myusername
DOCKER_REGISTRY_PASSWORD=dckr_pat_xxxxx
```

Images will be pushed as: `docker.io/model-name:version`

### GitHub Container Registry (GHCR)

```bash
DOCKER_REGISTRY=ghcr.io
DOCKER_USERNAME=my-github-username
DOCKER_REGISTRY_PASSWORD=ghp_xxxxx  # Personal access token with write:packages scope
```

Images will be pushed as: `ghcr.io/model-name:version`

### AWS Elastic Container Registry (ECR)

```bash
DOCKER_REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com
DOCKER_USERNAME=AWS
DOCKER_REGISTRY_PASSWORD=$(aws ecr get-login-password --region us-east-1)
```

Images will be pushed as: `123456789012.dkr.ecr.us-east-1.amazonaws.com/model-name:version`

Note: ECR passwords expire after 12 hours. Consider using a credential helper or refreshing the token periodically.

### Google Artifact Registry

```bash
DOCKER_REGISTRY=us-central1-docker.pkg.dev/my-project/my-repo
DOCKER_USERNAME=_json_key
DOCKER_REGISTRY_PASSWORD=$(cat service-account-key.json | base64)
```

Images will be pushed as: `us-central1-docker.pkg.dev/my-project/my-repo/model-name:version`

## API Reference

### `POST /webhook`

Receives MLflow webhook events. Requires the following headers:

| Header | Description |
|--------|-------------|
| `X-MLflow-Signature` | HMAC-SHA256 signature of the request body |
| `X-MLflow-Delivery-ID` | Unique identifier for the webhook delivery |
| `X-MLflow-Timestamp` | Unix timestamp of when the webhook was sent |

**Response Codes:**

| Code | Description |
|------|-------------|
| `202` | Webhook accepted, build queued |
| `400` | Invalid timestamp (expired or missing) |
| `401` | Invalid signature |
| `422` | Validation error (unsupported event or malformed payload) |

### `GET /health`

Health check endpoint.

**Response:**
```json
{"status": "healthy"}
```

## How It Works

1. MLflow triggers a webhook when a new model version or alias is created
2. mlflow-dock verifies the HMAC signature and timestamp
3. Upon successful verification, a background task is queued to:
   - Build a Docker image using MLflow's `build_docker` function
   - Push the image to the configured Docker registry
4. The image is tagged as `{DOCKER_REGISTRY}/{model_name}:{version}`

## Troubleshooting

### Authentication Errors

**Symptom:** Push fails with "unauthorized" or "access denied"

**Solutions:**
- Verify `DOCKER_USERNAME` and `DOCKER_REGISTRY_PASSWORD` are correct
- For Docker Hub, use an access token instead of password
- For GHCR, ensure the token has `write:packages` scope
- For ECR, ensure the token hasn't expired (12-hour limit)

### Signature Verification Errors

**Symptom:** `401 Invalid signature` response

**Solutions:**
- Ensure `MLFLOW_WEBHOOK_SECRET` matches the secret configured in MLflow
- Check that the webhook URL is correct (no trailing slash differences)

### Timestamp Errors

**Symptom:** `400 Timestamp is too old` response

**Solutions:**
- Ensure server clocks are synchronized (NTP)
- Increase `MAX_TIMESTAMP_AGE` if network latency is high

### Build Failures

**Symptom:** Image build fails

**Solutions:**
- Ensure Docker daemon is running and accessible
- Check that the model exists in MLflow and has artifacts
- Verify AWS credentials if model artifacts are stored in S3

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/moulin-louis/mlflow_dock.git
cd mlflow_dock

# Install dependencies
uv sync --extra dev

# Run tests
uv run pytest
```

### Running Locally

```bash
# Set environment variables
export MLFLOW_WEBHOOK_SECRET=test-secret
export DOCKER_USERNAME=myuser
export DOCKER_REGISTRY_PASSWORD=mypassword

# Run the server
uv run mlflow_dock
```

## Security

- **HMAC Signature Verification**: All webhook requests are verified using HMAC-SHA256
- **Timestamp Validation**: Prevents replay attacks by rejecting requests older than `MAX_TIMESTAMP_AGE` seconds (default: 5 minutes)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/)
- [MLflow](https://mlflow.org/)
- [Docker Python SDK](https://docker-py.readthedocs.io/)
