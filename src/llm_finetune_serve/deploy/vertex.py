"""Vertex AI endpoint running the vLLM OpenAI server container.

STATUS: NOT executed against Google Cloud. The upload and deploy arguments are built by pure
functions that are unit-tested; the thin ``deploy`` wrapper calls ``google-cloud-aiplatform``
(imported lazily, not a dependency of this package). ``gcloud_commands`` prints the equivalent
CLI steps. The merged model must already be in a GCS bucket (see ``llmft merge``).
"""

from __future__ import annotations

from typing import Any

VLLM_PORT = 8080


def build_upload_kwargs(
    display_name: str, image_uri: str, model_gcs_uri: str, served_name: str = "triage-lora"
) -> dict[str, Any]:
    """Arguments for ``aiplatform.Model.upload`` using the vLLM OpenAI-compatible server."""
    return {
        "display_name": display_name,
        "serving_container_image_uri": image_uri,
        "serving_container_command": ["python", "-m", "vllm.entrypoints.openai.api_server"],
        "serving_container_args": [
            "--model",
            model_gcs_uri,
            "--served-model-name",
            served_name,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--max-model-len",
            "1024",
        ],  # fmt: skip
        "serving_container_ports": [VLLM_PORT],
        "serving_container_predict_route": "/v1/chat/completions",
        "serving_container_health_route": "/health",
    }


def build_deploy_kwargs(
    machine_type: str = "g2-standard-4",
    accelerator_type: str = "NVIDIA_L4",
    accelerator_count: int = 1,
    min_replicas: int = 1,
    max_replicas: int = 1,
) -> dict[str, Any]:
    return {
        "machine_type": machine_type,
        "accelerator_type": accelerator_type,
        "accelerator_count": accelerator_count,
        "min_replica_count": min_replicas,
        "max_replica_count": max_replicas,
    }


def deploy(project: str, region: str, upload: dict[str, Any], deployment: dict[str, Any]) -> str:
    """Upload the model and deploy it to a new endpoint. Returns the endpoint resource name."""
    from google.cloud import aiplatform  # lazy import: not installed by default

    aiplatform.init(project=project, location=region)
    model = aiplatform.Model.upload(**upload)
    endpoint = model.deploy(**deployment)
    return endpoint.resource_name


def gcloud_commands(project: str, region: str, upload: dict[str, Any]) -> list[str]:
    """The same upload expressed as a gcloud command (deployment still needs endpoint steps)."""
    args = ",".join(upload["serving_container_args"])
    return [
        f"gcloud ai models upload --project={project} --region={region} "
        f"--display-name={upload['display_name']} "
        f"--container-image-uri={upload['serving_container_image_uri']} "
        f"--container-command={','.join(upload['serving_container_command'])} "
        f"--container-args={args} "
        f"--container-ports={upload['serving_container_ports'][0]} "
        f"--container-predict-route={upload['serving_container_predict_route']} "
        f"--container-health-route={upload['serving_container_health_route']}",
    ]
