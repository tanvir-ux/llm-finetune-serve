"""Amazon SageMaker real-time endpoint for the merged model (boto3, no SageMaker SDK needed).

STATUS: NOT executed against AWS. The request builders are unit-tested, and every request is
validated against botocore's own API model with the ``Stubber``. You must supply a serving
container image that can run the merged model (for example the AWS LMI/DJL container with the
vLLM rolling batch, or your own vLLM image in ECR). Check that container's documentation for
the environment variables it expects, and review ``environment`` below before deploying.
"""

from __future__ import annotations

from typing import Any


def build_model_request(
    name: str, image_uri: str, model_data_s3: str, role_arn: str, environment: dict[str, str] | None
) -> dict[str, Any]:
    return {
        "ModelName": name,
        "ExecutionRoleArn": role_arn,
        "PrimaryContainer": {
            "Image": image_uri,
            "ModelDataUrl": model_data_s3,
            "Environment": environment or {},
        },
    }


def build_endpoint_config_request(
    name: str, model_name: str, instance_type: str = "ml.g5.xlarge", instance_count: int = 1
) -> dict[str, Any]:
    return {
        "EndpointConfigName": name,
        "ProductionVariants": [
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": instance_type,
                "InitialInstanceCount": instance_count,
                "ContainerStartupHealthCheckTimeoutInSeconds": 600,
            }
        ],
    }


def build_endpoint_request(name: str, config_name: str) -> dict[str, Any]:
    return {"EndpointName": name, "EndpointConfigName": config_name}


DEFAULT_ENVIRONMENT = {
    # Example values for an LMI/DJL container; verify against the container you choose.
    "OPTION_ROLLING_BATCH": "vllm",
    "OPTION_MAX_MODEL_LEN": "1024",
}


def deploy(
    client: Any,
    name: str,
    image_uri: str,
    model_data_s3: str,
    role_arn: str,
    instance_type: str = "ml.g5.xlarge",
    environment: dict[str, str] | None = None,
) -> str:
    """Create the model, endpoint config and endpoint. Returns the endpoint name."""
    client.create_model(
        **build_model_request(
            name, image_uri, model_data_s3, role_arn, environment or DEFAULT_ENVIRONMENT
        )
    )
    client.create_endpoint_config(**build_endpoint_config_request(name, name, instance_type))
    client.create_endpoint(**build_endpoint_request(name, name))
    return name


def delete(client: Any, name: str) -> None:
    """Remove the endpoint, its config and the model (stops billing)."""
    client.delete_endpoint(EndpointName=name)
    client.delete_endpoint_config(EndpointConfigName=name)
    client.delete_model(ModelName=name)


if __name__ == "__main__":  # pragma: no cover - needs AWS credentials
    import argparse

    import boto3

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--image-uri", required=True)
    ap.add_argument("--model-data", required=True, help="s3://bucket/key/model.tar.gz")
    ap.add_argument("--role-arn", required=True)
    ap.add_argument("--instance-type", default="ml.g5.xlarge")
    a = ap.parse_args()
    print(
        deploy(
            boto3.client("sagemaker"),
            a.name,
            a.image_uri,
            a.model_data,
            a.role_arn,
            a.instance_type,
        )  # fmt: skip
    )
