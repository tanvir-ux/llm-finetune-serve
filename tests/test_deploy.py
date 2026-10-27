import boto3
from botocore.stub import Stubber

from llm_finetune_serve.cli import main
from llm_finetune_serve.deploy import sagemaker, vertex
from llm_finetune_serve.deploy.merge import make_tarball, merge_adapter
from llm_finetune_serve.train import TrainConfig, train


def sm_client():
    return boto3.client(
        "sagemaker", region_name="us-east-1",
        aws_access_key_id="dummy", aws_secret_access_key="dummy",
    )  # fmt: skip


def test_sagemaker_requests_are_valid_against_the_botocore_model():
    client = sm_client()
    role = "arn:aws:iam::123456789012:role/SageMakerExecution"
    req_model = sagemaker.build_model_request("m", "img:1", "s3://b/model.tar.gz", role, {"A": "b"})
    req_config = sagemaker.build_endpoint_config_request("m", "m")
    req_endpoint = sagemaker.build_endpoint_request("m", "m")
    arn = "arn:aws:sagemaker:us-east-1:1:"
    with Stubber(client) as stub:
        stub.add_response("create_model", {"ModelArn": arn + "model/m"}, req_model)
        stub.add_response(
            "create_endpoint_config", {"EndpointConfigArn": arn + "endpoint-config/m"}, req_config
        )
        stub.add_response("create_endpoint", {"EndpointArn": arn + "endpoint/m"}, req_endpoint)
        name = sagemaker.deploy(
            client, "m", "img:1", "s3://b/model.tar.gz", role, environment={"A": "b"}
        )
        stub.assert_no_pending_responses()
    assert name == "m"


def test_sagemaker_delete_calls_all_three_apis():
    client = sm_client()
    with Stubber(client) as stub:
        stub.add_response("delete_endpoint", {}, {"EndpointName": "m"})
        stub.add_response("delete_endpoint_config", {}, {"EndpointConfigName": "m"})
        stub.add_response("delete_model", {}, {"ModelName": "m"})
        sagemaker.delete(client, "m")
        stub.assert_no_pending_responses()


def test_vertex_kwargs():
    up = vertex.build_upload_kwargs("triage", "us-docker.pkg.dev/p/r/vllm:1", "gs://b/merged")
    assert up["serving_container_args"][:2] == ["--model", "gs://b/merged"]
    assert up["serving_container_predict_route"] == "/v1/chat/completions"
    assert up["serving_container_ports"] == [vertex.VLLM_PORT]
    dep = vertex.build_deploy_kwargs()
    assert dep["accelerator_type"] == "NVIDIA_L4" and dep["machine_type"].startswith("g2-")
    cmd = vertex.gcloud_commands("proj", "us-central1", up)[0]
    assert cmd.startswith("gcloud ai models upload") and "--project=proj" in cmd


def test_merge_adapter_and_tarball(tok, splits, tiny_peft, tmp_path):
    from tests.conftest import make_tiny

    small = {k: v[:8] for k, v in splits.items()}
    train(TrainConfig(out_dir=str(tmp_path / "ad"), epochs=1, max_len=160), small, tok, tiny_peft)
    out = merge_adapter(
        "tiny", str(tmp_path / "ad"), str(tmp_path / "merged"), tok, make_tiny(len(tok))
    )
    assert (out / "config.json").exists() and (out / "model.safetensors").exists()
    tar = make_tarball(out, tmp_path / "model.tar.gz")
    import tarfile

    names = tarfile.open(tar).getnames()
    assert "config.json" in names and all("/" not in n for n in names)


def test_cli_data_command(capsys):
    assert main(["data"]) == 0
    assert "train: 320 examples" in capsys.readouterr().out
