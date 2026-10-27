from fastapi.testclient import TestClient

from llm_finetune_serve.client import triage
from llm_finetune_serve.serve import create_app


def make_app(tok, tiny_peft):
    return create_app("tiny", "adapter-dir", loader=lambda: (tok, tiny_peft))


def test_health_does_not_load_the_model(tok, tiny_peft):
    c = TestClient(make_app(tok, tiny_peft))
    body = c.get("/health").json()
    assert body["status"] == "ok" and body["loaded"] is False


def test_models_lists_adapter_and_base(tok, tiny_peft):
    c = TestClient(make_app(tok, tiny_peft))
    ids = [m["id"] for m in c.get("/v1/models").json()["data"]]
    assert ids == ["triage-lora", "base"]


def test_chat_completion_shape_for_both_models(tok, tiny_peft):
    c = TestClient(make_app(tok, tiny_peft))
    for model in ("triage-lora", "base"):
        r = c.post("/v1/chat/completions", json={
            "model": model, "max_tokens": 6,
            "messages": [{"role": "user", "content": "hello there"}],
        })  # fmt: skip
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "chat.completion" and body["model"] == model
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["usage"]["total_tokens"] == (
            body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
        )
    assert c.get("/health").json()["loaded"] is True


def test_errors(tok, tiny_peft):
    c = TestClient(make_app(tok, tiny_peft))
    msg = [{"role": "user", "content": "x"}]
    assert (
        c.post("/v1/chat/completions", json={"model": "nope", "messages": msg}).status_code == 404
    )
    assert c.post("/v1/chat/completions", json={"messages": msg, "stream": True}).status_code == 400
    assert c.post("/v1/chat/completions", json={"messages": []}).status_code == 422


def test_openai_sdk_client_round_trip(tok, tiny_peft):
    http = TestClient(make_app(tok, tiny_peft), base_url="http://testserver/v1")
    out = triage(
        "My order 1234 has not arrived.", base_url="http://testserver/v1", http_client=http
    )
    assert out["model"] == "triage-lora"
    assert isinstance(out["reply"], str)
    assert out["parsed"] is None or set(out["parsed"]) == {"category", "priority"}
