"""A small OpenAI-compatible server (``/v1/models``, ``/v1/chat/completions``) on transformers.

This is the CPU fallback that was actually run. For GPU throughput use vLLM with the config in
``deploy/vllm`` (not executed here). Both expose the same API, so ``client.py`` works with either.
Streaming is not implemented: ``"stream": true`` returns HTTP 400.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .modeling import generate_texts, load_with_adapter, render_messages

ADAPTER_MODEL = "triage-lora"
BASE_MODEL = "base"


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = ADAPTER_MODEL
    messages: list[Message] = Field(min_length=1)
    max_tokens: int = Field(default=64, ge=1, le=256)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False


def create_app(
    base: str,
    adapter: str | None,
    loader: Callable[[], tuple[Any, Any]] | None = None,
) -> FastAPI:
    """Build the app. The model is loaded lazily on the first completion request."""
    app = FastAPI(title="llm-finetune-serve", version="0.1.0")
    state: dict[str, Any] = {}
    lock = threading.Lock()

    def get_model() -> tuple[Any, Any]:
        with lock:
            if "model" not in state:
                tok, model = loader() if loader else load_with_adapter(base, adapter)
                state["tok"], state["model"] = tok, model
        return state["tok"], state["model"]

    def served_models() -> list[str]:
        return [ADAPTER_MODEL, BASE_MODEL] if adapter or loader else [BASE_MODEL]

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "loaded": "model" in state, "base": base, "adapter": adapter}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {"id": m, "object": "model", "created": now, "owned_by": "llm-finetune-serve"}
                for m in served_models()
            ],
        }

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest) -> dict[str, Any]:
        if req.stream:
            raise HTTPException(400, "streaming is not supported by this server")
        if req.model not in served_models():
            raise HTTPException(404, f"unknown model {req.model!r}; use one of {served_models()}")
        tok, model = get_model()
        prompt = render_messages(tok, [m.model_dump() for m in req.messages])
        with lock:  # one generation at a time on CPU
            if req.model == BASE_MODEL and hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    text = generate_texts(tok, model, [prompt], req.max_tokens, 1, req.temperature)[
                        0
                    ]
            else:
                text = generate_texts(tok, model, [prompt], req.max_tokens, 1, req.temperature)[0]
        n_prompt = len(tok(prompt)["input_ids"])
        n_completion = len(tok(text, add_special_tokens=False)["input_ids"])
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": n_prompt,
                "completion_tokens": n_completion,
                "total_tokens": n_prompt + n_completion,
            },
        }

    return app
