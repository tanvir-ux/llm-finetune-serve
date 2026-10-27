"""OpenAI-compatible client. Works against ``llmft serve`` and against a vLLM server."""

from __future__ import annotations

import os

from openai import OpenAI

from .modeling import parse_triage, ticket_messages


def make_client(
    base_url: str | None = None, api_key: str | None = None, http_client=None
) -> OpenAI:
    return OpenAI(
        base_url=base_url or os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),
        timeout=120,
        http_client=http_client,
    )


def triage(
    ticket: str,
    base_url: str | None = None,
    model: str = "triage-lora",
    api_key: str | None = None,
    http_client=None,
) -> dict:
    """Send one ticket and return ``{"reply": raw text, "parsed": dict | None}``."""
    client = make_client(base_url, api_key, http_client)
    resp = client.chat.completions.create(
        model=model,
        messages=ticket_messages(ticket),
        max_tokens=40,
        temperature=0,
    )
    reply = (resp.choices[0].message.content or "").strip()
    return {"model": resp.model, "reply": reply, "parsed": parse_triage(reply)}
