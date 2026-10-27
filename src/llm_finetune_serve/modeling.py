"""Model loading, prompt formatting and batched greedy generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .data import SYSTEM, user_prompt

DEFAULT_BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"


def load_tokenizer(name_or_path: str) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_base(name_or_path: str, dtype: torch.dtype = torch.float32) -> PreTrainedModel:
    return AutoModelForCausalLM.from_pretrained(name_or_path, dtype=dtype)


def load_with_adapter(base: str, adapter: str | Path | None):
    """Return (tokenizer, model); the model carries the LoRA adapter when one is given."""
    tok = load_tokenizer(base)
    model = load_base(base)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return tok, model


def render_messages(tok: PreTrainedTokenizerBase, messages: list[dict[str, str]]) -> str:
    """Chat-template the messages and open the assistant turn."""
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback for tokenizers without a template (used by the offline unit tests).
    lines = [f"{m['role']}: {m['content']}" for m in messages]
    return "\n".join(lines) + "\nassistant:"


def ticket_messages(ticket: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_prompt(ticket)},
    ]


@torch.no_grad()
def generate_texts(
    tok: PreTrainedTokenizerBase,
    model: Any,
    prompts: list[str],
    max_new_tokens: int = 40,
    batch_size: int = 8,
    temperature: float = 0.0,
) -> list[str]:
    """Greedy (or sampled) generation for already-rendered prompts."""
    outs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i : i + batch_size], return_tensors="pt", padding=True)
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tok.pad_token_id,
        }
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature)
        else:
            kwargs.update(do_sample=False)
        gen = model.generate(**enc, **kwargs)
        for row in gen[:, enc["input_ids"].shape[1] :]:
            outs.append(tok.decode(row, skip_special_tokens=True).strip())
    return outs


def parse_triage(text: str) -> dict[str, str] | None:
    """Parse a model reply as the triage JSON object, or None when it is not valid."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj) != {"category", "priority"}:
        return None
    return {k: str(v) for k, v in obj.items()}
