"""LoRA / QLoRA fine-tuning loop in plain PyTorch (no Trainer), logging every step.

CPU is the default and is what was actually run. ``--qlora`` loads the base model in 4-bit
(NF4) through bitsandbytes and needs a CUDA GPU; that path was **not executed** for this repo.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from .data import Example, build_splits
from .modeling import DEFAULT_BASE, load_base, load_tokenizer, render_messages, ticket_messages


@dataclass
class TrainConfig:
    base_model: str = DEFAULT_BASE
    out_dir: str = "artifacts/adapter"
    epochs: int = 3
    batch_size: int = 8
    lr: float = 2e-4
    warmup_steps: int = 10
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )  # fmt: skip
    max_len: int = 128
    seed: int = 7
    qlora: bool = False
    max_steps: int | None = None  # for smoke tests
    extra: dict = field(default_factory=dict)


def encode(tok, ex: Example, max_len: int) -> dict[str, list[int]]:
    """Tokenise one example. Loss is computed on the answer tokens only (prompt is masked)."""
    prompt = render_messages(tok, ticket_messages(ex.ticket))
    answer = ex.target + (tok.eos_token or "")
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    return {"input_ids": ids, "labels": labels}


def collate(rows: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        n = len(r["input_ids"])
        ids[i, :n] = torch.tensor(r["input_ids"])
        labels[i, :n] = torch.tensor(r["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def build_peft_model(cfg: TrainConfig):
    if cfg.qlora:
        import bitsandbytes  # noqa: F401  (GPU only; this path is untested here)
        from peft import prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model, quantization_config=bnb)
        base = prepare_model_for_kbit_training(base)
    else:
        base = load_base(cfg.base_model)
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base, lora)


@torch.no_grad()
def evaluate_loss(model, batches: list[dict[str, torch.Tensor]]) -> float:
    model.eval()
    total, count = 0.0, 0
    for b in batches:
        out = model(**b)
        n = int((b["labels"][:, 1:] != -100).sum())
        total += float(out.loss) * n
        count += n
    model.train()
    return total / max(count, 1)


def lr_at(step: int, total: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))


def train(cfg: TrainConfig, splits: dict[str, list[Example]] | None = None, tok=None, model=None):
    """Run training and write the adapter plus ``train_log.json`` to ``cfg.out_dir``."""
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    splits = splits or build_splits(cfg.seed)
    tok = tok or load_tokenizer(cfg.base_model)
    model = model or build_peft_model(cfg)
    model.train()

    train_rows = [encode(tok, e, cfg.max_len) for e in splits["train"]]
    val_rows = [encode(tok, e, cfg.max_len) for e in splits["val"]]
    val_batches = [
        collate(val_rows[i : i + cfg.batch_size], tok.pad_token_id)
        for i in range(0, len(val_rows), cfg.batch_size)
    ]
    steps_per_epoch = math.ceil(len(train_rows) / cfg.batch_size)
    total_steps = cfg.max_steps or steps_per_epoch * cfg.epochs
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    total_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)

    log: dict = {
        "config": asdict(cfg),
        "trainable_params": trainable,
        "total_params": total_params,
        "steps": [],
        "val": [{"step": 0, "loss": evaluate_loss(model, val_batches)}],
    }
    step, start = 0, time.time()
    for epoch in range(cfg.epochs):
        order = list(range(len(train_rows)))
        random.shuffle(order)
        for i in range(0, len(order), cfg.batch_size):
            if step >= total_steps:
                break
            batch = collate(
                [train_rows[j] for j in order[i : i + cfg.batch_size]], tok.pad_token_id
            )
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, cfg)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            log["steps"].append({"step": step, "epoch": epoch + 1, "loss": loss.item(),
                                 "lr": opt.param_groups[0]["lr"]})  # fmt: skip
            if step % 10 == 0:
                print(f"step {step}/{total_steps} loss {loss.item():.4f}", flush=True)
        log["val"].append({"step": step, "loss": evaluate_loss(model, val_batches)})
        print(f"epoch {epoch + 1} val loss {log['val'][-1]['loss']:.4f}", flush=True)
    log["seconds"] = round(time.time() - start, 1)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    return log
