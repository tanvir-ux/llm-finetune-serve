"""Turn the training and eval artifacts into the static demo page data.

Reads ``artifacts/adapter/train_log.json`` and ``artifacts/eval.json`` (produced by
``llmft train`` and ``llmft eval``) and writes:

* ``docs/results.json``: everything the GitHub Pages demo renders,
* ``docs/loss_curve.png``: the same loss curve as a picture for the README,
* ``models/triage-lora/``: the small adapter, so the repo is usable without retraining.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADAPTER = ROOT / "artifacts" / "adapter"
EVAL = ROOT / "artifacts" / "eval.json"
EXAMPLES_PER_SET = 14  # clickable tickets per test set on the page; all rows stay in the file


def build() -> dict:
    train = json.loads((ADAPTER / "train_log.json").read_text())
    evaluation = json.loads(EVAL.read_text())
    cfg = train["config"]
    train["steps_per_epoch"] = math.ceil(320 / cfg["batch_size"])
    data = {"train": train, "eval": evaluation, "examples_per_set": EXAMPLES_PER_SET}
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "results.json").write_text(json.dumps(data, separators=(",", ":")))
    return data


def copy_adapter() -> None:
    dest = ROOT / "models" / "triage-lora"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_model.safetensors", "adapter_config.json", "train_log.json"):
        shutil.copy2(ADAPTER / name, dest / name)


def plot(train: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [s["step"] for s in train["steps"]]
    loss = [s["loss"] for s in train["steps"]]
    k = 5
    smooth = [sum(loss[max(0, i - k) : i + 1]) / len(loss[max(0, i - k) : i + 1])
              for i in range(len(loss))]  # fmt: skip
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=130)
    ax.plot(steps, loss, color="#9fb6ea", lw=1, label="training loss (per step)")
    ax.plot(steps, smooth, color="#2456d6", lw=2.5, label="training loss (smoothed)")
    ax.scatter([v["step"] for v in train["val"]], [v["loss"] for v in train["val"]],
               color="#d9480f", zorder=3, label="validation loss")  # fmt: skip
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy on the answer tokens")
    ax.set_title("LoRA fine-tuning of SmolLM2-135M-Instruct on CPU")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "loss_curve.png")


if __name__ == "__main__":
    d = build()
    copy_adapter()
    plot(d["train"])
    print("wrote docs/results.json, docs/loss_curve.png, models/triage-lora/")
