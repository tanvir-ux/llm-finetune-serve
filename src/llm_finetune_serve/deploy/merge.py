"""Merge a LoRA adapter into its base model so cloud runtimes can load a plain checkpoint."""

from __future__ import annotations

import tarfile
from pathlib import Path


def merge_adapter(base: str, adapter: str, out_dir: str, tok=None, model=None) -> Path:
    """Write base+adapter weights as one standard Hugging Face model directory."""
    from peft import PeftModel

    from .. import modeling

    tok = tok or modeling.load_tokenizer(base)
    model = model or modeling.load_base(base)
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    return out


def make_tarball(model_dir: str | Path, tar_path: str | Path) -> Path:
    """Create the ``model.tar.gz`` layout SageMaker expects (files at the archive root)."""
    model_dir, tar_path = Path(model_dir), Path(tar_path)
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in sorted(model_dir.iterdir()):
            tar.add(f, arcname=f.name)
    return tar_path
