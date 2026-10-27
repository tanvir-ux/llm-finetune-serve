"""Command line: ``llmft data | train | eval | merge | serve | chat``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import build_splits
from .modeling import DEFAULT_BASE


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="llmft", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("data", help="print dataset statistics and a few samples")

    t = sub.add_parser("train", help="LoRA fine-tune on CPU (or QLoRA on a CUDA GPU)")
    t.add_argument("--base", default=DEFAULT_BASE)
    t.add_argument("--out", default="artifacts/adapter")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--qlora", action="store_true", help="4-bit base (needs CUDA; untested here)")
    t.add_argument("--max-steps", type=int, default=None)

    e = sub.add_parser("eval", help="base vs tuned model on the held-out test sets")
    e.add_argument("--base", default=DEFAULT_BASE)
    e.add_argument("--adapter", default="artifacts/adapter")
    e.add_argument("--out", default="artifacts/eval.json")

    s = sub.add_parser("serve", help="OpenAI-compatible server (transformers backend, CPU)")
    s.add_argument("--base", default=DEFAULT_BASE)
    s.add_argument("--adapter", default="artifacts/adapter")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    m = sub.add_parser("merge", help="merge the adapter into the base model (for cloud deploys)")
    m.add_argument("--base", default=DEFAULT_BASE)
    m.add_argument("--adapter", default="artifacts/adapter")
    m.add_argument("--out", default="artifacts/merged")
    m.add_argument("--tar", help="also write a SageMaker-style model.tar.gz here")

    c = sub.add_parser("chat", help="send one ticket to an OpenAI-compatible endpoint")
    c.add_argument("ticket")
    c.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    c.add_argument("--model", default="triage-lora")

    args = ap.parse_args(argv)

    if args.cmd == "data":
        splits = build_splits()
        for name, rows in splits.items():
            print(f"{name}: {len(rows)} examples")
        for ex in splits["train"][:3]:
            print(json.dumps(ex.to_dict(), indent=1))
        return 0

    if args.cmd == "train":
        from .train import TrainConfig, train

        cfg = TrainConfig(
            base_model=args.base, out_dir=args.out, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, lora_r=args.rank, lora_alpha=2 * args.rank, qlora=args.qlora,
            max_steps=args.max_steps,
        )  # fmt: skip
        log = train(cfg)
        print(f"saved adapter to {args.out} ({log['trainable_params']:,} trainable parameters)")
        return 0

    if args.cmd == "eval":
        from .evaluate import run_eval, summary_table
        from .modeling import load_with_adapter

        tok, model = load_with_adapter(args.base, args.adapter)
        results = run_eval(tok, model, build_splits())
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1))
        print(summary_table(results))
        return 0

    if args.cmd == "serve":
        import uvicorn

        from .serve import create_app

        uvicorn.run(create_app(args.base, args.adapter), host=args.host, port=args.port)
        return 0

    if args.cmd == "merge":
        from .deploy.merge import make_tarball, merge_adapter

        out = merge_adapter(args.base, args.adapter, args.out)
        print(f"merged model written to {out}")
        if args.tar:
            print(f"tarball written to {make_tarball(out, args.tar)}")
        return 0

    if args.cmd == "chat":
        from .client import triage

        print(triage(args.ticket, base_url=args.base_url, model=args.model))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
