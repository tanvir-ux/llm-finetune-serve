"""Compare the base model with the LoRA-tuned model on the held-out test sets."""

from __future__ import annotations

from .data import Example
from .modeling import generate_texts, parse_triage, render_messages, ticket_messages


def score(examples: list[Example], outputs: list[str]) -> dict:
    n = len(examples)
    valid = cat = pri = exact = 0
    rows = []
    for ex, out in zip(examples, outputs, strict=True):
        parsed = parse_triage(out)
        ok_v = parsed is not None
        ok_c = ok_v and parsed["category"] == ex.category
        ok_p = ok_v and parsed["priority"] == ex.priority
        valid += ok_v
        cat += ok_c
        pri += ok_p
        exact += ok_c and ok_p
        rows.append({"ticket": ex.ticket, "expected": ex.target, "output": out,
                     "valid_json": ok_v, "exact": bool(ok_c and ok_p)})  # fmt: skip
    return {
        "n": n,
        "valid_json": valid / n,
        "category_accuracy": cat / n,
        "priority_accuracy": pri / n,
        "exact_match": exact / n,
        "rows": rows,
    }


def run_eval(tok, model, splits: dict[str, list[Example]], names=("test_seen", "test_unseen"),
             max_new_tokens: int = 40) -> dict:  # fmt: skip
    """Evaluate the base model (adapter disabled) and the tuned model on each test set."""
    results: dict = {"base": {}, "tuned": {}}
    for name in names:
        prompts = [render_messages(tok, ticket_messages(e.ticket)) for e in splits[name]]
        with model.disable_adapter():
            base_out = generate_texts(tok, model, prompts, max_new_tokens)
        tuned_out = generate_texts(tok, model, prompts, max_new_tokens)
        results["base"][name] = score(splits[name], base_out)
        results["tuned"][name] = score(splits[name], tuned_out)
    return results


def summary_table(results: dict) -> str:
    lines = ["| model | test set | n | valid JSON | category | priority | exact match |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip
    for model in ("base", "tuned"):
        for name, r in results[model].items():
            lines.append(
                f"| {model} | {name} | {r['n']} | {r['valid_json']:.0%} | "
                f"{r['category_accuracy']:.0%} | {r['priority_accuracy']:.0%} | "
                f"{r['exact_match']:.0%} |"
            )
    return "\n".join(lines)
