import json

import torch

from llm_finetune_serve.evaluate import run_eval, score, summary_table
from llm_finetune_serve.modeling import parse_triage, render_messages, ticket_messages
from llm_finetune_serve.train import TrainConfig, collate, encode, lr_at, train


def test_encode_masks_the_prompt_and_keeps_the_answer(tok, splits):
    ex = splits["train"][0]
    row = encode(tok, ex, max_len=512)
    assert len(row["input_ids"]) == len(row["labels"])
    n_prompt = len(
        tok(render_messages(tok, ticket_messages(ex.ticket)), add_special_tokens=False)["input_ids"]
    )
    assert all(x == -100 for x in row["labels"][:n_prompt])
    answer_ids = [x for x in row["labels"] if x != -100]
    assert "category" in tok.decode(answer_ids) and ex.category in tok.decode(answer_ids)
    assert answer_ids[-1] == tok.eos_token_id


def test_collate_pads_with_ignored_labels(tok):
    rows = [{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]}, {"input_ids": [5], "labels": [5]}]
    b = collate(rows, pad_id=1)
    assert b["input_ids"].shape == (2, 3)
    assert b["labels"][1, 1:].tolist() == [-100, -100]
    assert b["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_learning_rate_schedule():
    cfg = TrainConfig(lr=1.0, warmup_steps=4)
    assert lr_at(0, 100, cfg) == 0.25
    assert lr_at(3, 100, cfg) == 1.0
    assert lr_at(99, 100, cfg) < 0.01


def test_training_lowers_loss_and_saves_adapter(tok, splits, tiny_peft, tmp_path):
    small = {k: v[:16] for k, v in splits.items()}
    cfg = TrainConfig(out_dir=str(tmp_path / "ad"), epochs=6, batch_size=8, lr=3e-3,
                      warmup_steps=2, max_len=160)  # fmt: skip
    log = train(cfg, small, tok, tiny_peft)
    first = sum(s["loss"] for s in log["steps"][:2]) / 2
    last = sum(s["loss"] for s in log["steps"][-2:]) / 2
    assert last < first
    assert log["val"][-1]["loss"] < log["val"][0]["loss"]
    assert log["trainable_params"] < log["total_params"] / 5
    saved = json.loads((tmp_path / "ad" / "train_log.json").read_text())
    assert len(saved["steps"]) == 12
    assert (tmp_path / "ad" / "adapter_config.json").exists()
    assert any(p.suffix == ".safetensors" for p in (tmp_path / "ad").iterdir())


def test_score_counts_valid_json_and_exact_matches(splits):
    ex = splits["train"][:3]
    good = ex[0].target
    wrong_priority = json.dumps({"category": ex[1].category, "priority": "nonsense"})
    r = score(ex, [good, wrong_priority, "not json"])
    assert r["valid_json"] == 2 / 3
    assert r["category_accuracy"] == 2 / 3
    assert r["priority_accuracy"] == 1 / 3
    assert r["exact_match"] == 1 / 3


def test_parse_triage_rejects_extra_keys_and_text():
    assert parse_triage('{"category": "refund", "priority": "low"}') == {
        "category": "refund", "priority": "low",
    }  # fmt: skip
    assert parse_triage('{"category": "refund"}') is None
    assert parse_triage("Sure! Here is the JSON") is None
    assert parse_triage("[1, 2]") is None


def test_run_eval_compares_base_and_tuned(tok, splits, tiny_peft):
    small = {"test_seen": splits["test_seen"][:4], "test_unseen": splits["test_unseen"][:4]}
    res = run_eval(tok, tiny_peft, small, max_new_tokens=8)
    assert set(res) == {"base", "tuned"}
    assert res["base"]["test_seen"]["n"] == 4
    assert "| tuned | test_unseen |" in summary_table(res)
    assert torch.is_tensor(next(tiny_peft.parameters()))
