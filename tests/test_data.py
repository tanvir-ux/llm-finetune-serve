import json

from llm_finetune_serve import data


def test_splits_are_deterministic_and_sized(splits):
    again = data.build_splits()
    assert [e.ticket for e in again["train"]] == [e.ticket for e in splits["train"]]
    assert {k: len(v) for k, v in splits.items()} == {
        "train": 320, "val": 40, "test_seen": 60, "test_unseen": 40,
    }  # fmt: skip


def test_no_ticket_leaks_from_train_into_eval_sets(splits):
    train = {e.ticket for e in splits["train"]}
    for name in ("val", "test_seen", "test_unseen"):
        assert not train & {e.ticket for e in splits[name]}, name


def test_labels_match_the_text(splits):
    cues = {p: set(c[0]) | set(c[1]) for p, c in data.CUES.items()}
    cores = {c: v[0] + v[1] for c, v in data.CORE.items()}
    for e in splits["train"] + splits["test_unseen"]:
        assert any(cue in e.ticket for cue in cues[e.priority])
        assert any(t.split("{")[0] in e.ticket for t in cores[e.category])


def test_unseen_set_uses_only_held_out_wording(splits):
    train_cues = {c for p in data.CUES.values() for c in p[0]}
    train_core_starts = {t.split("{")[0] for v in data.CORE.values() for t in v[0] if "{" in t}
    for e in splits["test_unseen"]:
        assert not any(cue in e.ticket for cue in train_cues)
        assert not any(s and s in e.ticket for s in train_core_starts if len(s) > 20)


def test_all_label_pairs_are_covered(splits):
    pairs = {(e.category, e.priority) for e in splits["train"]}
    assert len(pairs) == len(data.CATEGORIES) * len(data.PRIORITIES)


def test_target_is_strict_json(splits):
    obj = json.loads(splits["train"][0].target)
    assert set(obj) == {"category", "priority"}
