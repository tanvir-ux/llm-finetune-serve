"""Synthetic support-ticket triage dataset.

Task: read a customer ticket and reply with strict JSON: {"category": ..., "priority": ...}.
Everything is generated from templates with a fixed seed, so the dataset is reproducible and
contains no real customer data. Two test sets are produced:

* ``test_seen``: new random fills of the *same* sentence templates used for training.
* ``test_unseen``: sentence templates and urgency phrases that never appear in training.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass

CATEGORIES = ("shipping", "billing", "refund", "account", "technical")
PRIORITIES = ("low", "normal", "high", "urgent")

SYSTEM = "You triage customer support tickets."
INSTRUCTION = (
    'Reply with JSON only: {"category": shipping|billing|refund|account|technical, '
    '"priority": low|normal|high|urgent}.'
)

# Core sentences per category: (train templates, held-out templates)
CORE: dict[str, tuple[list[str], list[str]]] = {
    "shipping": (
        [
            "My order {n} has not arrived yet.",
            "The tracking page for order {n} has not changed in a week.",
            "Order {n} shows as delivered but the package is not here.",
            "I think my parcel {n} went to the wrong address.",
            "The courier keeps rescheduling the delivery of order {n}.",
        ],
        [
            "Where is my {item}? Order {n} was shipped days ago.",
            "Still waiting for the {item} from order {n}.",
        ],
    ),
    "billing": (
        [
            "I was charged twice for order {n}.",
            "There is a strange fee on my invoice for order {n}.",
            "The total on my receipt for order {n} does not match the price I saw.",
            "My card was billed after I cancelled subscription {n}.",
            "Please explain the extra charge on invoice {n}.",
        ],
        [
            "Why did I get an unexpected bill for the {item} on order {n}?",
            "My statement lists an amount for order {n} that I never agreed to.",
        ],
    ),
    "refund": (
        [
            "I would like a refund for order {n}.",
            "I returned the {item} from order {n} but the money has not come back.",
            "Please send my money back for order {n}, the {item} does not work for me.",
            "How do I return the {item} I bought on order {n}?",
            "I want to cancel order {n} and get my money back.",
        ],
        [
            "Can you reimburse me for the {item} on order {n}?",
            "The {item} from order {n} was not what I wanted, I need to send it back.",
        ],
    ),
    "account": (
        [
            "I cannot log in to my account.",
            "I forgot my password and the reset email never arrives.",
            "Please change the email address on my account to a new one.",
            "My account was locked after too many sign-in attempts.",
            "How can I delete my account and my data?",
        ],
        [
            "The sign-in page keeps rejecting my credentials.",
            "I need to update the phone number saved in my profile.",
        ],
    ),
    "technical": (
        [
            "The app crashes whenever I open the {item} page.",
            "The {item} feature shows an error message and then freezes.",
            "The website will not load the checkout page on my phone.",
            "Notifications from the app stopped working after the last update.",
            "The {item} screen is blank after I sign in.",
        ],
        [
            "Something is broken: the {item} page keeps spinning forever.",
            "I get an error code every time I try to use the {item} tool.",
        ],
    ),
}

# Urgency cues per priority: (train phrases, held-out phrases)
CUES: dict[str, tuple[list[str], list[str]]] = {
    "low": (
        [
            "No rush, whenever you have time.",
            "This is not urgent at all.",
            "Just letting you know.",
        ],
        ["Take your time with this one.", "Only a minor thing, really."],
    ),
    "normal": (
        [
            "Please let me know what to do.",
            "Thanks for your help.",
            "I would appreciate an answer.",
        ],
        ["Could you take a look, please?", "Grateful for any guidance."],
    ),
    "high": (
        [
            "This is the second time I am writing about it.",
            "I need this solved this week.",
            "I am quite upset about this.",
        ],
        ["I have been waiting far too long.", "This is becoming a serious problem for me."],
    ),
    "urgent": (
        [
            "This is urgent, please answer today!",
            "I need help immediately, it is an emergency.",
            "Fix this ASAP, I am about to lose a client.",
        ],
        ["Drop everything and look at this now.", "Critical issue, I cannot wait any longer!"],
    ),
}

ITEMS = ["camera", "headphones", "dashboard", "invoice", "report", "keyboard", "tablet", "profile"]


@dataclass(frozen=True)
class Example:
    ticket: str
    category: str
    priority: str

    @property
    def target(self) -> str:
        return json.dumps({"category": self.category, "priority": self.priority})

    def to_dict(self) -> dict[str, str]:
        return asdict(self) | {"target": self.target}


def _make(rng: random.Random, category: str, priority: str, held_out: bool) -> Example:
    core_pool = CORE[category][1 if held_out else 0]
    cue_pool = CUES[priority][1 if held_out else 0]
    core = rng.choice(core_pool).format(n=rng.randint(1000, 9999), item=rng.choice(ITEMS))
    cue = rng.choice(cue_pool)
    ticket = f"{core} {cue}" if rng.random() < 0.5 else f"{cue} {core}"
    return Example(ticket, category, priority)


def generate(n: int, seed: int, held_out: bool = False) -> list[Example]:
    """Generate ``n`` balanced examples (all 20 category/priority pairs in rotation)."""
    rng = random.Random(seed)
    pairs = [(c, p) for c in CATEGORIES for p in PRIORITIES]
    out = [_make(rng, *pairs[i % len(pairs)], held_out) for i in range(n)]
    rng.shuffle(out)
    return out


def _fresh(n: int, seed: int, exclude: set[str], held_out: bool = False) -> list[Example]:
    """Generate examples whose ticket text does not occur in ``exclude`` (no train leakage)."""
    pool = [e for e in generate(n * 6, seed, held_out) if e.ticket not in exclude]
    unique: dict[str, Example] = {}
    for e in pool:
        unique.setdefault(e.ticket, e)
    return list(unique.values())[:n]


def build_splits(seed: int = 7) -> dict[str, list[Example]]:
    """Fixed splits: 320 train, 40 validation, 60 seen-style test, 40 unseen-wording test.

    Validation and test tickets never repeat a training ticket verbatim.
    """
    train = generate(320, seed)
    seen = {e.ticket for e in train}
    return {
        "train": train,
        "val": _fresh(40, seed + 1, seen),
        "test_seen": _fresh(60, seed + 2, seen),
        "test_unseen": _fresh(40, seed + 3, seen, held_out=True),
    }


def user_prompt(ticket: str) -> str:
    return f"{INSTRUCTION}\n\nTicket: {ticket}"
