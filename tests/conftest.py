"""Offline fixtures: a tiny random GPT-2 and a tokenizer trained on the dataset text.

No network and no model downloads are needed, so the whole suite runs in CI in seconds.
"""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from llm_finetune_serve.data import build_splits, user_prompt


@pytest.fixture(scope="session")
def splits():
    return build_splits()


@pytest.fixture(scope="session")
def tok(splits):
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel()
    texts = [user_prompt(e.ticket) + e.target for e in splits["train"]]
    tk.train_from_iterator(
        texts,
        trainers.BpeTrainer(
            vocab_size=400,
            special_tokens=["<unk>", "<pad>", "<eos>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>"
    )
    fast.padding_side = "left"
    return fast


def make_tiny(vocab: int):
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=vocab, n_positions=256, n_embd=32, n_layer=2, n_head=2)
    return GPT2LMHeadModel(cfg)


@pytest.fixture()
def tiny_peft(tok):
    base = make_tiny(len(tok))
    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM")
    return get_peft_model(base, lora)
