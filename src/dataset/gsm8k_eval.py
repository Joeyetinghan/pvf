# src/eval/gsm8k_eval.py
import re

import torch


# ---------- regex helpers ----------
NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")  # integers/decimals with optional commas
FRAC_RE = re.compile(r"-?\d+\s*/\s*\d+")  # simple fractions like 3/4


# ---------- parsing ----------
def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def extract_gold_number(ans_text: str) -> float | None:
    """
    GSM8K gold answers end with '#### <number>'.
    Try that first; fall back to the last number or fraction anywhere.
    """
    if "####" in ans_text:
        tail = ans_text.split("####")[-1]
        m = NUM_RE.search(tail)
        if m:
            return _to_float(m.group(0))
        m = FRAC_RE.search(tail)
        if m:
            a, b = m.group(0).split("/")
            return float(a) / float(b)

    nums = NUM_RE.findall(ans_text)
    if nums:
        return _to_float(nums[-1])

    fracs = FRAC_RE.findall(ans_text)
    if fracs:
        a, b = fracs[-1].split("/")
        return float(a) / float(b)

    return None


def extract_pred_number(pred_text: str) -> float | None:
    """
    Robustly parse model output:
    - use the last numeric token if multiple appear
    - support simple fractions as fallback
    """
    nums = NUM_RE.findall(pred_text)
    if nums:
        return _to_float(nums[-1])
    fracs = FRAC_RE.findall(pred_text)
    if fracs:
        a, b = fracs[-1].split("/")
        return float(a) / float(b)
    return None


# ---------- prompt + encoding ----------
'''
def build_gsm8k_prompt(q: str) -> str:
    """
    MODIFIED: This prompt now asks for a step-by-step explanation (Chain-of-Thought)
    which is the standard, high-accuracy way to evaluate GSM8K.
    """
    return (
        "Solve the problem, showing your reasoning step-by-step.\n\n"
        f"Question:\n{q.strip()}\n\nAnswer:"
    )
'''


def encode_for_llada_instruct(text: str, tokenizer, device: str = "cuda") -> torch.LongTensor:
    """
    Returns a tensor of shape (1, L) ready for your generate(...).
    """
    msgs = [{"role": "user", "content": text}]
    s = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(s)["input_ids"]
    return torch.tensor(ids, device=device).unsqueeze(0)


def encode_for_llada_base(text: str, tokenizer, device: str = "cuda") -> torch.LongTensor:
    """
    Returns a tensor of shape (1, L) ready for your generate(...).
    """
    ids = tokenizer(text)["input_ids"]
    return torch.tensor(ids, device=device).unsqueeze(0)


# ---------- accuracy ----------
def accuracy_gsm8k(pred_texts: list[str], gold_texts: list[str]) -> float:
    """
    Returns accuracy in [0, 1]. No printing.
    """
    assert len(pred_texts) == len(gold_texts)
    correct = 0
    for pred, gold in zip(pred_texts, gold_texts):
        g = extract_gold_number(gold)
        p = extract_pred_number(pred)
        ok = (g is not None) and (p is not None) and (abs(p - g) <= max(1e-6, 1e-6 * abs(g)))
        correct += int(ok)
    return 0.0 if len(pred_texts) == 0 else correct / len(pred_texts)
