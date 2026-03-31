"""
https://huggingface.co/datasets/allenai/winogrande

Dataset sample (allenai/winogrande):
{
    "sentence": "Kenneth went cheap on the gemstone present for Michael and _ was understanding about being a cheapskate.",
    "option1": "Kenneth",
    "option2": "Michael",
    "answer": 1
}
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import re

import torch


DEBUG_WINOGRANDE = True

# ---------- parsing ----------


_OPTION_PATTERNS = [
    (1, re.compile(r"\boption\s*(?:number\s*)?(?:1|one)\b", re.IGNORECASE)),
    (2, re.compile(r"\boption\s*(?:number\s*)?(?:2|two)\b", re.IGNORECASE)),
    (1, re.compile(r"\bchoice\s*(?:1|one)\b", re.IGNORECASE)),
    (2, re.compile(r"\bchoice\s*(?:2|two)\b", re.IGNORECASE)),
    (1, re.compile(r"\bblank\s*1\b", re.IGNORECASE)),
    (2, re.compile(r"\bblank\s*2\b", re.IGNORECASE)),
    (1, re.compile(r"\bfirst\b", re.IGNORECASE)),
    (2, re.compile(r"\bsecond\b", re.IGNORECASE)),
    (1, re.compile(r"\b(?:Answer|Ans)\s*[:\-]?\s*1\b", re.IGNORECASE)),
    (2, re.compile(r"\b(?:Answer|Ans)\s*[:\-]?\s*2\b", re.IGNORECASE)),
]

_DIGIT_PATTERN = re.compile(r"\b([12])\b")
_LETTER_PATTERN = re.compile(r"\b([AB])\b", re.IGNORECASE)


def _normalize_for_option_match(text: str | None) -> str:
    if not text:
        return ""
    lowered = text.lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def _extract_index_from_text(text: str | int | None) -> int | None:
    if text is None:
        return None
    if isinstance(text, int):
        return text if text in (1, 2) else None
    matches = _DIGIT_PATTERN.findall(text)
    if matches:
        return int(matches[-1])
    letter_matches = _LETTER_PATTERN.findall(text)
    if letter_matches:
        letter = letter_matches[-1].upper()
        return 1 if letter == "A" else 2 if letter == "B" else None
    return None


def parse_winogrande_gold_blob(gold_blob: str) -> tuple[int | None, list[str]]:
    """
    Gold strings may be raw digits ("1") or JSON blobs like:
    {"answer": 2, "options": ["Kenneth", "Michael"]}.
    """
    if not gold_blob:
        return None, []

    gold_blob = gold_blob.strip()
    if not gold_blob:
        return None, []

    if gold_blob.startswith("{"):
        try:
            payload = json.loads(gold_blob)
        except json.JSONDecodeError:
            return _extract_index_from_text(gold_blob), []
        answer = payload.get("answer")
        options_raw = payload.get("options") or []
        options = [str(opt) for opt in options_raw][:2]
        parsed = _extract_index_from_text(answer)
        if DEBUG_WINOGRANDE:
            print(f"[WINO][GOLD] payload: {payload} -> idx={parsed}")
        return parsed, options

    parsed = _extract_index_from_text(gold_blob)
    if DEBUG_WINOGRANDE:
        print(f"[WINO][GOLD] raw='{gold_blob}' -> idx={parsed}")
    return parsed, []


def extract_winogrande_pred_choice(pred_text: str, options: Sequence[str] | None = None) -> int | None:
    """
    Parse the model output and return 1 or 2 if determinable.
    """
    if not pred_text:
        return None

    for idx, pattern in _OPTION_PATTERNS:
        if pattern.search(pred_text):
            if DEBUG_WINOGRANDE:
                print(f"[WINO][PRED] matched pattern '{pattern.pattern}' -> {idx}")
            return idx

    matches = _DIGIT_PATTERN.findall(pred_text)
    if matches:
        value = int(matches[-1])
        if DEBUG_WINOGRANDE:
            print(f"[WINO][PRED] digit fallback -> {value}")
        return value

    letter_matches = _LETTER_PATTERN.findall(pred_text)
    if letter_matches:
        letter = letter_matches[-1].upper()
        if letter == "A":
            if DEBUG_WINOGRANDE:
                print("[WINO][PRED] letter fallback -> 1")
            return 1
        if letter == "B":
            if DEBUG_WINOGRANDE:
                print("[WINO][PRED] letter fallback -> 2")
            return 2

    if options:
        lowered_pred = pred_text.lower()
        for idx, option in enumerate(options, start=1):
            normalized = option.strip().lower()
            if normalized and normalized in lowered_pred:
                if DEBUG_WINOGRANDE:
                    print(f"[WINO][PRED] matched option text '{option}' -> {idx}")
                return idx
        normalized_pred = _normalize_for_option_match(pred_text)
        for idx, option in enumerate(options, start=1):
            opt_norm = _normalize_for_option_match(option)
            if opt_norm and (opt_norm in normalized_pred or normalized_pred in opt_norm):
                if DEBUG_WINOGRANDE:
                    print(f"[WINO][PRED] normalized match '{option}' -> {idx}")
                return idx

    if DEBUG_WINOGRANDE:
        snippet = pred_text.replace("\n", " ")[:200]
        print(f"[WINO][PRED] unable to parse from '{snippet}'")
    return None


# ---------- prompt + encoding ----------


def _normalize_options(option1: str, option2: str) -> list[str]:
    return [str(option1).strip(), str(option2).strip()]


def encode_for_llada_instruct(
    sentence: str,
    option1: str,
    option2: str,
    tokenizer,
    device: str = "cuda",
) -> torch.LongTensor:
    """
    Build a chat-style prompt for instruct-tuned models (WinoGrande).
    Generation + parsing friendly: enforce a strict final answer line.
    """
    opt1, opt2 = _normalize_options(option1, option2)

    text = f"""You are an expert at commonsense pronoun resolution. Fill in the blank (_) with the option that makes the sentence most coherent.

Sentence:
{sentence.strip()}

Options:
Option 1: {opt1}
Option 2: {opt2}

Instruction:
1. Briefly explain your reasoning. Keep it concise and direct.
2. On the final line, output exactly 'Answer: Option 1' or 'Answer: Option 2' (no other text).

Format:
Analysis: [Your concise reasoning]
Answer: Option [1/2]

Replace 'Option [1/2]' in the format block with either 'Option 1' or 'Option 2' exactly—no additional text.
"""
    msgs = [{"role": "user", "content": text}]
    s = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(s)["input_ids"]
    return torch.tensor(ids, device=device).unsqueeze(0)


def encode_for_llada_base(
    sentence: str,
    option1: str,
    option2: str,
    tokenizer,
    device: str = "cuda",
) -> torch.LongTensor:
    """
    Build a plain prompt for base models.
    """
    opt1, opt2 = _normalize_options(option1, option2)
    prompt = f"""Sentence: {sentence.strip()}

Option 1: {opt1}
Option 2: {opt2}

Answer (respond EXACTLY with 'Option 1' or 'Option 2'):"""
    ids = tokenizer(prompt)["input_ids"]
    return torch.tensor(ids, device=device).unsqueeze(0)


# ---------- accuracy ----------


def accuracy_winogrande(pred_texts: list[str], gold_texts: list[str]) -> float:
    """
    Compute accuracy for Winogrande predictions.
    """
    assert len(pred_texts) == len(gold_texts)
    correct = 0
    for pred, gold in zip(pred_texts, gold_texts, strict=True):
        gold_idx, options = parse_winogrande_gold_blob(gold)
        pred_idx = extract_winogrande_pred_choice(pred, options)
        correct += int(gold_idx is not None and pred_idx is not None and gold_idx == pred_idx)
    return 0.0 if not pred_texts else correct / len(pred_texts)
