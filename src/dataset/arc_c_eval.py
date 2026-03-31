"""
https://huggingface.co/datasets/allenai/ai2_arc

{
    "answerKey": "B",
    "choices": {
        "label": ["A", "B", "C", "D"],
        "text": [
            "Shady areas increased.",
            "Food sources increased.",
            "Oxygen levels increased.",
            "Available water increased."
        ]
    },
    "id": "Mercury_SC_405487",
    "question": "One year, the oak trees in a park began producing more acorns than usual. \
        The next year, the population of chipmunks in the park also increased. \
        Which best explains why there were more chipmunks the next year?"
}
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import re

import torch


DEBUG_ARC = True


# ---------- parsing ----------


CHOICE_PATTERN = re.compile(r"\b([A-E])\b", re.IGNORECASE)


def extract_arc_gold_choice(ans_text: str) -> str | None:
    """
    ARC gold answers are single letters (typically A-D, occasionally E).
    """
    if not ans_text:
        return None
    matches = CHOICE_PATTERN.findall(ans_text)
    result = matches[-1].upper() if matches else None
    if DEBUG_ARC:
        snippet = (ans_text or "").replace("\n", " ")[:200]
        print(f"[ARC][GOLD] text='{snippet}' -> {result}")
    return result


def extract_arc_pred_choice(pred_text: str) -> str | None:
    """
    Parse model output for a final choice letter.

    Supports common answer templates like:
      - "Answer: B"
      - "The correct answer is (C)."
      - "\\boxed{D}"
      - Trailing standalone letter.
    """
    if not pred_text:
        return None

    patterns_ci = [
        r"[Aa]nswer[:\s]+\(?([A-Ea-e])\)?",
        r"[Tt]he\s+(?:correct\s+)?answer\s+is[:\s]*\(?([A-Ea-e])\)?",
        r"[Cc]hoice[:\s]+\(?([A-Ea-e])\)?",
        r"[Oo]ption[:\s]+\(?([A-Ea-e])\)?",
        r"\\boxed\{([A-Ea-e])\}",
        r"\*\*([A-Ea-e])\.\*\*?",
        r"\*\*([A-Ea-e])\*\*",
    ]

    for pattern in patterns_ci:
        matches = re.findall(pattern, pred_text)
        if matches:
            choice = matches[-1].upper()
            if DEBUG_ARC:
                print(f"[ARC][PRED] matched pattern '{pattern}' -> {choice}")
            return choice

    trailing_patterns = [
        r"(?:is|be|choose|select|pick)\s+([A-E])\b",
        r"\b([A-E])\s*$",
    ]

    for pattern in trailing_patterns:
        matches = re.findall(pattern, pred_text, re.IGNORECASE | re.MULTILINE)
        if matches:
            choice = matches[-1].upper()
            if DEBUG_ARC:
                print(f"[ARC][PRED] matched trailing pattern '{pattern}' -> {choice}")
            return choice

    matches = CHOICE_PATTERN.findall(pred_text)
    choice = matches[-1].upper() if matches else None
    if DEBUG_ARC:
        snippet = pred_text.replace("\n", " ")[:200]
        print(f"[ARC][PRED] fallback text='{snippet}' -> {choice}")
    return choice


# ---------- prompt + encoding ----------


def _normalize_choices(choices: Sequence | dict) -> list[tuple[str, str]]:
    """
    Accepts either the native HF dict (with `label` / `text`) or an iterable of (label, text).
    """
    if isinstance(choices, dict):
        labels: Sequence = choices.get("label") or []
        texts: Sequence = choices.get("text") or []
        if len(labels) != len(texts):
            raise ValueError("ARC choices must have matching label/text lengths.")
        normalized = [(str(labels[i]), str(texts[i])) for i in range(len(labels))]
    else:
        normalized = []
        for entry in choices:
            if isinstance(entry, dict):
                label = str(entry.get("label", "")).strip()
                text = str(entry.get("text", "")).strip()
            elif isinstance(entry, Sequence) and len(entry) == 2:
                label, text = entry  # type: ignore[misc]
                label = str(label).strip()
                text = str(text).strip()
            else:
                raise ValueError("Unsupported choice format.")
            normalized.append((label, text))

    if not normalized:
        raise ValueError("No ARC choices provided.")
    return normalized


def _format_choices_block(choices: Iterable[tuple[str, str]]) -> str:
    return "\n".join(f"{label}. {text}" for label, text in choices)


def encode_for_llada_instruct(question: str, choices, tokenizer, device: str = "cuda") -> torch.LongTensor:
    """
    Returns a tensor of shape (1, L) for ARC-Challenge (ARC-C) using instruct models.
    Generation + parsing friendly, MMLU-style formatting.
    """
    normalized = _normalize_choices(choices)
    choices_block = _format_choices_block(normalized)

    text = f"""You are an expert science tutor. Answer the multiple-choice question below.

Question:
{question.strip()}

Options:
{choices_block}

Instruction:
1. Briefly explain your reasoning for the correct choice. Keep the analysis concise and direct.
2. State the final answer clearly at the end as a single option letter.

Format:
Analysis: [Your concise reasoning]
Answer: [A/B/C/D]
"""
    msgs = [{"role": "user", "content": text}]
    s = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(s)["input_ids"]
    return torch.tensor(ids, device=device).unsqueeze(0)


def encode_for_llada_base(question: str, choices, tokenizer, device: str = "cuda") -> torch.LongTensor:
    """
    Returns a tensor of shape (1, L) for ARC-Challenge when using base models.
    """
    normalized = _normalize_choices(choices)
    choices_block = _format_choices_block(normalized)
    prompt = f"""Question: {question.strip()}

{choices_block}

Answer:"""
    ids = tokenizer(prompt)["input_ids"]
    return torch.tensor(ids, device=device).unsqueeze(0)


# ---------- accuracy ----------


def accuracy_arc_c(pred_texts: list[str], gold_texts: list[str]) -> float:
    """
    Compute accuracy for ARC-Challenge predictions.
    """
    assert len(pred_texts) == len(gold_texts)
    correct = 0
    for pred, gold in zip(pred_texts, gold_texts, strict=True):
        g = extract_arc_gold_choice(gold)
        p = extract_arc_pred_choice(pred)
        correct += int(g is not None and p is not None and g == p)
    return 0.0 if not pred_texts else correct / len(pred_texts)
