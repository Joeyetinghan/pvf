import re

import torch


# ---------- parsing ----------
def extract_mmlu_pro_gold_choice(ans_text: str) -> str | None:
    """
    MMLU-Pro gold answers are typically a single letter (A-J for 10 choices).
    This function extracts the single letter answer.
    """
    # Look for the last single uppercase letter A-J
    matches = re.findall(r"\b([A-J])\b", ans_text)
    if matches:
        return matches[-1]
    return None


def extract_mmlu_pro_pred_choice(pred_text: str) -> str | None:
    """
    Robustly parse model output for MMLU-Pro:
    - Look for patterns like "Answer: (A)" or "The answer is B".
    - Fallback to looking for uppercase-only single letters A-J (to avoid matching articles like "a", "i").
    """
    # Common patterns in chain-of-thought responses
    patterns_ci = [
        r"[Aa]nswer[:\s]+\(?([A-Ja-j])\)?",
        r"[Tt]he\s+(?:correct\s+)?answer\s+is[:\s]*\(?([A-Ja-j])\)?",
        r"[Cc]hoice[:\s]+\(?([A-Ja-j])\)?",
        r"[Oo]ption[:\s]+\(?([A-Ja-j])\)?",
        r"\\boxed\{([A-Ja-j])\}",  # LaTeX boxed format
        r"\*\*([A-Ja-j])\.",  # Markdown bold like **H.
        r"\*\*([A-Ja-j])\*\*",  # Markdown bold like **H**
    ]

    # Multi-line pattern: "answer is:" followed by whitespace/newlines then the letter
    multiline_patterns = [
        r"[Aa]nswer\s+is:?\s+\*?\*?([A-Ja-j])[\.\)]",  # Answer is: E. or E)
        r"\b[Tt]he\s+(?:correct\s+)?answer\s+is:?\s+\*?\*?([A-Ja-j])[\.\)]",  # \b prevents matching "The" in "Therefore"
        r"(?:correct\s+)?answer\s+is:?\s+\*?\*?([A-Ja-j])[\.\)]",  # More flexible: just "answer is: H."
    ]

    for p in patterns_ci:
        matches = re.findall(p, pred_text)
        if matches:
            return matches[-1].upper()

    # Try multiline patterns
    for p in multiline_patterns:
        matches = re.findall(p, pred_text, re.MULTILINE)
        if matches:
            return matches[-1].upper()

    # Fallback: look for letter followed by period at end of sentence or standalone
    # Avoid matching "A." at start of option listings by requiring specific context
    final_answer_patterns = [
        r"(?:is|be|choose|select|pick)\s+([A-J])\b",  # "is A", "choose B"
        r"\b([A-J])\s*$",  # Letter at very end of text
    ]

    for p in final_answer_patterns:
        matches = re.findall(p, pred_text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    return None


# ---------- prompt + encoding ----------


def encode_for_llada_instruct(question: str, options: list, tokenizer, device: str = "cuda") -> torch.LongTensor:
    """
    Returns a tensor of shape (1, L) ready for your generate(...).

    Args:
        question: The question text
        options: List of answer options (up to 10 for MMLU-Pro)
        tokenizer: The tokenizer
        device: Device to put tensor on
    """
    choice_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    options_text = "\n".join([f"{label}. {opt}" for label, opt in zip(choice_labels[: len(options)], options)])

    text = f"""You are an expert. Answer the multiple-choice question below.

Question:
{question}

Options:
{options_text}

Instruction:
1. Briefly explain your reasoning for the correct choice. Keep the analysis concise and direct.
2. State the final answer clearly at the end.

Format:
Analysis: [Your concise reasoning]
Answer: [Option]
"""
    msgs = [{"role": "user", "content": text}]
    s = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(s)["input_ids"]
    return torch.tensor(ids, device=device).unsqueeze(0)


def encode_for_llada_base(question: str, options: list, tokenizer, device: str = "cuda") -> torch.LongTensor:
    """
    Returns a tensor of shape (1, L) ready for your generate(...).

    Args:
        question: The question text
        options: List of answer options (up to 10 for MMLU-Pro)
        tokenizer: The tokenizer
        device: Device to put tensor on
    """
    choice_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    options_text = "\n".join([f"{label}. {opt}" for label, opt in zip(choice_labels[: len(options)], options)])

    text = f"""Question: {question}

{options_text}

Answer:"""
    ids = tokenizer(text)["input_ids"]
    return torch.tensor(ids, device=device).unsqueeze(0)


# ---------- accuracy ----------
def accuracy_MMLU_Pro(pred_texts: list[str], gold_texts: list[str]) -> float:
    """
    Returns accuracy in [0, 1]. No printing.
    """
    assert len(pred_texts) == len(gold_texts)
    correct = 0
    for pred, gold in zip(pred_texts, gold_texts):
        g = extract_mmlu_pro_gold_choice(gold)
        p = extract_mmlu_pro_pred_choice(pred)
        # Compare the extracted choice letters
        ok = (g is not None) and (p is not None) and (p == g)
        correct += int(ok)
    return 0.0 if len(pred_texts) == 0 else correct / len(pred_texts)
