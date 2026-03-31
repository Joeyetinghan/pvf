import re

import torch


# ---------- parsing ----------
def extract_mmlu_gold_choice(ans_text: str) -> str | None:
    """
    MMLU gold answers are typically just a single letter (A, B, C, or D).
    However, sometimes they might be embedded in text.
    This function aims to extract the single letter answer.
    """
    # Assuming the gold answer is a single uppercase letter
    # or perhaps "Answer: X" format.
    # We'll look for the last single uppercase letter A-D.
    matches = re.findall(r"\b([A-D])\b", ans_text)
    if matches:
        return matches[-1]
    return None


def extract_mmlu_pred_choice(pred_text: str) -> str | None:
    """
    Robustly parse model output for MMLU:
    - Look for patterns like "Answer: (A)" or "The answer is B".
    - Fallback to looking for uppercase-only single letters A-D (to avoid matching article "a").
    """
    # Common patterns in chain-of-thought responses (case-insensitive)
    patterns_ci = [
        r"[Aa]nswer[:\s]+\(?([A-Da-d])\)?",
        r"[Tt]he\s+(?:correct\s+)?answer\s+is[:\s]*\(?([A-Da-d])\)?",
        r"[Cc]hoice[:\s]+\(?([A-Da-d])\)?",
        r"[Oo]ption[:\s]+\(?([A-Da-d])\)?",
        r"\\boxed\{([A-Da-d])\}",  # LaTeX boxed format
        r"\*\*([A-Da-d])\.",  # Markdown bold like **B.
        r"\*\*([A-Da-d])\*\*",  # Markdown bold like **B**
    ]

    # Multi-line pattern: "answer is:" followed by whitespace/newlines then the letter
    multiline_patterns = [
        r"[Aa]nswer\s+is:?\s+\*?\*?([A-Da-d])[\.\)]",  # Answer is: B. or B)
        r"[Tt]he\s+(?:correct\s+)?answer\s+is:?\s+\*?\*?([A-Da-d])[\.\)]",
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

    # Fallback: look for letter in specific answer contexts
    # Avoid matching "A." at start of option listings by requiring specific context
    final_answer_patterns = [
        r"(?:is|be|choose|select|pick)\s+([A-D])\b",  # "is A", "choose B"
        r"\b([A-D])\s*$",  # Letter at very end of text
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
        options: List of 4 answer options [A_text, B_text, C_text, D_text]
        tokenizer: The tokenizer
        device: Device to put tensor on
    """
    A, B, C, D = options[0], options[1], options[2], options[3]

    text = f"""Answer the following multiple-choice question. Think step by step, then provide your final answer in the format "Answer: X" where X is the letter of your choice (A, B, C, or D).

Question: {question}

Options:
A. {A}
B. {B}
C. {C}
D. {D}"""

    msgs = [{"role": "user", "content": text}]
    s = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(s)["input_ids"]
    return torch.tensor(ids, device=device).unsqueeze(0)


def encode_for_llada_base(question: str, options: list, tokenizer, device: str = "cuda") -> torch.LongTensor:
    """
    Returns a tensor of shape (1, L) ready for your generate(...).

    Args:
        question: The question text
        options: List of 4 answer options [A_text, B_text, C_text, D_text]
        tokenizer: The tokenizer
        device: Device to put tensor on
    """
    A, B, C, D = options[0], options[1], options[2], options[3]

    text = f"""Question: {question}

A. {A}
B. {B}
C. {C}
D. {D}

Answer:"""
    ids = tokenizer(text)["input_ids"]
    return torch.tensor(ids, device=device).unsqueeze(0)


# ---------- accuracy ----------
def accuracy_MMLU(pred_texts: list[str], gold_texts: list[str]) -> float:
    """
    Returns accuracy in [0, 1]. No printing.
    """
    assert len(pred_texts) == len(gold_texts)
    correct = 0
    for pred, gold in zip(pred_texts, gold_texts):
        g = extract_mmlu_gold_choice(gold)
        p = extract_mmlu_pred_choice(pred)
        # Compare the extracted choice letters
        ok = (g is not None) and (p is not None) and (p == g)
        correct += int(ok)
    return 0.0 if len(pred_texts) == 0 else correct / len(pred_texts)
