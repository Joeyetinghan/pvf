import json
import re

from datasets import load_dataset
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ==============================
# 1. Answer Extraction & Normalization
# (From lm-evaluation-harness - https://github.com/EleutherAI/lm-evaluation-harness)
# ==============================


def last_boxed_only_string(string):
    """
    Extract the last \boxed{...} or \fbox{...} from a string.
    Handles nested braces correctly.
    """
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    """
    Remove the \boxed{} wrapper and return the content inside.
    Handles both \boxed{...} and \boxed ... formats.
    """
    if s is None:
        return None

    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] == left:
            return s[len(left) :]

    left = "\\boxed{"

    if s[: len(left)] == left and s[-1] == "}":
        return s[len(left) : -1]

    return s


def fix_fracs(string):
    """
    Fix fraction notation: \frac1b -> \frac{1}{b}, \frac12 -> \frac{1}{2}
    """
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) == 0:
                continue
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    """
    Convert simple fractions like 1/2 to \frac{1}{2}
    """
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == f"{a}/{b}"
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except (AssertionError, ValueError):
        return string


def remove_right_units(string):
    """
    Remove units from the right side of the answer.
    """
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def fix_sqrt(string):
    r"""
    Fix sqrt notation: \sqrt3 -> \sqrt{3}
    """
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) > 0 and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    """
    Comprehensive string normalization for math expressions.
    From lm-evaluation-harness.
    """
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{."
    # Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc.
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    # Remove leading zeros in base notation (e.g., 0506_7 -> 506_7)
    string = re.sub(r"^0+([1-9][0-9]*_[0-9]+)$", r"\1", string)

    # Also handle leading zeros in plain integers (but not decimals)
    # e.g., 007 -> 7, but not 0.7 -> .7
    if re.match(r"^0+[1-9][0-9]*$", string):
        string = string.lstrip("0")

    return string


def try_numeric_equiv(str1, str2):
    """
    Try to compare two strings as numbers, handling:
    - Trailing zeros (1.4 vs 1.4000)
    - Dollar signs ($0.50 vs 0.5)
    - Simple fractions that evaluate to the same number
    """
    # Remove dollar signs
    s1 = str1.replace("\\$", "").replace("$", "")
    s2 = str2.replace("\\$", "").replace("$", "")

    # Try direct float comparison
    try:
        f1 = float(s1)
        f2 = float(s2)
        return abs(f1 - f2) < 1e-9
    except ValueError:
        pass

    # Try to evaluate simple fractions like \frac{9}{3}
    def eval_frac(s):
        match = re.match(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
        if match:
            num, den = int(match.group(1)), int(match.group(2))
            if den != 0:
                return num / den
        # Try simple a/b format
        if "/" in s and s.count("/") == 1:
            try:
                parts = s.split("/")
                return int(parts[0]) / int(parts[1])
            except:
                pass
        return None

    f1 = eval_frac(s1)
    f2 = eval_frac(s2)

    # Compare fraction to number
    if f1 is not None:
        try:
            return abs(f1 - float(s2)) < 1e-9
        except:
            pass
    if f2 is not None:
        try:
            return abs(float(s1) - f2) < 1e-9
        except:
            pass
    if f1 is not None and f2 is not None:
        return abs(f1 - f2) < 1e-9

    return False


def is_equiv(str1, str2, verbose=False):
    """
    Check if two math strings are equivalent after normalization.
    From lm-evaluation-harness, with additional numeric equivalence checks.
    """
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)

        # Direct string match
        if ss1 == ss2:
            return True

        # Try numeric equivalence (handles trailing zeros, fractions, dollar signs)
        if try_numeric_equiv(ss1, ss2):
            return True

        return False
    except Exception:
        return str1 == str2


def extract_boxed_answer(text: str) -> str | None:
    """
    Extract the answer from \boxed{} in the text.
    Returns the content inside the last \boxed{}.
    """
    if text is None:
        return None

    boxed_str = last_boxed_only_string(text)
    if boxed_str is None:
        return None

    return remove_boxed(boxed_str)


def check_answer_in_text(pred_text: str, gold_answer: str, strict: bool = False) -> bool:
    """
    Fallback check: see if the gold answer (normalized) appears in the prediction text
    as a FINAL answer (not just anywhere in intermediate steps).

    This helps catch false negatives where the model produces the correct answer
    but doesn't wrap it in \boxed{}.

    NOTE: This fallback is conservative to avoid false positives. It only matches
    when the answer appears in a clear "final answer" context.

    Args:
        pred_text: The predicted text
        gold_answer: The gold answer extracted from \boxed{}
        strict: If True, only match very strict final answer patterns.
                If False, match slightly broader patterns but still conservative.
    """
    if pred_text is None or gold_answer is None:
        return False

    # Normalize gold answer
    try:
        gold_normalized = strip_string(gold_answer)
    except Exception:
        gold_normalized = gold_answer

    # Skip if gold answer is too short (single digit/character) - high false positive risk
    if len(gold_normalized) <= 2:
        return False

    # Skip if gold contains complex LaTeX that might appear in intermediate steps
    complex_patterns = [r"\\frac", r"\\sqrt", r"\\text", r"\\begin", r"\\end"]
    for pattern in complex_patterns:
        if pattern in gold_answer:
            # Complex LaTeX answers are too risky for fallback matching
            return False

    # Look for very specific final answer patterns only
    # These patterns indicate the model is stating its final answer
    final_answer_patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*" + re.escape(gold_answer) + r"[.\s,\)]",
        r"(?:therefore|thus|hence|so),?\s+(?:the\s+)?(?:answer\s+is\s*)?" + re.escape(gold_answer) + r"[.\s,\)]",
        r"=\s*" + re.escape(gold_answer) + r"\s*$",  # Answer at end of line after =
    ]

    for pattern in final_answer_patterns:
        if re.search(pattern, pred_text, re.IGNORECASE | re.MULTILINE):
            return True

    # For strict mode, we're done
    if strict:
        return False

    # For non-strict, also check if answer appears right before end of text
    # (within last 50 characters, suggesting it's the conclusion)
    pred_end = pred_text[-100:] if len(pred_text) > 100 else pred_text
    if gold_answer in pred_end or gold_normalized in pred_end.replace(" ", ""):
        # Additional check: make sure it's not in a conditional context like "if x < 0"
        conditional_indicators = ["if", "when", "where", "case", "given"]
        pred_end_lower = pred_end.lower()
        for indicator in conditional_indicators:
            if indicator in pred_end_lower:
                return False
        return True

    return False


def accuracy_MATH(
    pred_texts: list[str], gold_texts: list[str], allow_fallback: bool = True, strict_fallback: bool = False
) -> float:
    """
    Returns accuracy in [0, 1] for the MATH dataset.
    Uses lm-evaluation-harness style normalization.

    Args:
        pred_texts: List of predicted texts
        gold_texts: List of gold texts
        allow_fallback: If True, when no \boxed{} is found in prediction,
                       check if gold answer appears in prediction text
        strict_fallback: If True, the fallback only matches if the answer appears
                        near keywords like "answer", "therefore", etc.
    """
    assert len(pred_texts) == len(gold_texts)
    correct = 0
    for pred, gold in zip(pred_texts, gold_texts):
        # Extract answers
        g = extract_boxed_answer(gold)
        p = extract_boxed_answer(pred)

        # Check correctness using is_equiv (handles normalization)
        ok = is_equiv(p, g)

        # Fallback: if no boxed answer in prediction, check if gold appears in text
        if not ok and allow_fallback and p is None and g is not None:
            ok = check_answer_in_text(pred, g, strict=strict_fallback)

        correct += int(ok)

    return 0.0 if len(pred_texts) == 0 else correct / len(pred_texts)


def check_single_MATH(pred: str, gold: str, allow_fallback: bool = True, strict_fallback: bool = False) -> tuple:
    """
    Check a single prediction against gold answer.
    Returns (is_correct, match_type) where match_type is:
    - "boxed": matched via \boxed{} extraction
    - "fallback": matched via fallback text search
    - "none": no match
    """
    g = extract_boxed_answer(gold)
    p = extract_boxed_answer(pred)

    # Check correctness using is_equiv (handles normalization)
    if is_equiv(p, g):
        return True, "boxed"

    # Fallback: if no boxed answer in prediction, check if gold appears in text
    if allow_fallback and p is None and g is not None:
        if check_answer_in_text(pred, g, strict=strict_fallback):
            return True, "fallback"

    return False, "none"


# ==============================
# 2. Prompting & Encoding
# ==============================


def encode_for_llada_instruct(text: str, tokenizer, device: str = "cuda") -> torch.LongTensor:
    """
    Uses chat template for instruction-tuned models.
    """
    prompt = (
        f"Solve the following math problem step by step.\n Put your final answer within \\boxed{{}}.\n Problem: {text}"
    )

    msgs = [{"role": "user", "content": text}]
    s = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    ids = tokenizer(s)["input_ids"]
    return torch.tensor(ids, device=device).unsqueeze(0)


def encode_for_llada_base(text: str, tokenizer, device: str = "cuda") -> torch.LongTensor:
    """
    Uses few-shot format for base models (if needed), or simple completion.
    """
    prompt = f"Problem:\n{text}\n\nSolution:\n"
    ids = tokenizer(prompt)["input_ids"]
    return torch.tensor(ids, device=device).unsqueeze(0)


# ==============================
# 3. Generation & Evaluation Loop
# ==============================


def generate_one(model, tokenizer, input_tensor: torch.Tensor, max_new_tokens: int = 512) -> str:
    """
    Simple greedy generation.
    """
    with torch.no_grad():
        outputs = model.generate(
            input_tensor, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    generated_ids = outputs[0][input_tensor.shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def evaluate_math(model_path: str, limit: int = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    print("Loading MATH dataset...")
    dataset = load_dataset("hendrycks/competition_math", split="test")

    if limit:
        dataset = dataset.select(range(limit))

    correct_count = 0
    total = 0
    results = []

    print(f"Starting evaluation on {len(dataset)} samples...")

    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        problem = example["problem"]
        gold_solution = example["solution"]

        gold_answer = extract_boxed_answer(gold_solution)

        if gold_answer is None:
            continue

        if "Instruct" in model_path:
            input_tensor = encode_for_llada_instruct(problem, tokenizer, device)
        else:
            input_tensor = encode_for_llada_base(problem, tokenizer, device)

        try:
            pred_solution = generate_one(model, tokenizer, input_tensor)
        except Exception as e:
            print(f"Error generating for problem {i}: {e}")
            pred_solution = ""

        pred_answer = extract_boxed_answer(pred_solution)

        is_correct = is_equiv(pred_answer, gold_answer)

        if is_correct:
            correct_count += 1
        total += 1

        results.append(
            {
                "problem": problem,
                "gold_solution": gold_solution,
                "gold_extracted": gold_answer,
                "pred_solution": pred_solution,
                "pred_extracted": pred_answer,
                "is_correct": is_correct,
            }
        )

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}: Accuracy = {correct_count}/{total} ({correct_count / total:.2%})")

    accuracy = correct_count / total if total > 0 else 0
    print(f"\nFinal Accuracy: {accuracy:.2%} ({correct_count}/{total})")

    with open("llada_math_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to llada_math_results.json")
