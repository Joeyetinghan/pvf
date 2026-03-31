import contextlib
import io
import re
import signal
from typing import Any

import torch


# ---------- Code Extraction ----------


def extract_python_code(text: str, entry_point: str) -> str:
    """
    Extracts the Python code block from the model's generation.
    It looks for the function definition matching the entry_point
    and tries to clean up markdown or extra text.
    """
    # 1. If code is inside markdown ```python ... ``` blocks, extract it
    if "```python" in text:
        pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1]  # Return the last block (often the final solution)

    # 2. If no markdown, try to clean up by finding the start of the function
    # This is a heuristic: find 'def entry_point' and take everything after
    if f"def {entry_point}" in text:
        start_idx = text.find(f"def {entry_point}")
        return text[start_idx:]

    # 3. Fallback: return the whole text (model might have output raw code)
    return text


# ---------- Execution & Testing ----------


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)


def check_correctness(problem: dict[str, Any], completion: str, timeout: float = 3.0) -> bool:
    """
    Evaluates the functional correctness of a generated candidate.
    This combines the candidate code with the test case provided in the dataset.

    WARNING: This executes untrusted code. Run only in a sandboxed environment.
    """

    # The HumanEval dataset provides a 'check' function in the 'test' column.
    # We need to concatenate:
    # 1. The prompt (imports + function signature)
    # 2. The model's completion (function body)
    # 3. The test code (assertions)
    # 4. The call to the check function

    test_setup = problem["test"]
    entry_point = problem["entry_point"]
    prompt = problem["prompt"]

    # Construct the full script to execute
    # We assume the model completes the function body started in 'prompt'
    # However, some models repeat the prompt. We handle this by checking overlap.

    if completion.strip().startswith("def " + entry_point):
        # If model rewrote the definition, just use completion
        full_code = completion + "\n" + test_setup + f"\ncheck({entry_point})"
    else:
        # If model just continued, append to prompt
        full_code = prompt + completion + "\n" + test_setup + f"\ncheck({entry_point})"

    # Capture stdout/stderr to avoid cluttering logs
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            # Using a simple exec with timeout
            # NOTE: Ideally, use multiprocessing or a docker container for safety
            with time_limit(timeout):
                exec_globals = {}
                exec(full_code, exec_globals)
            return True
        except TimeoutException:
            return False
        except Exception:
            return False


# ---------- Prompt & Encoding ----------


def encode_for_humaneval_instruct(prompt: str, tokenizer, device: str = "cuda") -> torch.LongTensor:
    """
    Wraps the HumanEval prompt (function signature) in a user message.
    """
    msgs = [{"role": "user", "content": f"Complete the following Python code:\n{prompt}"}]
    s = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(s)["input_ids"]
    return torch.tensor(ids, device=device).unsqueeze(0)


def encode_for_humaneval_base(prompt: str, tokenizer, device: str = "cuda") -> torch.LongTensor:
    """
    For base models, we just feed the raw function signature.
    """
    ids = tokenizer(prompt)["input_ids"]
    return torch.tensor(ids, device=device).unsqueeze(0)


# ---------- Accuracy Calculation ----------


def accuracy_HumanEval(pred_texts: list[str], problems: list[dict[str, Any]]) -> float:
    """
    Computes Pass@1 accuracy by executing the code.
    """
    assert len(pred_texts) == len(problems)
    correct = 0

    print(f"Evaluating {len(pred_texts)} samples (this may take a moment)...")

    for pred, problem in zip(pred_texts, problems):
        # 1. Extract cleaner code block
        clean_code = extract_python_code(pred, problem["entry_point"])

        # 2. Run the test
        # Note: In a real rigorous setting, you'd use 'human_eval.execution'
        is_passing = check_correctness(problem, clean_code)
        correct += int(is_passing)

    return 0.0 if len(pred_texts) == 0 else correct / len(pred_texts)
