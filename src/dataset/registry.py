from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from datasets import concatenate_datasets, load_dataset

from src.dataset.arc_c_eval import (
    accuracy_arc_c,
    encode_for_llada_base as encode_arc_base,
    encode_for_llada_instruct as encode_arc_instruct,
)
from src.dataset.gsm8k_eval import (
    accuracy_gsm8k,
    encode_for_llada_base as encode_gsm_base,
    encode_for_llada_instruct as encode_gsm_instruct,
)
from src.dataset.humaneval_eval import (
    accuracy_HumanEval,
    encode_for_humaneval_base,
    encode_for_humaneval_instruct,
)
from src.dataset.MATH_eval import (
    accuracy_MATH,
    check_single_MATH,
    encode_for_llada_base as encode_math_base,
    encode_for_llada_instruct as encode_math_instruct,
)
from src.dataset.mmlu_eval import (
    accuracy_MMLU,
    encode_for_llada_base as encode_mmlu_base,
    encode_for_llada_instruct as encode_mmlu_instruct,
)
from src.dataset.mmlu_pro_eval import (
    accuracy_MMLU_Pro,
    encode_for_llada_base as encode_mmlu_pro_base,
    encode_for_llada_instruct as encode_mmlu_pro_instruct,
)
from src.dataset.winogrande_eval import (
    accuracy_winogrande,
    encode_for_llada_base as encode_winogrande_base,
    encode_for_llada_instruct as encode_winogrande_instruct,
)


@dataclass
class PreparedExample:
    prompt: str
    gold: str
    extra: dict[str, Any]


@dataclass
class DatasetAdapter:
    name: str

    def load_split(self, seed: int):
        raise NotImplementedError

    def prepare(self, example: dict[str, Any]) -> PreparedExample:
        raise NotImplementedError

    def encode(self, prepared: PreparedExample, tokenizer, device: str, is_instruct: bool):
        raise NotImplementedError

    def single_accuracy(self, pred: str, prepared: PreparedExample) -> float:
        raise NotImplementedError

    def batch_accuracy(self, preds: list[str], prepared_examples: list[PreparedExample]) -> float:
        raise NotImplementedError


class GSM8KAdapter(DatasetAdapter):
    def __init__(self):
        super().__init__("gsm8k")

    def load_split(self, seed: int):
        return load_dataset("gsm8k", "main", split="test")

    def prepare(self, example: dict[str, Any]) -> PreparedExample:
        return PreparedExample(prompt=example["question"], gold=example["answer"], extra={})

    def encode(self, prepared: PreparedExample, tokenizer, device: str, is_instruct: bool):
        if is_instruct:
            return encode_gsm_instruct(prepared.prompt, tokenizer, device)
        return encode_gsm_base(prepared.prompt, tokenizer, device)

    def single_accuracy(self, pred: str, prepared: PreparedExample) -> float:
        return accuracy_gsm8k([pred], [prepared.gold])

    def batch_accuracy(self, preds: list[str], prepared_examples: list[PreparedExample]) -> float:
        return accuracy_gsm8k(preds, [p.gold for p in prepared_examples])


class MathAdapter(DatasetAdapter):
    def __init__(self):
        super().__init__("math")

    def load_split(self, seed: int):
        configs = [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ]
        test_sets = [load_dataset("EleutherAI/hendrycks_math", cfg, split="test") for cfg in configs]
        return concatenate_datasets(test_sets).shuffle(seed=seed)

    def prepare(self, example: dict[str, Any]) -> PreparedExample:
        return PreparedExample(prompt=example["problem"], gold=example["solution"], extra={})

    def encode(self, prepared: PreparedExample, tokenizer, device: str, is_instruct: bool):
        if is_instruct:
            return encode_math_instruct(prepared.prompt, tokenizer, device)
        return encode_math_base(prepared.prompt, tokenizer, device)

    def single_accuracy(self, pred: str, prepared: PreparedExample) -> float:
        is_correct, _ = check_single_MATH(pred, prepared.gold, allow_fallback=True, strict_fallback=False)
        return 1.0 if is_correct else 0.0

    def batch_accuracy(self, preds: list[str], prepared_examples: list[PreparedExample]) -> float:
        golds = [p.gold for p in prepared_examples]
        return accuracy_MATH(preds, golds, allow_fallback=True, strict_fallback=False)


class MMLUAdapter(DatasetAdapter):
    def __init__(self):
        super().__init__("mmlu")

    def load_split(self, seed: int):
        return load_dataset("cais/mmlu", "all", split="test").shuffle(seed=seed)

    def prepare(self, example: dict[str, Any]) -> PreparedExample:
        gold = ["A", "B", "C", "D"][example["answer"]]
        return PreparedExample(prompt=example["question"], gold=gold, extra={"options": example["choices"]})

    def encode(self, prepared: PreparedExample, tokenizer, device: str, is_instruct: bool):
        if is_instruct:
            return encode_mmlu_instruct(prepared.prompt, prepared.extra["options"], tokenizer, device)
        return encode_mmlu_base(prepared.prompt, prepared.extra["options"], tokenizer, device)

    def single_accuracy(self, pred: str, prepared: PreparedExample) -> float:
        return accuracy_MMLU([pred], [prepared.gold])

    def batch_accuracy(self, preds: list[str], prepared_examples: list[PreparedExample]) -> float:
        return accuracy_MMLU(preds, [p.gold for p in prepared_examples])


class MMLUProAdapter(DatasetAdapter):
    def __init__(self):
        super().__init__("mmlu_pro")

    def load_split(self, seed: int):
        return load_dataset("TIGER-Lab/MMLU-Pro", split="test").shuffle(seed=seed)

    def prepare(self, example: dict[str, Any]) -> PreparedExample:
        return PreparedExample(prompt=example["question"], gold=example["answer"], extra={"options": example["options"]})

    def encode(self, prepared: PreparedExample, tokenizer, device: str, is_instruct: bool):
        if is_instruct:
            return encode_mmlu_pro_instruct(prepared.prompt, prepared.extra["options"], tokenizer, device)
        return encode_mmlu_pro_base(prepared.prompt, prepared.extra["options"], tokenizer, device)

    def single_accuracy(self, pred: str, prepared: PreparedExample) -> float:
        return accuracy_MMLU_Pro([pred], [prepared.gold])

    def batch_accuracy(self, preds: list[str], prepared_examples: list[PreparedExample]) -> float:
        return accuracy_MMLU_Pro(preds, [p.gold for p in prepared_examples])


class HumanEvalAdapter(DatasetAdapter):
    def __init__(self):
        super().__init__("humaneval")

    def load_split(self, seed: int):
        return load_dataset("openai_humaneval", split="test").shuffle(seed=seed)

    def prepare(self, example: dict[str, Any]) -> PreparedExample:
        return PreparedExample(prompt=example["prompt"], gold=example["canonical_solution"], extra={"problem": example})

    def encode(self, prepared: PreparedExample, tokenizer, device: str, is_instruct: bool):
        if is_instruct:
            return encode_for_humaneval_instruct(prepared.prompt, tokenizer, device)
        return encode_for_humaneval_base(prepared.prompt, tokenizer, device)

    def single_accuracy(self, pred: str, prepared: PreparedExample) -> float:
        return accuracy_HumanEval([pred], [prepared.extra["problem"]])

    def batch_accuracy(self, preds: list[str], prepared_examples: list[PreparedExample]) -> float:
        return accuracy_HumanEval(preds, [p.extra["problem"] for p in prepared_examples])


class ArcCAdapter(DatasetAdapter):
    def __init__(self):
        super().__init__("arc_c")

    def load_split(self, seed: int):
        return load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").shuffle(seed=seed)

    def prepare(self, example: dict[str, Any]) -> PreparedExample:
        return PreparedExample(prompt=example["question"], gold=example["answerKey"], extra={"choices": example["choices"]})

    def encode(self, prepared: PreparedExample, tokenizer, device: str, is_instruct: bool):
        if is_instruct:
            return encode_arc_instruct(prepared.prompt, prepared.extra["choices"], tokenizer, device)
        return encode_arc_base(prepared.prompt, prepared.extra["choices"], tokenizer, device)

    def single_accuracy(self, pred: str, prepared: PreparedExample) -> float:
        return accuracy_arc_c([pred], [prepared.gold])

    def batch_accuracy(self, preds: list[str], prepared_examples: list[PreparedExample]) -> float:
        return accuracy_arc_c(preds, [p.gold for p in prepared_examples])


class WinograndeAdapter(DatasetAdapter):
    def __init__(self):
        super().__init__("winogrande")

    def load_split(self, seed: int):
        return load_dataset("allenai/winogrande", "winogrande_xl", split="validation").shuffle(seed=seed)

    def prepare(self, example: dict[str, Any]) -> PreparedExample:
        opt1 = example["option1"]
        opt2 = example["option2"]
        # Keep compatibility with current parser in winogrande accuracy helper.
        gold_blob = json.dumps({"answer": int(example["answer"]), "options": [opt1, opt2]})
        return PreparedExample(prompt=example["sentence"], gold=gold_blob, extra={"option1": opt1, "option2": opt2})

    def encode(self, prepared: PreparedExample, tokenizer, device: str, is_instruct: bool):
        if is_instruct:
            return encode_winogrande_instruct(
                prepared.prompt,
                prepared.extra["option1"],
                prepared.extra["option2"],
                tokenizer,
                device,
            )
        return encode_winogrande_base(
            prepared.prompt,
            prepared.extra["option1"],
            prepared.extra["option2"],
            tokenizer,
            device,
        )

    def single_accuracy(self, pred: str, prepared: PreparedExample) -> float:
        return accuracy_winogrande([pred], [prepared.gold])

    def batch_accuracy(self, preds: list[str], prepared_examples: list[PreparedExample]) -> float:
        return accuracy_winogrande(preds, [p.gold for p in prepared_examples])


_DATASET_REGISTRY: dict[str, DatasetAdapter] = {
    "gsm8k": GSM8KAdapter(),
    "math": MathAdapter(),
    "mmlu": MMLUAdapter(),
    "mmlu_pro": MMLUProAdapter(),
    "humaneval": HumanEvalAdapter(),
    "arc_c": ArcCAdapter(),
    "winogrande": WinograndeAdapter(),
}


def get_dataset_adapter(name: str) -> DatasetAdapter:
    if name not in _DATASET_REGISTRY:
        available = ", ".join(sorted(_DATASET_REGISTRY))
        raise ValueError(f"Unsupported dataset '{name}'. Available: {available}")
    return _DATASET_REGISTRY[name]
