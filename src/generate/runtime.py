from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from src.generate.backends import DEFAULT_PRIORITY_WORDS


@dataclass(frozen=True)
class LoadedModel:
    model: object
    tokenizer: object
    priority_token_ids: set[int]
    mask_token_id: int
    eos_token_id: int | None


def _resolve_local_model_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str).expanduser().resolve()
    return path if path.exists() else None


def load_model_and_tokenizer(model_cfg, hf_home: str) -> LoadedModel:
    model_name = model_cfg.model_name
    local_path = _resolve_local_model_path(model_cfg.get("local_model_path"))

    model_source = str(local_path) if local_path is not None else model_name

    model = (
        AutoModel.from_pretrained(
            model_source,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            cache_dir=hf_home,
        )
        .to("cuda")
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
        cache_dir=hf_home,
    )

    priority_words = list(model_cfg.get("priority_words") or DEFAULT_PRIORITY_WORDS)
    priority_ids: list[int] = []
    for word in priority_words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            priority_ids.extend(ids)
        spaced_ids = tokenizer.encode(f" {word}", add_special_tokens=False)
        if spaced_ids:
            priority_ids.extend(spaced_ids)

    mask_token_id = tokenizer.mask_token_id
    cfg_mask_id = model_cfg.get("mask_token_id")
    if mask_token_id is None and cfg_mask_id is not None:
        mask_token_id = int(cfg_mask_id)

    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        priority_token_ids=set(priority_ids),
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
    )


def is_instruct_model(model_name: str) -> bool:
    return "Instruct" in model_name
