from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


TIME_FORMAT = "%Y%m%d_%H%M%S"


class ConfigNode(dict):
    """Dictionary with attribute access and recursive node conversion."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @staticmethod
    def from_obj(obj: Any) -> Any:
        if isinstance(obj, dict):
            node = ConfigNode()
            for key, value in obj.items():
                node[key] = ConfigNode.from_obj(value)
            return node
        if isinstance(obj, list):
            return [ConfigNode.from_obj(v) for v in obj]
        return obj


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    artifacts_root: Path
    cache_root: Path
    hf_home: Path
    hf_datasets_cache: Path
    hf_hub_cache: Path
    tmp_dir: Path


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def load_task_config(config_path: str | Path, overrides: list[str] | None = None) -> ConfigNode:
    """Load layered config by merging base -> task base -> selected config -> CLI overrides."""
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    task_dir = path.parent
    task_name = task_dir.name
    root_config_dir = task_dir.parent

    base_cfg_path = root_config_dir / "base.yml"
    task_base_cfg_path = task_dir / "base.yml"

    merged_omega = OmegaConf.create({})
    if base_cfg_path.exists():
        merged_omega = OmegaConf.merge(merged_omega, OmegaConf.load(base_cfg_path))
    if task_base_cfg_path.exists() and task_base_cfg_path != path:
        merged_omega = OmegaConf.merge(merged_omega, OmegaConf.load(task_base_cfg_path))
    merged_omega = OmegaConf.merge(merged_omega, OmegaConf.load(path))
    if overrides:
        merged_omega = OmegaConf.merge(merged_omega, OmegaConf.from_dotlist(overrides))

    merged = ConfigNode.from_obj(OmegaConf.to_container(merged_omega, resolve=True))
    if "runtime" not in merged:
        merged["runtime"] = ConfigNode()
    merged.runtime.task = task_name
    merged.runtime.config_path = str(path)
    return merged


def build_paths(cfg: ConfigNode) -> Paths:
    base_dir = Path(cfg.paths.base_dir).resolve()
    artifacts_root = _resolve_path(base_dir, cfg.paths.artifacts_root)
    cache_root = _resolve_path(base_dir, cfg.paths.cache_root)
    hf_home = _resolve_path(base_dir, cfg.paths.hf_home)
    hf_datasets_cache = _resolve_path(base_dir, cfg.paths.hf_datasets_cache)
    hf_hub_cache = _resolve_path(base_dir, cfg.paths.hf_hub_cache)
    tmp_dir = _resolve_path(base_dir, cfg.paths.tmp_dir)

    return Paths(
        repo_root=base_dir,
        artifacts_root=artifacts_root,
        cache_root=cache_root,
        hf_home=hf_home,
        hf_datasets_cache=hf_datasets_cache,
        hf_hub_cache=hf_hub_cache,
        tmp_dir=tmp_dir,
    )


def ensure_runtime_dirs(paths: Paths) -> None:
    for directory in (
        paths.artifacts_root,
        paths.cache_root,
        paths.hf_home,
        paths.hf_datasets_cache,
        paths.hf_hub_cache,
        paths.tmp_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def export_hf_env(paths: Paths) -> None:
    os.environ["HF_HOME"] = str(paths.hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(paths.hf_datasets_cache)
    os.environ["HF_HUB_CACHE"] = str(paths.hf_hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(paths.hf_hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(paths.hf_hub_cache)
    os.environ["TMPDIR"] = str(paths.tmp_dir)
    os.environ["TMP"] = str(paths.tmp_dir)


def get_current_time() -> str:
    return datetime.now().strftime(TIME_FORMAT)


def build_timestamped_output_dir(base_output_dir: str | Path, timestamp: str | None = None) -> Path:
    base_dir = Path(base_output_dir).resolve()
    run_timestamp = str(timestamp) if timestamp is not None else get_current_time()
    run_dir = base_dir / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_logger(name: str, log_file: str | Path) -> logging.Logger:
    path = Path(log_file).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
