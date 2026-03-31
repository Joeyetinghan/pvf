from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import numpy as np

from src.utils import (
    ConfigNode,
    build_paths,
    build_timestamped_output_dir,
    ensure_runtime_dirs,
    export_hf_env,
    load_task_config,
    make_logger,
)


def _run_hyperparam_log(cfg: ConfigNode, logger: logging.Logger, output_file: Path) -> None:
    logger.info("=" * 60)
    logger.info("HYPERPARAMETERS")
    logger.info("=" * 60)
    logger.info("  model_name:                    %s", cfg.model.model_name)
    logger.info("  dataset_name:                  %s", cfg.run.dataset_name)
    logger.info("  gen_length:                    %s", cfg.generation.gen_length)
    logger.info("  block_length:                  %s", cfg.generation.block_length)
    logger.info("  temperature:                   %s", cfg.generation.temperature)
    logger.info("  cfg_scale:                     %s", cfg.generation.cfg_scale)
    logger.info("  confidence_threshold:          %s", cfg.generation.confidence_threshold)
    logger.info("  confidence_filter_threshold:   %s", cfg.generation.confidence_filter_threshold)
    logger.info("  hub_strategy:                  %s", cfg.generation.hub_strategy)
    logger.info("  priority_words:                %s", cfg.model.get("priority_words") or "DEFAULT")
    logger.info("  priority_confidence_threshold: %s", cfg.generation.priority_confidence_threshold)
    logger.info("  cumulative_fallback:           %s", cfg.generation.cumulative_fallback)
    logger.info("  cumulative_fallback_num:       %s", cfg.generation.cumulative_fallback_num)
    logger.info("  cumulative_fallback_order:     %s", cfg.generation.cumulative_fallback_order)
    logger.info("  priority_batch_inference:      %s", cfg.generation.priority_batch_inference)
    logger.info("  priority_batch_num:            %s", cfg.generation.priority_batch_num)
    logger.info("  priority_batch_mode:           %s", cfg.generation.priority_batch_mode)
    logger.info("  priority_selection_criterion:  %s", cfg.generation.priority_selection_criterion)
    logger.info("  unlock_next_block_threshold:   %s", cfg.generation.unlock_next_block_threshold)
    logger.info("  priority_confidence_upper_bound: %s", cfg.generation.priority_confidence_upper_bound)
    logger.info("  high_conf_topk:                %s", cfg.generation.high_conf_topk)
    logger.info("  show_token_labels:             %s", cfg.generation.show_token_labels)
    logger.info("  skip_priority_verification:    %s", cfg.generation.skip_priority_verification)
    logger.info("  seed:                          %s", cfg.run.seed)
    logger.info("  start_idx:                     %s", cfg.run.start_idx)
    logger.info("  end_idx:                       %s", cfg.run.end_idx)
    logger.info("  output_dir:                    %s", cfg.run.output_dir)
    logger.info("  timestamp:                     %s", cfg.run.get("timestamp"))
    logger.info("  output_file:                   %s", output_file)
    logger.info("=" * 60)


def _prepare_dataset_slice(cfg: ConfigNode):
    from src.dataset.registry import get_dataset_adapter

    adapter = get_dataset_adapter(cfg.run.dataset_name)
    ds = adapter.load_split(seed=int(cfg.run.seed))

    total_size = len(ds)
    start_idx = int(cfg.run.start_idx)
    end_idx_cfg = cfg.run.get("end_idx")
    end_idx = total_size if end_idx_cfg is None else min(int(end_idx_cfg), total_size)

    if start_idx >= end_idx:
        raise ValueError(f"Invalid index range: start_idx={start_idx}, end_idx={end_idx}, dataset_size={total_size}")

    return adapter, ds.select(range(start_idx, end_idx)), start_idx, end_idx


def _decode_single(
    cfg: ConfigNode,
    prepared: Any,
    loaded,
    is_instruct: bool,
    adapter,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    from src.generate.backends import generate

    inp = adapter.encode(prepared, loaded.tokenizer, "cuda", is_instruct)

    out = generate(
        loaded.model,
        loaded.tokenizer,
        inp,
        gen_length=int(cfg.generation.gen_length),
        block_length=int(cfg.generation.block_length),
        temp=float(cfg.generation.temperature),
        cfg=float(cfg.generation.cfg_scale),
        mask_id=int(loaded.mask_token_id),
        conf_thresh=float(cfg.generation.confidence_threshold),
        filter_thresh=float(cfg.generation.confidence_filter_threshold),
        hub_strategy=str(cfg.generation.hub_strategy),
        priority_token_ids=loaded.priority_token_ids,
        seed=int(cfg.run.seed),
        priority_confidence_threshold=cfg.generation.priority_confidence_threshold,
        cumulative_fallback=bool(cfg.generation.cumulative_fallback),
        cumulative_fallback_num=int(cfg.generation.cumulative_fallback_num),
        unlock_next_block_threshold=cfg.generation.unlock_next_block_threshold,
        priority_batch_inference=bool(cfg.generation.priority_batch_inference),
        priority_batch_num=int(cfg.generation.priority_batch_num),
        priority_batch_mode=str(cfg.generation.priority_batch_mode),
        priority_selection_criterion=str(cfg.generation.priority_selection_criterion),
        priority_confidence_upper_bound=float(cfg.generation.priority_confidence_upper_bound),
        cumulative_fallback_order=str(cfg.generation.cumulative_fallback_order),
        high_conf_topk=cfg.generation.high_conf_topk,
        track_token_labels=bool(cfg.generation.show_token_labels),
        skip_priority_verification=bool(cfg.generation.skip_priority_verification),
        model_type=str(cfg.model.model_type),
    )

    (
        generated_ids,
        calls,
        strategy_tokens,
        fallback_tokens,
        avg_priority_conf,
        priority_token_stats,
        consistency_passed,
        consistency_total,
        cumulative_fb_counts,
        token_labels,
        priority_batch_fallback_count,
    ) = out

    generated_tokens = generated_ids[inp.shape[1] :]
    eos_token_id = loaded.eos_token_id
    mask_id = loaded.mask_token_id
    non_eos_count = sum(1 for tok in generated_tokens if tok != eos_token_id and tok != mask_id)
    tokens_per_iter = non_eos_count / calls if calls > 0 else 0.0

    pred = loaded.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    sample_stats = {
        "calls": calls,
        "strategy_tokens": strategy_tokens,
        "fallback_tokens": fallback_tokens,
        "avg_priority_conf": avg_priority_conf,
        "tokens_per_iter": tokens_per_iter,
        "priority_consistency_passed": consistency_passed,
        "priority_consistency_total": consistency_total,
        "priority_batch_fallback_count": priority_batch_fallback_count,
    }

    extra = {
        "priority_token_stats": priority_token_stats,
        "cumulative_fb_counts": cumulative_fb_counts,
        "token_labels": token_labels,
    }

    return pred, sample_stats, extra


def run_worker(cfg: ConfigNode) -> None:
    from src.dataset.registry import PreparedExample
    from src.generate.runtime import is_instruct_model, load_model_and_tokenizer

    paths = build_paths(cfg)
    ensure_runtime_dirs(paths)
    export_hf_env(paths)

    run_timestamp = cfg.run.get("timestamp")
    output_dir = build_timestamped_output_dir(str(cfg.run.output_dir), timestamp=run_timestamp)
    output_file = output_dir / "results.json"
    logger = make_logger("src.eval", output_dir / "run.log")
    _run_hyperparam_log(cfg, logger, output_file)

    loaded = load_model_and_tokenizer(cfg.model, str(paths.hf_home))
    adapter, ds_slice, start_idx, end_idx = _prepare_dataset_slice(cfg)

    logger.info("Running on %s examples (indices %s to %s)", len(ds_slice), start_idx, end_idx)

    pred_texts: list[str] = []
    prepared_examples: list[PreparedExample] = []
    stats: list[dict[str, Any]] = []

    all_priority_token_stats: dict[str, dict[str, Any]] = {}
    all_cumulative_fb_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    instruct = is_instruct_model(str(cfg.model.model_name))

    for i, ex in enumerate(ds_slice):
        prepared = adapter.prepare(ex)
        prepared_examples.append(prepared)

        pred, sample_stats, extra = _decode_single(cfg, prepared, loaded, instruct, adapter)

        pred_texts.append(pred)
        stats.append(sample_stats)

        for tok_str, tok_data in extra["priority_token_stats"].items():
            if tok_str not in all_priority_token_stats:
                all_priority_token_stats[tok_str] = {"count": 0, "high_conf_increases": []}
            all_priority_token_stats[tok_str]["count"] += tok_data["count"]
            all_priority_token_stats[tok_str]["high_conf_increases"].extend(tok_data["high_conf_increases"])

        for k, v in extra["cumulative_fb_counts"].items():
            all_cumulative_fb_counts[k] = all_cumulative_fb_counts.get(k, 0) + v

        acc_single = adapter.single_accuracy(pred, prepared)
        logger.info("=== Example %s ===", i + start_idx)
        logger.info("Pred: %s | Gold: %s", pred, prepared.gold)
        logger.info("Correct: %s", acc_single > 0)

    acc = adapter.batch_accuracy(pred_texts, prepared_examples)
    avg_iterations = float(np.mean([s["calls"] for s in stats])) if stats else 0.0
    avg_tokens_per_iter = float(np.mean([s["tokens_per_iter"] for s in stats])) if stats else 0.0
    all_priority_confs = [s["avg_priority_conf"] for s in stats if s["avg_priority_conf"] > 0]
    overall_avg_priority_conf = float(np.mean(all_priority_confs)) if all_priority_confs else 0.0

    priority_token_stats_summary: dict[str, dict[str, Any]] = {}
    for tok_str, tok_data in all_priority_token_stats.items():
        increases = tok_data["high_conf_increases"]
        priority_token_stats_summary[tok_str] = {
            "count": tok_data["count"],
            "avg_conf_increase_per_token": float(np.mean(increases)) if increases else 0.0,
        }

    results = {
        "pred_texts": pred_texts,
        "gold_texts": [p.gold for p in prepared_examples],
        "stats": stats,
        "accuracy": acc,
        "avg_iterations": avg_iterations,
        "avg_tokens_per_iter": avg_tokens_per_iter,
        "avg_priority_confidence": overall_avg_priority_conf,
        "priority_token_stats": priority_token_stats_summary,
        "hyperparameters": {
            "model_name": cfg.model.model_name,
            "hub_strategy": cfg.generation.hub_strategy,
            "priority_words": cfg.model.get("priority_words"),
            "priority_confidence_threshold": cfg.generation.priority_confidence_threshold,
            "cumulative_fallback": cfg.generation.cumulative_fallback,
            "cumulative_fallback_num": cfg.generation.cumulative_fallback_num,
            "cumulative_fallback_order": cfg.generation.cumulative_fallback_order,
            "priority_batch_inference": cfg.generation.priority_batch_inference,
            "priority_batch_num": cfg.generation.priority_batch_num,
            "priority_batch_mode": cfg.generation.priority_batch_mode,
            "priority_selection_criterion": cfg.generation.priority_selection_criterion,
            "unlock_next_block_threshold": cfg.generation.unlock_next_block_threshold,
            "priority_confidence_upper_bound": cfg.generation.priority_confidence_upper_bound,
            "high_conf_topk": cfg.generation.high_conf_topk,
            "skip_priority_verification": cfg.generation.skip_priority_verification,
            "dataset_name": cfg.run.dataset_name,
        },
    }

    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(results, handle)

    logger.info("--- FINAL RESULTS ---")
    logger.info("Accuracy: %.4f", acc)
    logger.info("Avg Iterations: %.2f", avg_iterations)
    logger.info("Avg Tokens Per Iteration: %.2f", avg_tokens_per_iter)
    logger.info("Avg Priority Confidence: %.4f", overall_avg_priority_conf)
    logger.info("Saved results to %s", output_file)


def _build_sbatch_command(cfg: ConfigNode, worker_command: str, job_name: str, log_path: Path) -> list[str]:
    slurm = cfg.slurm
    command = [
        "sbatch",
        f"--job-name={job_name}",
        f"--account={slurm.account}",
        f"--ntasks={slurm.ntasks}",
        f"--cpus-per-task={slurm.cpus_per_task}",
        f"--partition={slurm.partition}",
        f"--gres={slurm.gres}",
        f"--mem={slurm.mem}",
        f"--time={slurm.time}",
        f"--output={log_path}",
    ]

    if slurm.get("constraint"):
        command.append(f"--constraint={slurm.constraint}")
    if slurm.get("exclude"):
        command.append(f"--exclude={slurm.exclude}")
    if slurm.get("queue"):
        command.extend(["-q", str(slurm.queue)])

    command.extend(["--wrap", worker_command])
    return command


def run_submit(cfg: ConfigNode, config_path: str, cli_overrides: list[str]) -> None:
    submit_cfg = cfg.submit
    dataset_name = str(cfg.run.dataset_name)

    dataset_sizes = submit_cfg.get("dataset_sizes")
    if not isinstance(dataset_sizes, dict) or dataset_name not in dataset_sizes:
        raise ValueError(f"Missing submit.dataset_sizes['{dataset_name}'] in config.")
    dataset_size = int(dataset_sizes[dataset_name])
    base_start_idx = int(submit_cfg.base_start_idx)
    base_end_cfg = submit_cfg.get("base_end_idx")
    base_end_idx = min(dataset_size, int(base_end_cfg)) if base_end_cfg is not None else dataset_size

    total_size = base_end_idx - base_start_idx
    if total_size <= 0:
        raise ValueError("Invalid submit range: base_end_idx must be greater than base_start_idx.")

    num_chunks = int(submit_cfg.num_chunks)
    chunk_size = math.ceil(total_size / num_chunks)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_bucket = str(cfg.model.model_type)
    gen_length = int(cfg.generation.gen_length)
    block_length = int(cfg.generation.block_length)
    experiment_tag = str(submit_cfg.get("experiment_tag", "baseline"))
    base_output_dir = (
        Path(str(submit_cfg.output_root)).resolve()
        / model_bucket
        / dataset_name
        / f"GL-{gen_length}"
        / f"BL-{block_length}"
        / experiment_tag
        / timestamp
    )
    log_dir = base_output_dir / "logs"
    results_dir = base_output_dir / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    submit_logger = make_logger("src.eval.submit", base_output_dir / "submit.log")

    submit_logger.info("Dataset: %s (size: %s)", dataset_name, dataset_size)
    submit_logger.info("Submitting %s jobs with chunk size %s", num_chunks, chunk_size)
    submit_logger.info("Output root: %s", base_output_dir)

    for chunk in range(num_chunks):
        relative_start = chunk * chunk_size
        relative_end = min((chunk + 1) * chunk_size, total_size)
        start_idx = base_start_idx + relative_start
        end_idx = base_start_idx + relative_end

        job_name = f"{submit_cfg.job_name_prefix}_{dataset_name}_chunk{chunk}_GL-{gen_length}_BL-{block_length}_{experiment_tag}"
        log_path = log_dir / f"{job_name}-%j.out"
        worker_sets = list(cli_overrides)
        worker_sets.extend(
            [
                f"run.start_idx={start_idx}",
                f"run.end_idx={end_idx}",
                f"run.output_dir={results_dir / job_name}",
                f"run.timestamp={timestamp}",
            ]
        )

        worker_cmd_parts = [
            sys.executable,
            "-m",
            "src.eval",
            "--config",
            str(Path(config_path).resolve()),
            "--mode",
            "run",
        ]
        for override in worker_sets:
            worker_cmd_parts.extend(["--set", override])
        worker_command = shlex.join(worker_cmd_parts)

        setup_commands = list(cfg.slurm.get("setup_commands") or [])
        if setup_commands:
            worker_command = " && ".join([*setup_commands, worker_command])

        sbatch_cmd = _build_sbatch_command(cfg, worker_command, job_name, log_path)
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to submit {job_name}: {result.stderr.strip()}")

        submit_logger.info("Submitted %s: %s", job_name, result.stdout.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation entrypoint")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--mode", default="run", choices=["run", "submit"])
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="OmegaConf dotlist override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_task_config(args.config, overrides=args.overrides)

    if args.mode == "submit":
        run_submit(cfg, args.config, list(args.overrides))
    else:
        run_worker(cfg)


if __name__ == "__main__":
    main()
