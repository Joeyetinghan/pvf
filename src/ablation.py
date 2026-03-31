from __future__ import annotations

import argparse
import csv
from datetime import datetime
import fcntl
import json
import logging
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
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


def append_decode_summary_csv(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())

    for attempt in range(50):
        try:
            with csv_path.open("a+", newline="", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.seek(0, os.SEEK_END)
                    is_empty = handle.tell() == 0
                    writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
                    if is_empty:
                        writer.writeheader()
                    writer.writerow(row)
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return
        except Exception:
            if attempt == 49:
                raise
            time.sleep(0.05 * (attempt + 1))


def _prepare_dataset_slice(cfg: ConfigNode):
    from src.dataset.registry import get_dataset_adapter

    adapter = get_dataset_adapter(cfg.run.dataset_name)
    ds = adapter.load_split(seed=int(cfg.run.seed))

    total_size = len(ds)
    num_samples = cfg.run.get("num_samples")

    if num_samples is not None and int(num_samples) > 0:
        np.random.seed(int(cfg.run.seed))
        random_indices = np.random.choice(total_size, size=min(int(num_samples), total_size), replace=False)
        random_indices = sorted(random_indices.tolist())
        return adapter, ds.select(random_indices), 0, len(random_indices)

    start_idx = int(cfg.run.start_idx)
    end_idx_cfg = cfg.run.get("end_idx")
    end_idx = total_size if end_idx_cfg is None else min(int(end_idx_cfg), total_size)

    if start_idx >= end_idx:
        raise ValueError(f"Invalid index range: start_idx={start_idx}, end_idx={end_idx}, dataset_size={total_size}")

    return adapter, ds.select(range(start_idx, end_idx)), start_idx, end_idx


def _run_hyperparam_log(
    cfg: ConfigNode,
    logger: logging.Logger,
    output_dir: Path,
    output_file: Path,
    summary_csv: Path,
) -> None:
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
    logger.info("  hub_strategy:                  %s", cfg.generation.hub_strategy)
    logger.info("  ablation_comparison:           %s", cfg.ablation.ablation_comparison)
    logger.info("  priority_pick_strategy:        %s", cfg.ablation.priority_pick_strategy)
    logger.info("  num_samples:                   %s", cfg.run.get("num_samples"))
    logger.info("  start_idx:                     %s", cfg.run.start_idx)
    logger.info("  end_idx:                       %s", cfg.run.end_idx)
    logger.info("  output_dir:                    %s", output_dir)
    logger.info("  output_file:                   %s", output_file)
    logger.info("  summary_csv:                   %s", summary_csv)
    logger.info("  timestamp:                     %s", cfg.run.get("timestamp"))
    logger.info("=" * 60)


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


def _count_my_jobs(prefix: str) -> int:
    result = subprocess.run(
        ["bash", "-lc", f"squeue -u $USER | grep -c {shlex.quote(prefix)}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return 0
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def run_submit(cfg: ConfigNode, config_path: str, cli_overrides: list[str]) -> None:
    submit_cfg = cfg.submit
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submit_root = Path(str(submit_cfg.output_root)).resolve()
    submit_log_root = submit_root / "logs" / timestamp
    submit_log_root.mkdir(parents=True, exist_ok=True)
    submit_logger = make_logger("src.ablation.submit", submit_log_root / "submit.log")

    datasets = list(cfg.ablation.datasets)
    seeds = [int(seed) for seed in cfg.ablation.seeds]
    ablation_values = [int(value) for value in cfg.ablation.comparison_values]

    total_jobs = 0
    for dataset_name in datasets:
        dataset_sizes = submit_cfg.get("dataset_sizes")
        if not isinstance(dataset_sizes, dict) or dataset_name not in dataset_sizes:
            raise ValueError(f"Missing submit.dataset_sizes['{dataset_name}'] in config.")

        dataset_size = int(dataset_sizes[dataset_name])
        base_start = int(submit_cfg.base_start_idx)
        base_end_cfg = submit_cfg.get("base_end_idx")
        base_end = min(dataset_size, int(base_end_cfg)) if base_end_cfg is not None else dataset_size
        chunk_size = int(submit_cfg.chunk_size)
        total_size = base_end - base_start
        num_chunks = math.ceil(total_size / chunk_size)

        submit_logger.info(
            "Dataset %s: size=%s, range=[%s,%s), chunk_size=%s, chunks=%s",
            dataset_name,
            dataset_size,
            base_start,
            base_end,
            chunk_size,
            num_chunks,
        )

        for seed in seeds:
            for abla in ablation_values:
                for chunk in range(num_chunks):
                    rel_start = chunk * chunk_size
                    rel_end = min((chunk + 1) * chunk_size, total_size)
                    start_idx = base_start + rel_start
                    end_idx = base_start + rel_end

                    run_tag = f"{dataset_name}_seed{seed}_abla{abla}_chunk{chunk}".replace(".", "p")
                    log_dir = submit_log_root / run_tag
                    run_output_dir = submit_root / "results" / run_tag
                    log_dir.mkdir(parents=True, exist_ok=True)
                    run_output_dir.mkdir(parents=True, exist_ok=True)

                    job_name = f"{submit_cfg.job_name_prefix}_{run_tag}"
                    log_path = log_dir / f"{job_name}-%j.out"

                    worker_sets = list(cli_overrides)
                    worker_sets.extend(
                        [
                            f"run.dataset_name={dataset_name}",
                            f"run.seed={seed}",
                            f"run.start_idx={start_idx}",
                            f"run.end_idx={end_idx}",
                            f"run.num_samples=null",
                            f"run.output_dir={run_output_dir}",
                            f"run.timestamp={timestamp}",
                            f"ablation.ablation_comparison={abla}",
                        ]
                    )

                    worker_cmd_parts = [
                        sys.executable,
                        "-m",
                        "src.ablation",
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

                    max_jobs = int(submit_cfg.get("max_jobs_in_queue", 0))
                    wait_time = int(submit_cfg.get("wait_time_seconds", 60))
                    if max_jobs > 0:
                        while _count_my_jobs(submit_cfg.job_name_prefix) >= max_jobs:
                            submit_logger.info(
                                "Queue full for prefix %s; sleeping %ss",
                                submit_cfg.job_name_prefix,
                                wait_time,
                            )
                            time.sleep(wait_time)

                    sbatch_cmd = _build_sbatch_command(cfg, worker_command, job_name, log_path)
                    result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=False)
                    if result.returncode != 0:
                        raise RuntimeError(f"Failed to submit {job_name}: {result.stderr.strip()}")

                    total_jobs += 1
                    submit_logger.info("Submitted %s: %s", job_name, result.stdout.strip())

    submit_logger.info("Total submitted ablation jobs: %s", total_jobs)


def run_worker(cfg: ConfigNode) -> None:
    from src.dataset.registry import PreparedExample
    from src.generate.backends import generate_ablation
    from src.generate.runtime import is_instruct_model, load_model_and_tokenizer

    paths = build_paths(cfg)
    ensure_runtime_dirs(paths)
    export_hf_env(paths)

    run_output_dir_cfg = cfg.run.get("output_dir")
    if run_output_dir_cfg is None:
        output_file_cfg = cfg.run.get("output_file")
        if output_file_cfg is not None:
            run_output_dir_cfg = str(Path(str(output_file_cfg)).resolve().parent)
        else:
            run_output_dir_cfg = "z_artifacts/ablation/default"

    run_timestamp = cfg.run.get("timestamp")
    output_dir = build_timestamped_output_dir(str(run_output_dir_cfg), timestamp=run_timestamp)
    output_file = output_dir / "results.json"
    summary_csv = output_dir / "decode_summary.csv"
    logger = make_logger("src.ablation", output_dir / "run.log")

    _run_hyperparam_log(cfg, logger, output_dir, output_file, summary_csv)

    loaded = load_model_and_tokenizer(cfg.model, str(paths.hf_home))
    adapter, ds_slice, start_idx, end_idx = _prepare_dataset_slice(cfg)
    logger.info("Running on %s examples (indices %s to %s)", len(ds_slice), start_idx, end_idx)

    pred_texts: list[str] = []
    prepared_examples: list[PreparedExample] = []
    stats: list[dict[str, Any]] = []
    decode_rows: list[dict[str, Any]] = []

    all_priority_token_stats: dict[str, dict[str, Any]] = {}
    tokens_per_iter_values: list[float] = []

    instruct = is_instruct_model(str(cfg.model.model_name))

    for i, ex in enumerate(ds_slice):
        prepared = adapter.prepare(ex)
        prepared_examples.append(prepared)

        inp = adapter.encode(prepared, loaded.tokenizer, "cuda", instruct)

        out = generate_ablation(
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
            unlock_next_block_threshold=cfg.generation.unlock_next_block_threshold,
            priority_batch_inference=bool(cfg.generation.priority_batch_inference),
            priority_batch_num=int(cfg.generation.priority_batch_num),
            priority_batch_mode=str(cfg.generation.priority_batch_mode),
            priority_selection_criterion=str(cfg.generation.priority_selection_criterion),
            priority_confidence_upper_bound=float(cfg.generation.priority_confidence_upper_bound),
            high_conf_topk=cfg.generation.high_conf_topk,
            ablation_comparison=bool(cfg.ablation.ablation_comparison),
            priority_pick_strategy=str(cfg.ablation.priority_pick_strategy),
            model_type=str(cfg.model.model_type),
        )

        generated_ids, calls, strategy_tokens, avg_priority_conf, prio_stats, decode_stats = out
        generated_tokens = generated_ids[inp.shape[1] :]
        pred = loaded.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        non_eos_count = sum(1 for tok in generated_tokens if tok != loaded.eos_token_id and tok != loaded.mask_token_id)
        tokens_per_iter = non_eos_count / calls if calls > 0 else 0.0
        tokens_per_iter_values.append(tokens_per_iter)

        pred_texts.append(pred)
        stats.append(
            {
                "calls": calls,
                "strategy_tokens": strategy_tokens,
                "avg_priority_conf": avg_priority_conf,
                "tokens_per_iter": tokens_per_iter,
            }
        )

        for tok_str, tok_data in prio_stats.items():
            if tok_str not in all_priority_token_stats:
                all_priority_token_stats[tok_str] = {"count": 0, "high_conf_increases": []}
            all_priority_token_stats[tok_str]["count"] += tok_data["count"]
            all_priority_token_stats[tok_str]["high_conf_increases"].extend(tok_data["high_conf_increases"])

        acc_single = adapter.single_accuracy(pred, prepared)

        row = {
            "test_index": int(i + start_idx),
            "is_correct": int(acc_single > 0),
            "calls": int(calls),
            "strategy_tokens": int(strategy_tokens),
            "total_inner_iterations": int(decode_stats["total_inner_iterations"]),
            "planning_tokens_count": int(decode_stats["planning_tokens_count"]),
            "extra_commit_iters": int(decode_stats["extra_commit_iters"]),
            "extra_tokens_committed": int(decode_stats["extra_tokens_committed"]),
            "avg_priority_conf": float(decode_stats["avg_priority_conf"]),
            "tokens_per_iter": float(tokens_per_iter),
            "gen_length": int(cfg.generation.gen_length),
            "block_length": int(cfg.generation.block_length),
            "confidence_threshold": float(cfg.generation.confidence_threshold),
            "priority_confidence_lower_bound": cfg.generation.priority_confidence_threshold,
            "priority_confidence_upper_bound": float(cfg.generation.priority_confidence_upper_bound),
            "hub_strategy": cfg.generation.hub_strategy,
            "priority_batch_inference": int(bool(cfg.generation.priority_batch_inference)),
            "priority_batch_num": int(cfg.generation.priority_batch_num),
            "priority_batch_mode": cfg.generation.priority_batch_mode,
            "priority_selection_criterion": cfg.generation.priority_selection_criterion,
            "unlock_next_block_threshold": cfg.generation.unlock_next_block_threshold,
            "ablation_comparison": int(bool(cfg.ablation.ablation_comparison)),
            "priority_pick_strategy": cfg.ablation.priority_pick_strategy,
            "seed": int(cfg.run.seed),
            "num_samples": cfg.run.get("num_samples"),
            "model_name": cfg.model.model_name,
            "dataset_name": cfg.run.dataset_name,
        }
        decode_rows.append(row)
        append_decode_summary_csv(summary_csv, row)
        logger.info(
            "Example %s | correct=%s | calls=%s | strategy_tokens=%s | tokens_per_iter=%.3f",
            i + start_idx,
            bool(acc_single > 0),
            calls,
            strategy_tokens,
            tokens_per_iter,
        )

    acc = adapter.batch_accuracy(pred_texts, prepared_examples)
    avg_iterations = float(np.mean([s["calls"] for s in stats])) if stats else 0.0
    avg_tokens_per_iter = float(np.mean(tokens_per_iter_values)) if tokens_per_iter_values else 0.0
    all_priority_confs = [s["avg_priority_conf"] for s in stats if s["avg_priority_conf"] > 0]
    overall_avg_priority_conf = float(np.mean(all_priority_confs)) if all_priority_confs else 0.0

    priority_token_stats_summary: dict[str, dict[str, Any]] = {}
    for tok_str, tok_data in all_priority_token_stats.items():
        increases = tok_data["high_conf_increases"]
        priority_token_stats_summary[tok_str] = {
            "count": tok_data["count"],
            "avg_high_conf_increase": float(np.mean(increases)) if increases else 0.0,
        }

    decode_summary_aggregates = {
        "count_examples": len(decode_rows),
        "avg_total_inner_iterations": float(np.mean([r["total_inner_iterations"] for r in decode_rows])) if decode_rows else 0.0,
        "avg_planning_tokens_count": float(np.mean([r["planning_tokens_count"] for r in decode_rows])) if decode_rows else 0.0,
        "avg_extra_commit_iters": float(np.mean([r["extra_commit_iters"] for r in decode_rows])) if decode_rows else 0.0,
        "avg_extra_tokens_committed": float(np.mean([r["extra_tokens_committed"] for r in decode_rows])) if decode_rows else 0.0,
        "correct_examples": int(sum(r["is_correct"] for r in decode_rows)),
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
        "decode_summary_aggregates": decode_summary_aggregates,
        "output": {
            "output_dir": str(output_dir),
            "output_file": str(output_file),
            "summary_csv": str(summary_csv),
            "timestamp": output_dir.name,
        },
        "hyperparameters": {
            "model_name": cfg.model.model_name,
            "hub_strategy": cfg.generation.hub_strategy,
            "priority_words": cfg.model.get("priority_words"),
            "priority_confidence_threshold": cfg.generation.priority_confidence_threshold,
            "priority_batch_inference": cfg.generation.priority_batch_inference,
            "priority_batch_num": cfg.generation.priority_batch_num,
            "priority_batch_mode": cfg.generation.priority_batch_mode,
            "priority_selection_criterion": cfg.generation.priority_selection_criterion,
            "unlock_next_block_threshold": cfg.generation.unlock_next_block_threshold,
            "priority_confidence_upper_bound": cfg.generation.priority_confidence_upper_bound,
            "high_conf_topk": cfg.generation.high_conf_topk,
            "dataset_name": cfg.run.dataset_name,
            "ablation_comparison": cfg.ablation.ablation_comparison,
            "priority_pick_strategy": cfg.ablation.priority_pick_strategy,
            "seed": cfg.run.seed,
            "start_idx": cfg.run.start_idx,
            "end_idx": cfg.run.end_idx,
            "num_samples": cfg.run.get("num_samples"),
        },
    }

    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")

    logger.info("--- FINAL RESULTS ---")
    logger.info("Accuracy: %.4f", acc)
    logger.info("Avg Iterations: %.2f", avg_iterations)
    logger.info("Avg Tokens Per Iteration: %.2f", avg_tokens_per_iter)
    logger.info("Avg Priority Confidence: %.4f", overall_avg_priority_conf)
    logger.info("Saved results to %s", output_file)
    logger.info("Saved decode summary to %s", summary_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation entrypoint")
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
