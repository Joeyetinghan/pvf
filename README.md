## Plan, Verify and Fill: A Structured Parallel Decoding Approach for Diffusion Language Models

This repo contains the code for the paper [**Plan, Verify and Fill: A Structured Parallel Decoding Approach for Diffusion Language Models**](https://arxiv.org/abs/2601.12247).  
A local copy of the paper is available at [`assets/pvf`](assets/pvf_Jan-29-2026.pdf).

## Project Status

This repo is a **work in progress** and will continue to be updated.  
It is not polished yet, so you may run into rough edges or missing pieces.  
If you hit any issues or have questions, please open a GitHub issue — we’re happy to follow up and improve things over time.


## Repository Tree

The tree below focuses on the codebase and excludes paths listed in `.gitignore` (for example: `z_artifacts/`, `ref/`, `.vscode/`).

```text
.
├── LICENSE
├── README.md
├── pyproject.toml
├── assets/
│   ├── commit_behavior.gif
│   └── pvf_Jan-29-2026.pdf
└── src/
    ├── __init__.py
    ├── eval.py
    ├── ablation.py
    ├── vis.py
    ├── utils.py
    ├── config/
    │   ├── base.yml
    │   ├── eval.yml
    │   └── ablation.yml
    ├── generate/
    │   ├── __init__.py
    │   ├── runtime.py
    │   └── backends.py
    └── dataset/
        ├── __init__.py
        ├── registry.py
        ├── gsm8k_eval.py
        ├── MATH_eval.py
        ├── mmlu_eval.py
        ├── mmlu_pro_eval.py
        ├── humaneval_eval.py
        ├── arc_c_eval.py
        └── winogrande_eval.py
```

## What Each Part Does (Brief)

- `src/config/`: experiment configuration (global defaults + eval/ablation-specific settings).
- `src/eval.py`: main evaluation entrypoint (`run` and SLURM `submit` modes).
- `src/ablation.py`: ablation entrypoint (`run` and SLURM `submit` modes), plus per-example decode summary CSV.
- `src/generate/`: model loading and decoding logic.
- `src/dataset/`: dataset adapters, prompt formatting, and accuracy computation.
- `src/vis.py`: visualization utility for commit behavior GIFs.
- `src/utils.py`: config loading/merging, runtime paths, logging, timestamped output directories.
- `assets/`: paper PDF and example visualization artifact.

## Typical Workflow

1. Edit [base.yml](src/config/base.yml) first to define shared runtime settings (paths, SLURM defaults).
2. Adjust task settings in:
   - [eval.yml](src/config/eval.yml) for evaluation runs.
   - [ablation.yml](src/config/ablation.yml) for ablation runs.
3. Run eval or ablation from the repo root.

## Run Evaluation

Run directly:

```bash
python -m src.eval --config src/config/eval.yml --mode run
```

Optional overrides:

```bash
python -m src.eval --config src/config/eval.yml --mode run \
  --set run.dataset_name=mmlu \
  --set run.start_idx=0 \
  --set run.end_idx=200
```

Output location:

- Default base output directory is resolved from config:
  `z_artifacts/<model_type>/<dataset_name>/GL-<gen_length>/BL-<block_length>/<experiment_tag>/`
- Each run creates a timestamp folder:
  `.../<timestamp>/`
- Main files:
  - `results.json`
  - `run.log`

SLURM submission mode:

```bash
python -m src.eval --config src/config/eval.yml --mode submit
```

Submitted jobs write outputs under a `results/` subtree and logs under a `logs/` subtree of the generated timestamped experiment folder.

## Run Ablation

Run directly:

```bash
python -m src.ablation --config src/config/ablation.yml --mode run
```

Optional overrides:

```bash
python -m src.ablation --config src/config/ablation.yml --mode run \
  --set run.dataset_name=gsm8k \
  --set run.num_samples=100 \
  --set ablation.ablation_comparison=1
```

Output location:

- Default base output directory:
  `z_artifacts/ablation/default/`
- Each run creates:
  `z_artifacts/ablation/default/<timestamp>/`
- Main files:
  - `results.json`
  - `decode_summary.csv`
  - `run.log`

SLURM submission mode:

```bash
python -m src.ablation --config src/config/ablation.yml --mode submit
```

Submitted jobs write results under:
`z_artifacts/ablation/results/<run_tag>/<timestamp>/`  
and logs under:
`z_artifacts/ablation/logs/<timestamp>/`.

## Qualitative Decoding Trajectories

These animations illustrate how token commitments accumulate over decoding iterations under two different decoding settings. They are intended as qualitative views of generation dynamics and commit patterns.

<table>
  <tr>
    <td align="center" width="50%">
      <img src="assets/confidence_based.gif" alt="Default decoding trajectory" width="100%">
    </td>
    <td align="center" width="50%">
      <img src="assets/pvf_style.gif" alt="Alternative decoding trajectory" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <p align="center"><em>Fast-dLLM</em></p>
    </td>
    <td align="center" width="50%">
      <p align="center"><em>PVF</em></p>
    </td>
  </tr>
</table>
