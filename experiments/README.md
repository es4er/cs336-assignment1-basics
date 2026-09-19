# Experiments

This directory separates reproducible protocols from machine-local artifacts.
Raw datasets, checkpoints, and `wandb/` caches are intentionally excluded from Git.

## Experiment map

| Study | Status | Entry point | Report |
|---|---:|---|---|
| Learning-rate sweep | Ready to rerun on repaired data | `main_train.py` | [W&B dashboard](https://wandb.ai/meiyuxin7-china-university-of-petroleum/cs336-assignment1/table) |
| Architectural ablations | Planned | CLI flags in `main_train.py` | [`ablations.md`](ablations.md) |

## Reproducibility policy

Every reported run should record the Git commit, random seed, hardware, CUDA/PyTorch
versions, tokenizer checksum, dataset checksum, full configuration, wall-clock time,
and W&B run URL. Comparisons are valid only when the tokenizer, train/validation
split, token budget, evaluation cadence, and seed policy are held fixed.
