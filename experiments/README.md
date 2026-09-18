# Experiments

This directory separates reproducible protocols from machine-local artifacts.
Raw datasets, checkpoints, and `wandb/` caches are intentionally excluded from Git.

## Experiment map

| Study | Status | Entry point | Report |
|---|---:|---|---|
| Learning-rate range finding | Completed locally | `main_train.py` | [`lr_sweep/README.md`](lr_sweep/README.md) |
| B200 baseline training | Planned | `main_train.py` | [`b200_protocol.md`](b200_protocol.md) |
| Sampling study | Planned | to be added after baseline | [`b200_protocol.md`](b200_protocol.md#sampling-evaluation) |
| Architectural ablations | Planned | CLI flags in `main_train.py` | [`ablations.md`](ablations.md) |

## Reproducibility policy

Every reported run should record the Git commit, random seed, hardware, CUDA/PyTorch
versions, tokenizer checksum, dataset checksum, full configuration, wall-clock time,
and W&B run URL. Comparisons are valid only when the tokenizer, train/validation
split, token budget, evaluation cadence, and seed policy are held fixed.

