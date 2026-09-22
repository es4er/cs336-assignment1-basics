# CS336 Assignment 1 — Language Modeling from Scratch

An implementation-and-experiment repository for Stanford CS336 Assignment 1,
covering byte-level BPE, core Transformer components, optimization, checkpointing,
and small-scale language-model training. The repository pairs executable code with
detailed Chinese study notes and an evidence-based learning-rate sweep report.

> **Status.** Core model, optimization, data, and serialization tests pass. The
> BPE implementation is functional but its assignment-reference tie-breaking and
> speed tests are still being optimized. See [Reproducibility and limitations](#reproducibility-and-limitations).

## Quick links

- [Implementation](cs336_basics/) · [Training entry point](main_train.py)
- [Complete Assignment 1 notes](docs/notes/assignment1.md)
- [Ablation plan](experiments/ablations.md)
- [Assignment handout](cs336_assignment1_basics.pdf)

## Research questions

This project uses a from-scratch implementation to study three questions:

1. How do tokenizer, normalization, attention, and optimizer design choices interact
   in a compact autoregressive language model?
2. Where is the transition between under-training, useful convergence, and
   over-aggressive optimization in a learning-rate sweep?
3. Which architectural components remain useful when compute and token budget are
   controlled through one-factor-at-a-time ablations?

## Implemented components

| Area | Components |
|---|---|
| Tokenization | UTF-8 byte representation, GPT-2-style pre-tokenization, BPE training, encode/decode |
| Model | Embedding, Linear, RMSNorm, RoPE, scaled dot-product attention, multi-head causal attention, SwiGLU, Transformer LM |
| Training | Cross-entropy, AdamW, cosine schedule with warmup, gradient clipping, memory-mapped batches, checkpoint save/resume |
| Experimentation | W&B logging, learning-rate sweeps, architectural switches, planned B200 baseline and sampling study |

## Repository structure

```text
.
├── cs336_basics/             # tokenizer, model, optimizer, and training utilities
├── tests/                    # Stanford assignment tests and adapter layer
├── docs/notes/               # chapter-indexed Chinese notes
├── experiments/
│   └── ablations.md           # controlled ablation matrix
├── main_train.py             # training and W&B entry point
└── cs336_assignment1_basics.pdf
```

## Setup

The environment is managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest
```

The upstream tokenizer test imports the Unix-only Python `resource` module. Run the
full suite on Linux/WSL; on native Windows, the remaining tests can be checked with:

```powershell
uv run pytest --ignore=tests/test_tokenizer.py
```

Datasets, checkpoints, and W&B caches are intentionally excluded from Git. Follow
the download instructions in the assignment handout, train one tokenizer, and use
that same frozen vocabulary/merge table to encode both train and validation splits.

Example training command:

```bash
uv run python main_train.py \
  --train_data_path data/TinyStoriesV2-GPT4-train.bin \
  --valid_data_path data/TinyStoriesV2-GPT4-valid.bin \
  --lr 1e-4 --min_lr 1e-5 --run_name baseline_lr_1e-4
```

## Reproducibility and limitations

- Runs with different iteration counts or evaluation sample counts are labeled and
  should not be compared as if they were controlled replicates.
- Random seeds were not frozen in the existing sweep; final claims require at least
  three seeds and mean ± standard deviation.
- The BPE adapter is connected, but 3 BPE reference tests currently fail (speed and
  exact tie-breaking); the other 20 native-Windows-compatible tests pass.
- The ablation CLI is experimental and must be validated with parameter-count and
  forward-path checks before expensive training.

## Attribution

The assignment scaffold, tests, and handout originate from Stanford CS336. The
implementation, notes, experiment records, and analysis in this repository are the
author's work. The upstream license is preserved in [LICENSE](LICENSE). This is an
educational reproduction, not an official Stanford repository.
