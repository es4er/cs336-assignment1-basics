# Ablation plan

The training entry point exposes four architectural switches. Treat them as an
experimental interface until forward-path checks confirm that each switch changes
the intended module and only that module. Then run one change at a time against a
frozen baseline.

| ID | Change | CLI setting | Hypothesis |
|---|---|---|---|
| A0 | Baseline | `--norm_mode pre --ffn_type swiglu` | reference |
| A1 | Remove RMSNorm | `--no_rms_norm` | optimization becomes less stable |
| A2 | Post-norm | `--norm_mode post` | slower or less stable deep optimization |
| A3 | Remove RoPE | `--no_rope` | weaker position-sensitive modeling |
| A4 | Replace SwiGLU | `--ffn_type silu` | lower parameter efficiency or quality |

Use the same tokenizer, data order, token budget, optimizer configuration,
evaluation prompts, and seeds. Report parameter counts and compute because an
ablation is not fair if it silently changes model size. Use at least three seeds
for the final table and report mean ± standard deviation rather than selecting the
best seed.
