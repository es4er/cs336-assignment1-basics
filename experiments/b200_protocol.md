# B200 training and sampling protocol

## Gate before renting the GPU

- Encode train and validation data with one frozen tokenizer artifact.
- Record SHA-256 checksums for data, vocabulary, and merges.
- Make training deterministic enough to reproduce a smoke run.
- Complete a 100-step GPU smoke test, checkpoint save/resume test, and sampling test.
- Estimate tokens/second, memory headroom, and total cost from the smoke run.

## Baseline training

Start from the validated `1e-4` learning-rate region. Log tokens processed rather
than only optimizer steps, gradient norm before clipping, throughput, GPU memory,
validation loss, and checkpoint metadata. Keep a last checkpoint and a best-by-
validation checkpoint. Use early stopping only with a policy declared in advance.

## Sampling evaluation

For each selected checkpoint, store the exact prompt set and random seeds. Compare
greedy decoding and temperature sampling at several temperatures, with fixed
`top_k`/`top_p` settings. Report both qualitative samples and aggregate measures
such as repetition rate, distinct n-grams, average completion length, and held-out
perplexity. Never select only favorable samples.

## Minimum artifacts

- frozen configuration and Git commit;
- tokenizer/data checksums;
- W&B run URL and exported CSV;
- loss/learning-rate/gradient-norm curves;
- checkpoint manifest;
- fixed prompts, seeds, decoding parameters, and generated samples;
- short failure analysis and limitations.

