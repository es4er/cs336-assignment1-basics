"""Encode a validation corpus with the tokenizer trained on the training split."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from cs336_basics.preprocess import load_trained_tokenizer, process_corpus


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-text", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    args = parser.parse_args()

    for path in (args.validation_text, args.vocab, args.merges):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    tokenizer = load_trained_tokenizer(
        str(args.vocab), str(args.merges), args.special_token
    )
    process_corpus(str(args.validation_text), str(args.output), tokenizer)

    ids = np.memmap(args.output, dtype=np.uint16, mode="r")
    if not len(ids):
        raise RuntimeError("Encoded validation set is empty")
    maximum = int(ids.max())
    if maximum >= len(tokenizer.vocab):
        raise RuntimeError(
            f"Token id {maximum} is outside vocabulary size {len(tokenizer.vocab)}"
        )

    source_prefix = args.validation_text.read_text(encoding="utf-8")[:4096]
    round_trip = tokenizer.decode(tokenizer.encode(source_prefix))
    if round_trip != source_prefix:
        raise RuntimeError("Tokenizer failed the validation-text prefix round-trip check")

    print(f"tokens={len(ids):,}")
    print(f"token_id_range={int(ids.min())}..{maximum}")
    print(f"vocab_size={len(tokenizer.vocab):,}")
    print(f"output_sha256={sha256(args.output)}")
    print(f"vocab_sha256={sha256(args.vocab)}")
    print(f"merges_sha256={sha256(args.merges)}")


if __name__ == "__main__":
    main()
