"""Fine-tune a single-sentence BERT/RoBERTa classifier.

The shared transformer_utils.py performs Novel Validation early stopping,
threshold selection, and Full/Novel metric reporting.
"""
from __future__ import annotations

import argparse

try:
    from scripts.transformer_utils import train_transformer
except ImportError:  # Support: python scripts/train_bert.py
    from transformer_utils import train_transformer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the single-sentence Transformer baseline.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    train_transformer(experiment="bert", use_context=False, config_path=args.config)


if __name__ == "__main__":
    main()
