"""Fine-tune a Transformer using previous/target/next sentence context.

The whole article sequence remains intact. Novel Validation filters only the
target sentence rows included in metrics and threshold selection.
"""
from __future__ import annotations

import argparse

try:
    from scripts.transformer_utils import train_transformer
except ImportError:  # Support: python scripts/train_context.py
    from transformer_utils import train_transformer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the context-enhanced Transformer model.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    train_transformer(experiment="context", use_context=True, config_path=args.config)


if __name__ == "__main__":
    main()
