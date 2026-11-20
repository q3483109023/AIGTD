"""Transformer sentence embeddings used by the optional AIGTD fusion model."""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def extract_transformer_embeddings(
    texts: Iterable[str],
    model_name: str = "roberta-base",
    max_length: int = 256,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> np.ndarray:
    """Compute attention-mask mean-pooled embeddings in input order.

    The first call downloads model files from Hugging Face unless they are cached locally.
    """
    texts = [str(x) for x in texts]
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            chunks.append(pooled.detach().cpu().numpy().astype(np.float32))
    del model
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(chunks, axis=0)
