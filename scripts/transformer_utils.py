"""Shared Transformer fine-tuning for single-sentence and context AIGTD models.

Early stopping and threshold selection use Novel Validation. Full Validation
is still evaluated and reported at the same selected threshold.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from scripts.common import (
        build_context_text,
        cfg_path,
        load_config,
        mark_novel_validation,
        set_seed,
    )
    from scripts.evaluate import evaluate_validation
except ImportError:  # Support: python scripts/train_bert.py
    from common import build_context_text, cfg_path, load_config, mark_novel_validation, set_seed
    from evaluate import evaluate_validation


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None):
        if labels is not None and len(texts) != len(labels):
            raise ValueError("texts 与 labels 长度不一致。")
        self.texts = [str(value) for value in texts]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        item = {"text": self.texts[index]}
        if self.labels is not None:
            item["label"] = int(self.labels[index])
        return item


def _make_collate(tokenizer, max_length: int):
    def collate(batch: List[dict]) -> dict:
        texts = [item["text"] for item in batch]
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        if "label" in batch[0]:
            encoded["labels"] = torch.tensor(
                [item["label"] for item in batch], dtype=torch.long
            )
        return encoded
    return collate


def _predict_probabilities(model, loader, device: str) -> Tuple[np.ndarray, float]:
    model.eval()
    all_probs: List[np.ndarray] = []
    total_loss = 0.0
    total_count = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)
            probabilities = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.append(probabilities.detach().cpu().numpy())
            total_loss += float(loss.item()) * labels.size(0)
            total_count += labels.size(0)
    if not all_probs:
        raise RuntimeError("验证 DataLoader 没有产生任何 batch。")
    return np.concatenate(all_probs), total_loss / max(total_count, 1)


def train_transformer(
    experiment: str,
    use_context: bool = False,
    config_path: str = "configs/experiment.yaml",
) -> None:
    cfg = load_config(config_path)
    seed = int(cfg.get("project", {}).get("random_seed", 42))
    set_seed(seed)

    transformer_cfg = cfg.get("transformer", {})
    model_name = str(transformer_cfg.get("model_name", "roberta-base"))
    max_length = int(transformer_cfg.get("max_length", 256))
    train_batch_size = int(transformer_cfg.get("train_batch_size", 16))
    eval_batch_size = int(transformer_cfg.get("eval_batch_size", 32))
    epochs = int(transformer_cfg.get("epochs", 5))
    learning_rate = float(transformer_cfg.get("learning_rate", 2e-5))
    weight_decay = float(transformer_cfg.get("weight_decay", 0.01))
    patience = int(transformer_cfg.get("patience", 2))
    context_window = int(transformer_cfg.get("context_window", 1))
    device_cfg = str(transformer_cfg.get("device", "auto"))
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device_cfg == "auto" else device_cfg
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "配置要求 CUDA，但当前环境没有可用 CUDA。请将 transformer.device 设置为 auto 或 cpu。"
        )
    if train_batch_size < 1 or eval_batch_size < 1 or epochs < 1:
        raise ValueError("batch_size 和 epochs 必须是正整数。")

    splits_dir = cfg_path(cfg, "splits_dir", "data/splits")
    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "validation.csv"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "缺少 data/splits/train.csv 或 validation.csv，请先运行 python scripts/preprocess.py。"
        )
    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    val_df = pd.read_csv(val_path, encoding="utf-8-sig")

    train_ids = set(train_df["article_id"].astype(str))
    val_ids = set(val_df["article_id"].astype(str))
    overlap = train_ids & val_ids
    if overlap:
        raise RuntimeError(
            f"训练集和验证集存在 {len(overlap)} 个 article_id 重叠，停止训练。示例：{sorted(overlap)[:10]}"
        )

    # Novel flag is sentence-level. It does not remove rows or alter article
    # context, and it uses precisely the same normalization as duplicate audit.
    val_df = mark_novel_validation(val_df, train_df)
    novel_mask = val_df["is_novel_validation"].astype(bool).to_numpy()
    y_train = train_df["label_id"].astype(int).to_numpy()
    y_val = val_df["label_id"].astype(int).to_numpy()
    if not novel_mask.any():
        raise ValueError("Novel Validation 为空，无法进行 early stopping 和阈值选择。")
    if len(np.unique(y_val[novel_mask])) < 2:
        raise ValueError("Novel Validation 只包含一个类别，无法用 Macro-F1 进行模型选择。")
    if len(np.unique(y_train)) < 2:
        raise ValueError("训练数据缺少 human 或 machine 类别。")

    if use_context:
        # Build contexts on the complete split, then evaluate only novel target rows.
        train_texts = build_context_text(train_df, context_window)
        val_texts = build_context_text(val_df, context_window)
    else:
        train_texts = train_df["text"].astype(str).tolist()
        val_texts = val_df["text"].astype(str).tolist()

    print(f"模型：{model_name}；device：{device}")
    print(f"训练目标句子：{len(train_df):,}")
    print(f"Full Validation 目标句子：{len(val_df):,}")
    print(
        f"Novel Validation 目标句子：{int(novel_mask.sum()):,} "
        f"({novel_mask.mean():.2%})"
    )
    print(f"训练模式：{'上下文窗口=' + str(context_window) if use_context else '仅目标句子'}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    num_workers = int(transformer_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        TextDataset(train_texts, y_train.tolist()),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_make_collate(tokenizer, max_length),
    )
    val_loader = DataLoader(
        TextDataset(val_texts, y_val.tolist()),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_make_collate(tokenizer, max_length),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    class_weighted = bool(transformer_cfg.get("class_weighted_loss", True))
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    if (counts == 0).any():
        raise ValueError(f"训练集缺少某个类别，类别计数为 {counts.tolist()}。")
    class_weights = (
        counts.sum() / (2.0 * counts)
        if class_weighted else np.ones(2, dtype=np.float32)
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    history: List[dict] = []
    best_novel_macro = -1.0
    best_state = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen_count = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item()) * labels.size(0)
            seen_count += labels.size(0)

        val_probs, val_loss = _predict_probabilities(model, val_loader, device)
        val_pred_05 = (val_probs >= 0.5).astype(int)
        novel_macro = float(f1_score(
            y_val[novel_mask], val_pred_05[novel_mask],
            labels=[0, 1], average="macro", zero_division=0,
        ))
        full_macro = float(f1_score(
            y_val, val_pred_05, labels=[0, 1], average="macro", zero_division=0
        ))
        train_loss = running_loss / max(seen_count, 1)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "novel_validation_macro_f1_at_0.5": novel_macro,
            "full_validation_macro_f1_at_0.5": full_macro,
            "novel_validation_count": int(novel_mask.sum()),
            "full_validation_count": int(len(val_df)),
        })
        print(
            f"[{experiment}] epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"novel_macro_f1@0.5={novel_macro:.4f} | "
            f"full_macro_f1@0.5={full_macro:.4f}"
        )

        # Checkpoint selection is based on Novel Validation, not the full set.
        if novel_macro > best_novel_macro + 1e-6:
            best_novel_macro = novel_macro
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(
                    f"[{experiment}] Early stopping at epoch {epoch}; "
                    f"best epoch={best_epoch}, novel_macro_f1@0.5={best_novel_macro:.4f}."
                )
                break

    if best_state is None:
        raise RuntimeError("训练没有产生有效模型状态。")
    model.load_state_dict(best_state)

    best_model_dir = cfg_path(cfg, "best_model_dir", "models/best_model") / experiment
    checkpoint_dir = cfg_path(cfg, "checkpoints_dir", "models/checkpoints") / experiment
    best_model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    pd.DataFrame(history).to_csv(
        checkpoint_dir / "training_history.csv", index=False, encoding="utf-8-sig"
    )

    # Threshold optimization and final validation report use the selected checkpoint.
    val_probs, _ = _predict_probabilities(model, val_loader, device)
    threshold_cfg = cfg.get("threshold", {})
    threshold, metrics, _ = evaluate_validation(
        frame=val_df,
        probabilities=val_probs,
        experiment=experiment,
        results_dir=cfg_path(cfg, "results_dir", "results"),
        threshold_min=float(threshold_cfg.get("min", 0.30)),
        threshold_max=float(threshold_cfg.get("max", 0.80)),
        threshold_step=float(threshold_cfg.get("step", 0.01)),
        optimize_threshold=True,
        train_df=train_df,
    )

    metadata = {
        "model_type": "context" if use_context else "bert",
        "model_name": model_name,
        "max_length": max_length,
        "context_window": context_window if use_context else 0,
        "threshold": threshold,
        "threshold_selection_metric": "novel_macro_f1",
        "checkpoint_selection_metric": "novel_validation_macro_f1_at_0.5",
        "best_epoch": best_epoch,
        "best_novel_validation_macro_f1_at_0.5": best_novel_macro,
        "class_weighted_loss": class_weighted,
        "seed": seed,
        "novel_validation_count": int(novel_mask.sum()),
        "full_validation_count": int(len(val_df)),
        "novel_text_normalization": "Unicode NFKC + casefold + whitespace normalization; punctuation retained",
        "label_to_id": {"human": 0, "machine": 1},
    }
    (best_model_dir / "aigtd_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (checkpoint_dir / "validation_metrics.json").write_text(
        json.dumps({"threshold": threshold, **metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[{experiment}] 最优模型：{best_model_dir}")
    print(f"[{experiment}] Novel Validation 最优阈值：{threshold:.4f}")
    print("以下未加前缀的五项指标代表 Novel Validation：")
    print(pd.Series({
        key: metrics[key]
        for key in ("accuracy", "precision", "recall", "macro_f1", "micro_f1")
    }).round(4).to_string())
    print("Full Validation Macro-F1（同一阈值）：{:.4f}".format(metrics["full_macro_f1"]))


def predict_transformer_probabilities(
    texts: List[str],
    model_dir: str | Path,
    batch_size: int = 32,
    device: str = "auto",
) -> np.ndarray:
    model_dir = Path(model_dir)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    metadata_path = model_dir / "aigtd_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"模型元数据不存在：{metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    max_length = int(metadata.get("max_length", 256))

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    all_probs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = [str(value) for value in texts[start:start + batch_size]]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            all_probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())

    del model
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(all_probs) if all_probs else np.empty((0,), dtype=np.float32)
