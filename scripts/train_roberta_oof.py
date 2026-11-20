#!/usr/bin/env python3
"""Train fixed-epoch purged source-ID OOF RoBERTa models.

The outer OOF validation fold is never used for early stopping or checkpoint
selection. By default each fold trains for 3 fixed epochs, matching the best
single-split epoch observed in the existing baseline. This is a deliberate,
pre-fixed schedule; change only as a separately documented experiment.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SentenceDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int] | None = None):
        self.texts = [str(x) for x in texts]
        self.labels = labels
        if labels is not None and len(texts) != len(labels):
            raise ValueError("texts/labels length mismatch")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {"text": self.texts[idx]}
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
        return item


def make_collate(tokenizer, max_length: int):
    def collate(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            [x["text"] for x in items],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        if "label" in items[0]:
            encoded["labels"] = torch.tensor([x["label"] for x in items], dtype=torch.long)
        return encoded
    return collate


def infer(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    margins: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**batch).logits
            margin = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            margins.append(margin)
            probs.append(p)
    if not margins:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    return np.concatenate(margins), np.concatenate(probs)


def add_prediction_columns(frame: pd.DataFrame, fold: int, margin: np.ndarray, prob: np.ndarray, split: str) -> pd.DataFrame:
    if len(frame) != len(margin) or len(frame) != len(prob):
        raise ValueError("Prediction length mismatch")
    result = frame.copy()
    result["fold"] = fold
    result["split"] = split
    result["roberta_logit_margin"] = margin.astype(float)
    result["roberta_p_machine"] = prob.astype(float)
    if "label_id" in result.columns:
        result["y_true"] = result["label_id"].astype(int)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--folds-dir", default="data/oof_folds")
    parser.add_argument("--output-dir", default="results/oof")
    parser.add_argument("--model-dir", default="models/oof/roberta")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto, cuda, or cpu")
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parents[1] / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    folds_dir = Path(args.folds_dir)
    output_dir = Path(args.output_dir)
    model_root = Path(args.model_dir)
    manifest_path = folds_dir / "manifest.json"
    novel_path = folds_dir / "novel_validation.csv"
    if not manifest_path.exists() or not novel_path.exists():
        raise FileNotFoundError("Prepared OOF folds not found; run prepare_oof_folds.py first.")
    fold_meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(fold_meta["n_splits"]) != args.n_splits:
        raise ValueError(f"Prepared folds={fold_meta['n_splits']} but requested={args.n_splits}")

    tr_cfg = cfg.get("transformer", {})
    model_name = str(tr_cfg.get("model_name", "roberta-base"))
    max_length = int(tr_cfg.get("max_length", 256))
    train_batch = args.train_batch_size or int(tr_cfg.get("train_batch_size", 16))
    eval_batch = args.eval_batch_size or int(tr_cfg.get("eval_batch_size", 32))
    learning_rate = float(tr_cfg.get("learning_rate", 2e-5))
    weight_decay = float(tr_cfg.get("weight_decay", 0.01))
    num_workers = int(tr_cfg.get("num_workers", 0))
    class_weighted = bool(tr_cfg.get("class_weighted_loss", True))
    base_seed = int(cfg.get("project", {}).get("random_seed", 42))
    requested_device = args.device or str(tr_cfg.get("device", "auto"))
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    novel_df = pd.read_csv(novel_path, encoding="utf-8-sig")
    if novel_df.empty:
        raise ValueError("Novel Validation is empty")

    print(f"Model={model_name}; device={device}; epochs/fold={args.epochs}; max_length={max_length}")
    print(f"Novel validation rows={len(novel_df):,}; will be scored by each of {args.n_splits} fold models")

    all_oof: list[pd.DataFrame] = []
    all_novel: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []

    for fold in range(args.n_splits):
        set_seed(base_seed + fold)
        fold_dir = folds_dir / f"fold_{fold}"
        train_path = fold_dir / "train.csv"
        val_path = fold_dir / "validation.csv"
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(f"Missing fold files for fold {fold}")
        train_df = pd.read_csv(train_path, encoding="utf-8-sig")
        val_df = pd.read_csv(val_path, encoding="utf-8-sig")
        y_train = train_df["label_id"].astype(int).to_numpy()
        if set(np.unique(y_train)) != {0, 1}:
            raise ValueError(f"Fold {fold}: training fold must contain both labels")

        print(f"\n[RoBERTa OOF] Fold {fold + 1}/{args.n_splits}: train={len(train_df):,}, val={len(val_df):,}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

        train_loader = DataLoader(
            SentenceDataset(train_df["text"].astype(str).tolist(), y_train.tolist()),
            batch_size=train_batch, shuffle=True, num_workers=num_workers,
            collate_fn=make_collate(tokenizer, max_length), pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            SentenceDataset(val_df["text"].astype(str).tolist()),
            batch_size=eval_batch, shuffle=False, num_workers=num_workers,
            collate_fn=make_collate(tokenizer, max_length), pin_memory=(device.type == "cuda"),
        )
        novel_loader = DataLoader(
            SentenceDataset(novel_df["text"].astype(str).tolist()),
            batch_size=eval_batch, shuffle=False, num_workers=num_workers,
            collate_fn=make_collate(tokenizer, max_length), pin_memory=(device.type == "cuda"),
        )

        counts = np.bincount(y_train, minlength=2).astype(np.float32)
        class_weights = counts.sum() / (2.0 * counts) if class_weighted else np.ones(2, dtype=np.float32)
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        history: list[dict[str, float]] = []

        # Crucial: the outer OOF validation fold is NOT used for early stopping.
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            seen = 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                optimizer.zero_grad(set_to_none=True)
                logits = model(**batch).logits
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_loss += float(loss.item()) * labels.size(0)
                seen += labels.size(0)
            epoch_loss = running_loss / max(seen, 1)
            history.append({"epoch": float(epoch), "train_loss": float(epoch_loss)})
            print(f"[RoBERTa OOF] fold={fold} epoch={epoch}/{args.epochs} train_loss={epoch_loss:.4f}")

        fold_model_dir = model_root / f"roberta_fold_{fold}"
        fold_model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(fold_model_dir)
        tokenizer.save_pretrained(fold_model_dir)
        pd.DataFrame(history).to_csv(fold_model_dir / "training_history.csv", index=False, encoding="utf-8-sig")
        metadata = {
            "model_name": model_name,
            "fold": fold,
            "epochs": args.epochs,
            "max_length": max_length,
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "class_weighted_loss": class_weighted,
            "seed": base_seed + fold,
            "outer_oof_validation_used_for_early_stopping": False,
            "purge_strategy": fold_meta["fold_strategy"],
        }
        (fold_model_dir / "oof_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        val_margin, val_prob = infer(model, val_loader, device)
        novel_margin, novel_prob = infer(model, novel_loader, device)
        all_oof.append(add_prediction_columns(val_df, fold, val_margin, val_prob, "oof_validation"))
        all_novel.append(add_prediction_columns(novel_df, fold, novel_margin, novel_prob, "novel_validation"))
        fold_summaries.append({
            "fold": fold,
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "novel_rows_scored": len(novel_df),
            "epochs": args.epochs,
            "model_dir": str(fold_model_dir),
            "last_train_loss": history[-1]["train_loss"],
        })

        del model, tokenizer, optimizer, train_loader, val_loader, novel_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof = pd.concat(all_oof, ignore_index=True)
    novel_all = pd.concat(all_novel, ignore_index=True)
    expected_rows = int(fold_meta["train_rows"])
    if len(oof) != expected_rows or oof["oof_row_id"].nunique() != expected_rows:
        raise RuntimeError(
            f"OOF coverage mismatch: rows={len(oof)}, unique={oof['oof_row_id'].nunique()}, expected={expected_rows}"
        )
    if oof["oof_row_id"].duplicated().any():
        raise RuntimeError("Duplicate oof_row_id in RoBERTa OOF predictions")
    if len(novel_all) != len(novel_df) * args.n_splits:
        raise RuntimeError("Novel fold prediction count mismatch")

    oof_path = output_dir / "roberta_oof_predictions.csv"
    novel_out_path = output_dir / "roberta_novel_fold_predictions.csv"
    oof.to_csv(oof_path, index=False, encoding="utf-8-sig")
    novel_all.to_csv(novel_out_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_summaries).to_csv(output_dir / "roberta_oof_fold_summary.csv", index=False, encoding="utf-8-sig")
    print("\n[RoBERTa OOF] Completed.")
    print(f"OOF predictions: {len(oof):,} -> {oof_path}")
    print(f"Novel fold predictions: {len(novel_all):,} -> {novel_out_path}")
    print("Saved both raw roberta_logit_margin and softmax roberta_p_machine.")


if __name__ == "__main__":
    main()
