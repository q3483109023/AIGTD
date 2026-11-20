#!/usr/bin/env python3
"""Train fixed-epoch source-grouped, window-purged context RoBERTa OOF models."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class ContextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int] | None = None):
        if labels is not None and len(texts) != len(labels):
            raise ValueError("texts/labels length mismatch")
        self.texts, self.labels = [str(x) for x in texts], labels
    def __len__(self) -> int: return len(self.texts)
    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {"text": self.texts[index]}
        if self.labels is not None: item["label"] = int(self.labels[index])
        return item


def make_collate(tokenizer, max_length: int):
    def collate(items):
        encoded = tokenizer([x["text"] for x in items], truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        if "label" in items[0]:
            encoded["labels"] = torch.tensor([x["label"] for x in items], dtype=torch.long)
        return encoded
    return collate


def infer(model, loader, device):
    model.eval(); margins, probs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**batch).logits
            margins.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
            probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    if not margins: return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    return np.concatenate(margins), np.concatenate(probs)


def attach(frame, fold, margin, prob, split):
    if len(frame) != len(margin) or len(frame) != len(prob): raise ValueError("Prediction length mismatch")
    out = frame.copy(); out["fold"] = fold; out["split"] = split
    out["context_logit_margin"] = margin.astype(float); out["context_p_machine"] = prob.astype(float)
    if "label_id" in out: out["y_true"] = out["label_id"].astype(int)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--folds-dir", default="data/context_oof_folds")
    p.add_argument("--output-dir", default="results/context_oof")
    p.add_argument("--model-dir", default="models/oof/context")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train-batch-size", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--device", default=None, help="auto, cuda, or cpu")
    p.add_argument("--input-column", choices=["text", "context_text"], default="context_text", help="text gives a matched sentence-only control on identical purged rows.")
    p.add_argument("--run-name", default="context", help="Output/checkpoint prefix.")
    a = p.parse_args()
    if a.epochs < 1: raise ValueError("--epochs must be >= 1")
    config_path = Path(a.config)
    if not config_path.is_absolute(): config_path = Path(__file__).resolve().parents[1] / config_path
    if not config_path.exists(): raise FileNotFoundError(config_path)
    import yaml
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    folds_dir, output_dir, model_root = map(Path, (a.folds_dir, a.output_dir, a.model_dir))
    manifest_path, novel_path = folds_dir / "manifest.json", folds_dir / "novel_validation.csv"
    if not manifest_path.exists() or not novel_path.exists(): raise FileNotFoundError("Run prepare_context_oof_folds.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest["n_splits"]) != a.n_splits: raise ValueError("Prepared/requested fold count mismatch.")
    checks = manifest.get("checks", {})
    for key in ("training_window_components_disjoint_from_outer_validation_windows", "training_window_components_disjoint_from_novel_validation_windows"):
        if checks.get(key) is not True: raise RuntimeError(f"Manifest does not certify {key}.")
    tr = cfg.get("transformer", {})
    model_name = str(tr.get("model_name", "roberta-base")); max_length = int(tr.get("max_length", 256))
    train_bs = a.train_batch_size or int(tr.get("train_batch_size", 16))
    eval_bs = a.eval_batch_size or int(tr.get("eval_batch_size", 32))
    lr, wd = float(tr.get("learning_rate", 2e-5)), float(tr.get("weight_decay", 0.01))
    workers, weighted = int(tr.get("num_workers", 0)), bool(tr.get("class_weighted_loss", True))
    seed = int(cfg.get("project", {}).get("random_seed", 42))
    requested = a.device or str(tr.get("device", "auto"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable.")
    if train_bs < 1 or eval_bs < 1: raise ValueError("Batch sizes must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True); model_root.mkdir(parents=True, exist_ok=True)
    novel = pd.read_csv(novel_path, encoding="utf-8-sig")
    if novel.empty or a.input_column not in novel: raise ValueError(f"Novel validation empty or missing {a.input_column}.")
    all_oof, all_novel, summaries = [], [], []
    print(f"Model={model_name}; device={device}; epochs/fold={a.epochs}; max_length={max_length}")
    print(f"Novel targets={len(novel):,}; window={manifest['context_window']}")
    for fold in range(a.n_splits):
        set_seed(seed + fold)
        fold_dir = folds_dir / f"fold_{fold}"
        train_path, val_path = fold_dir / "train.csv", fold_dir / "validation.csv"
        if not train_path.exists() or not val_path.exists(): raise FileNotFoundError(f"Missing fold {fold} files.")
        train_df, val_df = pd.read_csv(train_path, encoding="utf-8-sig"), pd.read_csv(val_path, encoding="utf-8-sig")
        for name, frame in (("train", train_df), ("validation", val_df)):
            if a.input_column not in frame or "label_id" not in frame: raise ValueError(f"Fold {fold} {name} missing {a.input_column}/label_id.")
        y = train_df["label_id"].astype(int).to_numpy()
        if set(np.unique(y)) != {0, 1}: raise ValueError(f"Fold {fold} training requires both classes.")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
        collate = make_collate(tokenizer, max_length)
        train_loader = DataLoader(ContextDataset(train_df[a.input_column].tolist(), y.tolist()), batch_size=train_bs, shuffle=True, num_workers=workers, collate_fn=collate, pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(ContextDataset(val_df[a.input_column].tolist()), batch_size=eval_bs, shuffle=False, num_workers=workers, collate_fn=collate, pin_memory=(device.type == "cuda"))
        novel_loader = DataLoader(ContextDataset(novel[a.input_column].tolist()), batch_size=eval_bs, shuffle=False, num_workers=workers, collate_fn=collate, pin_memory=(device.type == "cuda"))
        counts = np.bincount(y, minlength=2).astype(np.float32)
        if (counts == 0).any(): raise ValueError(f"Fold {fold} empty class after purge: {counts.tolist()}")
        weights = counts.sum() / (2.0 * counts) if weighted else np.ones(2, dtype=np.float32)
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        history = []
        print(f"\n[Context OOF] Fold {fold+1}/{a.n_splits}: train={len(train_df):,}, val={len(val_df):,}")
        for epoch in range(1, a.epochs + 1):
            model.train(); total, seen = 0.0, 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels"); optimizer.zero_grad(set_to_none=True)
                logits = model(**batch).logits; loss = criterion(logits, labels)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                total += float(loss.item()) * labels.size(0); seen += labels.size(0)
            epoch_loss = total / max(seen, 1); history.append({"epoch": epoch, "train_loss": epoch_loss})
            print(f"[Context OOF] fold={fold} epoch={epoch}/{a.epochs} train_loss={epoch_loss:.4f}")
        model_dir = model_root / f"{a.run_name}_fold_{fold}"; model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir); tokenizer.save_pretrained(model_dir)
        pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False, encoding="utf-8-sig")
        metadata = {
            "model_type": a.run_name, "input_column": a.input_column, "model_name": model_name, "fold": fold, "epochs": a.epochs,
            "max_length": max_length, "context_window": int(manifest["context_window"]),
            "train_rows": len(train_df), "validation_rows": len(val_df), "class_weighted_loss": weighted,
            "seed": seed + fold, "outer_oof_validation_used_for_early_stopping": False,
            "window_purge_strategy": manifest["fold_strategy"], "training_input_column": a.input_column,
        }
        (model_dir / "oof_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        vm, vp = infer(model, val_loader, device); nm, npb = infer(model, novel_loader, device)
        all_oof.append(attach(val_df, fold, vm, vp, "oof_validation"))
        all_novel.append(attach(novel, fold, nm, npb, "novel_validation"))
        summaries.append({"fold": fold, "train_rows": len(train_df), "validation_rows": len(val_df), "novel_rows_scored": len(novel), "epochs": a.epochs, "model_dir": str(model_dir), "last_train_loss": history[-1]["train_loss"]})
        del model, tokenizer, optimizer, train_loader, val_loader, novel_loader
        if device.type == "cuda": torch.cuda.empty_cache()
    oof, novel_all = pd.concat(all_oof, ignore_index=True), pd.concat(all_novel, ignore_index=True)
    expected = int(manifest["train_rows"])
    if len(oof) != expected or oof["oof_row_id"].nunique() != expected or oof["oof_row_id"].duplicated().any():
        raise RuntimeError("OOF predictions do not cover each training row exactly once.")
    if len(novel_all) != len(novel) * a.n_splits: raise RuntimeError("Novel prediction count mismatch.")
    oof.to_csv(output_dir / f"{a.run_name}_oof_predictions.csv", index=False, encoding="utf-8-sig")
    novel_all.to_csv(output_dir / f"{a.run_name}_novel_fold_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(output_dir / f"{a.run_name}_oof_fold_summary.csv", index=False, encoding="utf-8-sig")
    print("\n[Context OOF] Completed.")
    print(f"OOF predictions: {len(oof):,} -> {output_dir / (a.run_name + '_oof_predictions.csv')}")
    print(f"Novel fold predictions: {len(novel_all):,} -> {output_dir / (a.run_name + '_novel_fold_predictions.csv')}")


if __name__ == "__main__":
    main()
