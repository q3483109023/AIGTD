#!/usr/bin/env python3
"""Train purged source-ID OOF LinearSVC models and score Novel Validation per fold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

try:
    from scripts.common import load_config, resolve_path, set_seed
except ImportError:
    from common import load_config, resolve_path, set_seed


def build_pipeline(cfg: dict, seed: int) -> Pipeline:
    svm_cfg = cfg.get("svm", {})
    word_ngram = tuple(svm_cfg.get("word_ngram_range", [1, 2]))
    char_ngram = tuple(svm_cfg.get("char_ngram_range", [2, 5]))
    min_df = int(svm_cfg.get("min_df", 2))
    max_features = svm_cfg.get("max_features", 250000)
    if max_features is not None:
        max_features = int(max_features)
    features = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=word_ngram, min_df=min_df,
            max_features=max_features, sublinear_tf=True, strip_accents="unicode",
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char", ngram_range=char_ngram, min_df=min_df,
            max_features=max_features, sublinear_tf=True, strip_accents="unicode",
        )),
    ])
    classifier = LinearSVC(
        C=float(svm_cfg.get("C", 2.0)),
        max_iter=int(svm_cfg.get("max_iter", 3000)),
        class_weight="balanced",
        random_state=seed,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def add_scores(frame: pd.DataFrame, model: Pipeline, fold: int, split_name: str) -> pd.DataFrame:
    result = frame.copy()
    scores = model.decision_function(result["text"].astype(str))
    result["fold"] = fold
    result["split"] = split_name
    result["svm_decision_score"] = scores.astype(float)
    result["svm_p_machine"] = expit(scores).astype(float)
    if "label_id" in result.columns:
        result["y_true"] = result["label_id"].astype(int)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--folds-dir", default="data/oof_folds")
    parser.add_argument("--output-dir", default="results/oof")
    parser.add_argument("--model-dir", default="models/oof/svm")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("project", {}).get("random_seed", 42))
    set_seed(seed)
    folds_dir = Path(args.folds_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    novel_path = folds_dir / "novel_validation.csv"
    manifest_path = folds_dir / "manifest.json"
    if not manifest_path.exists() or not novel_path.exists():
        raise FileNotFoundError("Missing prepared OOF folds. Run scripts/prepare_oof_folds.py first.")
    fold_meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(fold_meta["n_splits"]) != args.n_splits:
        raise ValueError(f"Prepared folds={fold_meta['n_splits']} but requested={args.n_splits}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    novel_df = pd.read_csv(novel_path, encoding="utf-8-sig")
    if len(novel_df) == 0:
        raise ValueError("Novel Validation is empty.")

    oof_parts: list[pd.DataFrame] = []
    novel_parts: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for fold in range(args.n_splits):
        fold_dir = folds_dir / f"fold_{fold}"
        train_path = fold_dir / "train.csv"
        val_path = fold_dir / "validation.csv"
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(f"Missing files for fold {fold}: {train_path}, {val_path}")
        train_df = pd.read_csv(train_path, encoding="utf-8-sig")
        val_df = pd.read_csv(val_path, encoding="utf-8-sig")
        if train_df["label_id"].nunique() != 2:
            raise ValueError(f"Fold {fold}: training data does not contain both labels.")

        fold_seed = seed + fold
        print(
            f"\n[SVM OOF] Fold {fold + 1}/{args.n_splits}: "
            f"train={len(train_df):,}, validation={len(val_df):,}, "
            f"human/machine={int((train_df.label_id == 0).sum()):,}/{int((train_df.label_id == 1).sum()):,}"
        )
        model = build_pipeline(cfg, fold_seed)
        model.fit(train_df["text"].astype(str), train_df["label_id"].astype(int))

        val_scored = add_scores(val_df, model, fold, "oof_validation")
        oof_parts.append(val_scored)
        novel_scored = add_scores(novel_df, model, fold, "novel_validation")
        novel_parts.append(novel_scored)

        model_path = model_dir / f"svm_fold_{fold}.joblib"
        joblib.dump({
            "pipeline": model,
            "fold": fold,
            "seed": fold_seed,
            "score_transform": "sigmoid(decision_function); relative score, not calibrated probability",
            "training_rows": int(len(train_df)),
            "purge_strategy": fold_meta["fold_strategy"],
        }, model_path, compress=3)
        summaries.append({
            "fold": fold,
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "novel_validation_rows_scored": len(novel_df),
            "model_path": str(model_path),
        })
        print(f"[SVM OOF] Fold {fold + 1} saved: {model_path}")

    oof = pd.concat(oof_parts, ignore_index=True)
    novel_all = pd.concat(novel_parts, ignore_index=True)
    if "oof_row_id" not in oof.columns:
        raise ValueError("OOF validation data lacks oof_row_id.")
    if oof["oof_row_id"].duplicated().any():
        dup = int(oof["oof_row_id"].duplicated().sum())
        raise RuntimeError(f"OOF predictions contain {dup} duplicated oof_row_id rows.")
    expected_train_rows = int(fold_meta["train_rows"])
    if len(oof) != expected_train_rows or oof["oof_row_id"].nunique() != expected_train_rows:
        raise RuntimeError(
            f"OOF coverage mismatch: rows={len(oof)}, unique={oof['oof_row_id'].nunique()}, expected={expected_train_rows}"
        )
    expected_novel_rows = len(novel_df) * args.n_splits
    if len(novel_all) != expected_novel_rows:
        raise RuntimeError(f"Novel fold predictions={len(novel_all)}, expected={expected_novel_rows}")

    oof_path = output_dir / "svm_oof_predictions.csv"
    novel_out_path = output_dir / "svm_novel_fold_predictions.csv"
    oof.to_csv(oof_path, index=False, encoding="utf-8-sig")
    novel_all.to_csv(novel_out_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(output_dir / "svm_oof_fold_summary.csv", index=False, encoding="utf-8-sig")

    print("\n[SVM OOF] Completed.")
    print(f"OOF predictions: {len(oof):,} rows -> {oof_path}")
    print(f"Novel predictions: {len(novel_all):,} rows ({args.n_splits} x {len(novel_df):,}) -> {novel_out_path}")
    print("Columns include raw svm_decision_score and sigmoid-transformed svm_p_machine.")


if __name__ == "__main__":
    main()
