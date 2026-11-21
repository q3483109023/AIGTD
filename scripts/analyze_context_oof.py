#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired evaluation of sentence-only vs context-window OOF predictions.

Reads the four prediction CSVs produced by train_context_oof.py. Writes compact
metrics and cluster-bootstrap summaries; raw prediction files are not modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)


def metrics(y: np.ndarray, p: np.ndarray, score: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=int)
    score = np.asarray(score, dtype=float)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "f1_human": float(f1_score(y, p, pos_label=0, zero_division=0)),
        "f1_machine": float(f1_score(y, p, pos_label=1, zero_division=0)),
        "precision_machine": float(precision_score(y, p, pos_label=1, zero_division=0)),
        "recall_machine": float(recall_score(y, p, pos_label=1, zero_division=0)),
        "precision_human": float(precision_score(y, p, pos_label=0, zero_division=0)),
        "recall_human": float(recall_score(y, p, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan"),
        "average_precision": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else float("nan"),
        "human_support": int((y == 0).sum()),
        "machine_support": int((y == 1).sum()),
    }


def load_pair(path_a: Path, path_b: Path, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    a, b = pd.read_csv(path_a), pd.read_csv(path_b)
    required = {key, "fold", "y_true", "context_p_machine", "context_logit_margin"}
    for name, frame in (("sentence_matched", a), ("context", b)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} file missing columns: {missing}")
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"{name} {key} must be non-null and unique")
        if frame["y_true"].isna().any() or not set(frame["y_true"].astype(int).unique()).issubset({0, 1}):
            raise ValueError(f"{name} y_true must contain only 0/1 labels")
        for col in ("context_p_machine", "context_logit_margin"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
            if not np.isfinite(frame[col].to_numpy(float)).all():
                raise ValueError(f"{name} {col} contains non-finite values")
        if not frame["context_p_machine"].between(0, 1).all():
            raise ValueError(f"{name} probabilities outside [0, 1]")
    a = a.sort_values(key).reset_index(drop=True)
    b = b.sort_values(key).reset_index(drop=True)
    if len(a) != len(b) or not np.array_equal(a[key].to_numpy(), b[key].to_numpy()):
        raise ValueError("Paired OOF files have different row IDs or row counts")
    for col in ("y_true", "fold"):
        if not np.array_equal(a[col].to_numpy(), b[col].to_numpy()):
            raise ValueError(f"Paired OOF files disagree on {col}")
    return a, b


def cluster_bootstrap(a: pd.DataFrame, b: pd.DataFrame, cluster_col: str,
                      reps: int, seed: int) -> list[dict]:
    if cluster_col not in a.columns:
        return [{"cluster": cluster_col, "status": "column_missing"}]
    if a[cluster_col].isna().any():
        return [{"cluster": cluster_col, "status": "missing_cluster_ids"}]
    y = a["y_true"].to_numpy(int)
    pa = (a["context_p_machine"].to_numpy(float) >= 0.5).astype(int)
    pb = (b["context_p_machine"].to_numpy(float) >= 0.5).astype(int)
    groups = a[cluster_col].astype(str).to_numpy()
    unique, inverse = np.unique(groups, return_inverse=True)
    n_groups = len(unique)
    if n_groups < 2:
        return [{"cluster": cluster_col, "clusters": n_groups, "status": "too_few_clusters"}]

    # Compute paired, cluster-resampled differences from weighted confusion counts.
    rng = np.random.default_rng(seed)
    diffs = {m: [] for m in ("macro_f1", "accuracy", "f1_human", "f1_machine")}
    for _ in range(reps):
        multiplicity = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups))
        w = multiplicity[inverse].astype(float)
        total = w.sum()
        if total == 0:
            continue

        def weighted_stats(pred: np.ndarray) -> dict:
            tp = float(w[(y == 1) & (pred == 1)].sum())
            fp = float(w[(y == 0) & (pred == 1)].sum())
            fn = float(w[(y == 1) & (pred == 0)].sum())
            tn = float(w[(y == 0) & (pred == 0)].sum())
            f1_pos = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            f1_neg = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
            return {
                "macro_f1": (f1_pos + f1_neg) / 2,
                "accuracy": (tp + tn) / total,
                "f1_human": f1_neg,
                "f1_machine": f1_pos,
            }

        ma, mb = weighted_stats(pa), weighted_stats(pb)
        for metric_name in diffs:
            diffs[metric_name].append(mb[metric_name] - ma[metric_name])

    rows = []
    for metric_name, values in diffs.items():
        values = np.asarray(values, dtype=float)
        rows.append({
            "cluster": cluster_col, "clusters": n_groups, "reps": len(values),
            "comparison": "context_minus_sentence_matched", "metric": metric_name,
            "mean_difference": float(values.mean()),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "probability_difference_gt_zero": float((values > 0).mean()),
            "status": "ok",
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="results/context_oof")
    p.add_argument("--output-dir", default="results/context_oof_analysis")
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    root, out = Path(args.input_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    a, b = load_pair(
        root / "sentence_matched_oof_predictions.csv",
        root / "context_oof_predictions.csv",
        "oof_row_id",
    )
    integrity = {
        "oof_rows_sentence": int(len(a)),
        "oof_rows_context": int(len(b)),
        "unique_oof_row_ids": int(a["oof_row_id"].nunique()),
        "folds": sorted(int(x) for x in a["fold"].unique()),
        "labels_match": bool(np.array_equal(a["y_true"], b["y_true"])),
        "folds_match": bool(np.array_equal(a["fold"], b["fold"])),
        "prediction_files_paired": True,
    }
    (out / "integrity_report.json").write_text(
        json.dumps(integrity, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    overall = []
    by_fold = []
    for model_name, frame in (("sentence_matched", a), ("context", b)):
        y = frame["y_true"].to_numpy(int)
        score = frame["context_p_machine"].to_numpy(float)
        pred = (score >= 0.5).astype(int)
        overall.append({"model": model_name, "split": "pooled_oof_fixed_threshold_0.5",
                        **metrics(y, pred, score)})
        for fold, g in frame.groupby("fold", sort=True):
            gy = g["y_true"].to_numpy(int)
            gs = g["context_p_machine"].to_numpy(float)
            by_fold.append({"model": model_name, "fold": int(fold),
                            **metrics(gy, (gs >= 0.5).astype(int), gs)})

    # Pooled AUC/AP across independently trained folds can be scale-sensitive.
    # Also report fold-standardized margins as a descriptive pooled ranking score.
    for model_name, frame in (("sentence_matched", a), ("context", b)):
        z = frame["context_logit_margin"].astype(float).copy()
        z = z.groupby(frame["fold"]).transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1.0)
        )
        y = frame["y_true"].to_numpy(int)
        overall.append({
            "model": model_name, "split": "pooled_oof_fold_standardized_margin",
            "n": int(len(frame)), "roc_auc": float(roc_auc_score(y, z)),
            "average_precision": float(average_precision_score(y, z)),
        })

    pd.DataFrame(overall).to_csv(out / "metrics_overall.csv", index=False)
    pd.DataFrame(by_fold).to_csv(out / "metrics_by_fold.csv", index=False)

    bootstrap_rows = []
    for cluster in ("article_key", "article_id", "source_article_id"):
        bootstrap_rows.extend(cluster_bootstrap(a, b, cluster, args.bootstrap_reps, args.seed))
    pd.DataFrame(bootstrap_rows).to_csv(out / "paired_cluster_bootstrap.csv", index=False)

    # Novel validation is scored by every fold model. Average the five fold
    # probabilities per target first, so each target contributes exactly once.
    na, nb = load_pair(
        root / "sentence_matched_novel_fold_predictions.csv",
        root / "context_novel_fold_predictions.csv",
        "novel_row_id",
    ) if False else (None, None)

    def load_novel(path: Path, model_name: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        required = {"novel_row_id", "fold", "y_true", "context_p_machine", "context_logit_margin"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{model_name} Novel file missing columns: {missing}")
        if df[list(required)].isna().any().any():
            raise ValueError(f"{model_name} Novel file contains missing required values")
        df["context_p_machine"] = pd.to_numeric(df["context_p_machine"], errors="coerce")
        df["context_logit_margin"] = pd.to_numeric(df["context_logit_margin"], errors="coerce")
        if not np.isfinite(df["context_p_machine"]).all() or not np.isfinite(df["context_logit_margin"]).all():
            raise ValueError(f"{model_name} Novel file contains non-finite scores")
        if not df["context_p_machine"].between(0, 1).all():
            raise ValueError(f"{model_name} Novel probabilities outside [0, 1]")
        return df

    novel_a = load_novel(root / "sentence_matched_novel_fold_predictions.csv", "sentence_matched")
    novel_b = load_novel(root / "context_novel_fold_predictions.csv", "context")
    for name, df in (("sentence_matched", novel_a), ("context", novel_b)):
        counts = df.groupby("novel_row_id")["fold"].nunique()
        if not (counts == len(integrity["folds"])).all():
            raise ValueError(f"{name} Novel rows do not each have predictions from every fold")
    if set(novel_a["novel_row_id"]) != set(novel_b["novel_row_id"]):
        raise ValueError("Novel prediction files have different novel_row_id sets")
    agg_a = novel_a.groupby("novel_row_id", as_index=False).agg(
        y_true=("y_true", "first"), p_machine=("context_p_machine", "mean"),
        folds=("fold", "nunique"),
    )
    agg_b = novel_b.groupby("novel_row_id", as_index=False).agg(
        y_true=("y_true", "first"), p_machine=("context_p_machine", "mean"),
        folds=("fold", "nunique"),
    )
    agg_a = agg_a.sort_values("novel_row_id").reset_index(drop=True)
    agg_b = agg_b.sort_values("novel_row_id").reset_index(drop=True)
    if not np.array_equal(agg_a["novel_row_id"], agg_b["novel_row_id"]) or not np.array_equal(agg_a["y_true"], agg_b["y_true"]):
        raise ValueError("Novel labels or row IDs disagree between models")
    novel_rows = []
    for model_name, df in (("sentence_matched", agg_a), ("context", agg_b)):
        y = df["y_true"].to_numpy(int)
        s = df["p_machine"].to_numpy(float)
        novel_rows.append({"model": model_name, "split": "novel_validation_mean_probability_over_5_fold_models",
                           **metrics(y, (s >= 0.5).astype(int), s)})
    pd.DataFrame(novel_rows).to_csv(out / "novel_metrics.csv", index=False)

    novel_fold_rows = []
    for model_name, frame in (("sentence_matched", novel_a), ("context", novel_b)):
        for fold, g in frame.groupby("fold", sort=True):
            y = g["y_true"].to_numpy(int)
            s = g["context_p_machine"].to_numpy(float)
            novel_fold_rows.append({"model": model_name, "fold": int(fold),
                                    **metrics(y, (s >= 0.5).astype(int), s)})
    pd.DataFrame(novel_fold_rows).to_csv(out / "novel_metrics_by_fold.csv", index=False)

    print("Context OOF analysis complete.")
    print(f"OOF rows: {len(a):,}; novel targets: {len(agg_a):,}")
    print(f"Outputs: {out.resolve()}")
    print(pd.DataFrame(overall).to_string(index=False))
    print(pd.DataFrame(novel_rows).to_string(index=False))
    print(pd.DataFrame(bootstrap_rows).to_string(index=False))


if __name__ == "__main__":
    main()
