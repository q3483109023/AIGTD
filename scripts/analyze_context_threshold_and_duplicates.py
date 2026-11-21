#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Threshold sensitivity and duplicate/group-boundary audit for Context OOF.

This is a diagnostic only: it does not retrain models, tune a deployable
threshold, delete records, or modify source data.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def safe_auc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def safe_ap(y, score):
    return float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def score_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "n": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "human_f1": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "machine_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "human_precision": float(precision_score(y, pred, pos_label=0, zero_division=0)),
        "human_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "machine_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "machine_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "human_as_machine": int(cm[0, 1]),
        "machine_as_human": int(cm[1, 0]),
        "roc_auc": safe_auc(y, score),
        "average_precision": safe_ap(y, score),
        "brier_score": float(brier_score_loss(y, score)),
    }


def load_prediction_pair(root: Path):
    paths = {
        "sentence": root / "sentence_matched_oof_predictions.csv",
        "context": root / "context_oof_predictions.csv",
    }
    a = pd.read_csv(paths["sentence"], low_memory=False)
    b = pd.read_csv(paths["context"], low_memory=False)
    required = {"oof_row_id", "fold", "y_true", "context_p_machine"}
    for name, df in (("sentence", a), ("context", b)):
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{name} predictions missing columns: {missing}")
        if df["oof_row_id"].isna().any() or df["oof_row_id"].duplicated().any():
            raise ValueError(f"{name} oof_row_id must be unique and non-null")
        df["context_p_machine"] = pd.to_numeric(df["context_p_machine"], errors="coerce")
        if not np.isfinite(df["context_p_machine"].to_numpy(float)).all():
            raise ValueError(f"{name} contains non-finite probabilities")
        if not df["context_p_machine"].between(0, 1).all():
            raise ValueError(f"{name} probabilities outside [0,1]")
    a = a.sort_values("oof_row_id").reset_index(drop=True)
    b = b.sort_values("oof_row_id").reset_index(drop=True)
    if not np.array_equal(a["oof_row_id"], b["oof_row_id"]):
        raise ValueError("Sentence and Context OOF row IDs differ")
    for col in ("fold", "y_true"):
        if not np.array_equal(a[col], b[col]):
            raise ValueError(f"Sentence and Context disagree on {col}")
    return a, b


def threshold_analysis(a: pd.DataFrame, b: pd.DataFrame, out: Path, step: float):
    if not 0 < step <= 0.1:
        raise ValueError("--threshold-step must be in (0, 0.1]")
    coarse = np.arange(step, 0.951, step)
    fine = np.arange(0.95, 0.999951, 0.0005)
    thresholds = np.unique(np.round(np.concatenate([coarse, fine, [0.9999]]), 6))
    rows, fold_rows, opt_rows, dist_rows = [], [], [], []
    for model, df in (("sentence_matched", a), ("context", b)):
        y = df["y_true"].to_numpy(int)
        s = df["context_p_machine"].to_numpy(float)
        for t in thresholds:
            rows.append({"model": model, "split": "pooled_oof", **score_metrics(y, s, t)})
        base = score_metrics(y, s, 0.5)
        best_t, best_m = max(
            ((t, score_metrics(y, s, t)) for t in thresholds),
            key=lambda item: (item[1]["macro_f1"], -abs(item[0] - 0.5)),
        )
        opt_rows.append({
            "model": model, "scope": "pooled_oof_diagnostic_only",
            "selected_threshold": float(best_t), "best_macro_f1": best_m["macro_f1"],
            "macro_f1_at_0_5": base["macro_f1"],
            "macro_f1_gain_vs_0_5": best_m["macro_f1"] - base["macro_f1"],
            "auc_threshold_invariant": base["roc_auc"],
            "ap_threshold_invariant": base["average_precision"],
            "warning": "Threshold selected and evaluated on the same OOF labels; optimistic diagnostic, not an unbiased final estimate.",
        })
        for fold, g in df.groupby("fold", sort=True):
            gy = g["y_true"].to_numpy(int)
            gs = g["context_p_machine"].to_numpy(float)
            for t in thresholds:
                fold_rows.append({"model": model, "fold": int(fold),
                                  **score_metrics(gy, gs, t)})
            best_ft, best_fm = max(
                ((t, score_metrics(gy, gs, t)) for t in thresholds),
                key=lambda item: (item[1]["macro_f1"], -abs(item[0] - 0.5)),
            )
            opt_rows.append({
                "model": model, "scope": f"fold_{int(fold)}_diagnostic_only",
                "selected_threshold": float(best_ft), "best_macro_f1": best_fm["macro_f1"],
                "macro_f1_at_0_5": score_metrics(gy, gs, 0.5)["macro_f1"],
                "macro_f1_gain_vs_0_5": best_fm["macro_f1"] - score_metrics(gy, gs, 0.5)["macro_f1"],
                "auc_threshold_invariant": safe_auc(gy, gs),
                "ap_threshold_invariant": safe_ap(gy, gs),
                "warning": "Per-fold threshold fitted on that same fold's labels; descriptive upper-bound diagnostic only.",
            })
            dist_rows.append({
                "model": model, "fold": int(fold), "n": len(g),
                "p_machine_min": float(np.min(gs)), "p_machine_q01": float(np.quantile(gs, .01)),
                "p_machine_q05": float(np.quantile(gs, .05)), "p_machine_q25": float(np.quantile(gs, .25)),
                "p_machine_median": float(np.median(gs)), "p_machine_q75": float(np.quantile(gs, .75)),
                "p_machine_q95": float(np.quantile(gs, .95)), "p_machine_q99": float(np.quantile(gs, .99)),
                "p_machine_max": float(np.max(gs)),
                "human_score_median": float(np.median(gs[gy == 0])) if np.any(gy == 0) else np.nan,
                "machine_score_median": float(np.median(gs[gy == 1])) if np.any(gy == 1) else np.nan,
                "fraction_below_0_01": float(np.mean(gs < .01)),
                "fraction_above_0_99": float(np.mean(gs > .99)),
            })
    pd.DataFrame(rows).to_csv(out / "threshold_sensitivity_overall.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "threshold_sensitivity_by_fold.csv", index=False)
    pd.DataFrame(opt_rows).to_csv(out / "threshold_optima_diagnostic_only.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(out / "score_distribution_by_fold.csv", index=False)


def prepare_source_frame(path: Path, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {"text"}
    if required - set(df.columns):
        raise ValueError(f"{path} missing text column")
    df = df.copy()
    df["split_name"] = split_name
    df["row_in_split"] = np.arange(len(df), dtype=int)
    df["text_raw"] = df["text"].fillna("").astype(str)
    df["text_norm"] = df["text_raw"].map(normalize_text)
    df["text_exact_key"] = df["text_raw"]
    for col in ("article_key", "article_id", "source_article_id", "sentence_index", "label_id"):
        if col not in df:
            df[col] = pd.NA
    # Existing project convention: article_id is the source ID if source_article_id is absent.
    if df["source_article_id"].isna().all() and not df["article_id"].isna().all():
        df["source_article_id"] = df["article_id"].astype(str)
    return df


def duplicate_group_report(df: pd.DataFrame, key_col: str, min_rows: int = 2):
    valid = df.loc[df[key_col].fillna("").astype(str).ne("")].copy()
    rows = []
    for key, g in valid.groupby(key_col, sort=False, dropna=False):
        if len(g) < min_rows:
            continue
        labels = sorted(g["label_id"].dropna().astype(str).unique().tolist())
        splits = sorted(g["split_name"].astype(str).unique().tolist())
        folds = sorted(pd.to_numeric(g.loc[g["split_name"].eq("train"), "fold"], errors="coerce").dropna().astype(int).unique().tolist()) if "fold" in g else [])
        rows.append({
            "key": str(key), "rows": int(len(g)), "splits": "|".join(splits),
            "split_count": len(splits), "train_rows": int(g["split_name"].eq("train").sum()),
            "validation_rows": int(g["split_name"].eq("validation").sum()),
            "article_key_count": int(g["article_key"].dropna().astype(str).nunique()),
            "article_id_count": int(g["article_id"].dropna().astype(str).nunique()),
            "source_article_id_count": int(g["source_article_id"].dropna().astype(str).nunique()),
            "sentence_index_count": int(g["sentence_index"].dropna().astype(str).nunique()),
            "label_values": "|".join(labels), "label_count": len(labels),
            "conflicting_labels": len(labels) > 1,
            "train_fold_values": "|".join(map(str, folds)),
            "cross_train_validation": "train" in splits and "validation" in splits,
        })
    if not rows:
        return pd.DataFrame(columns=["key", "rows", "splits", "split_count"])
    return pd.DataFrame(rows).sort_values(["rows", "key"], ascending=[False, True])


def source_audit(train_path: Path, val_path: Path, out: Path, manifest_path: Path):
    train = prepare_source_frame(train_path, "train")
    val = prepare_source_frame(val_path, "validation")
    combined = pd.concat([train, val], ignore_index=True, sort=False)
    # Map OOF fold membership to source rows by row order, as in the preparation protocol.
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path, low_memory=False)
        if len(manifest) == len(train) and "fold" in manifest:
            train["fold"] = pd.to_numeric(manifest.sort_values("row_id")["fold"], errors="raise").to_numpy()
            combined = pd.concat([train, val], ignore_index=True, sort=False)
    exact_groups = duplicate_group_report(combined, "text_exact_key")
    norm_groups = duplicate_group_report(combined, "text_norm")
    exact_groups.to_csv(out / "duplicate_groups_exact_text.csv", index=False)
    norm_groups.to_csv(out / "duplicate_groups_normalized_text.csv", index=False)

    # Summary counts count rows participating in duplicate groups, not just groups.
    def group_summary(groups, name):
        if groups.empty:
            return {"kind": name, "duplicate_groups": 0, "rows_in_groups": 0,
                    "cross_split_groups": 0, "conflicting_label_groups": 0}
        return {
            "kind": name, "duplicate_groups": int(len(groups)),
            "rows_in_groups": int(groups["rows"].sum()),
            "cross_split_groups": int(groups["cross_train_validation"].sum()),
            "conflicting_label_groups": int(groups["conflicting_labels"].sum()),
            "groups_spanning_multiple_article_keys": int((groups["article_key_count"] > 1).sum()),
            "groups_spanning_multiple_source_ids": int((groups["source_article_id_count"] > 1).sum()),
        }
    summary = [group_summary(exact_groups, "exact_raw_text"),
               group_summary(norm_groups, "normalized_text")]
    # ID mapping integrity: each article_key should map consistently to IDs.
    mapping_rows = []
    for key, g in combined.groupby("article_key", dropna=False):
        if pd.isna(key):
            continue
        mapping_rows.append({
            "article_key": str(key), "rows": int(len(g)),
            "article_id_nunique": int(g["article_id"].dropna().astype(str).nunique()),
            "source_article_id_nunique": int(g["source_article_id"].dropna().astype(str).nunique()),
            "article_id_values": "|".join(sorted(g["article_id"].dropna().astype(str).unique().tolist())[:20]),
            "source_article_id_values": "|".join(sorted(g["source_article_id"].dropna().astype(str).unique().tolist())[:20]),
        })
    mapping = pd.DataFrame(mapping_rows)
    mapping.to_csv(out / "article_key_id_mapping.csv", index=False)
    # Direct article_id <-> source_article_id mapping; aliases should be one-to-one if truly equivalent.
    id_map = combined[["article_id", "source_article_id"]].dropna().astype(str).drop_duplicates()
    id_map.to_csv(out / "article_id_source_id_pairs.csv", index=False)
    id_to_source = id_map.groupby("article_id")["source_article_id"].nunique()
    source_to_id = id_map.groupby("source_article_id")["article_id"].nunique()
    summary.append({
        "kind": "id_mapping",
        "article_key_with_multiple_article_ids": int((mapping["article_id_nunique"] > 1).sum()) if len(mapping) else 0,
        "article_key_with_multiple_source_ids": int((mapping["source_article_id_nunique"] > 1).sum()) if len(mapping) else 0,
        "article_id_mapping_to_multiple_source_ids": int((id_to_source > 1).sum()),
        "source_id_mapping_to_multiple_article_ids": int((source_to_id > 1).sum()),
        "article_id_source_id_pair_count": int(len(id_map)),
    })
    (out / "duplicate_audit_summary.json").write_text(
        json.dumps({"train_rows": len(train), "validation_rows": len(val),
                    "normalization": "Unicode NFKC + casefold + collapsed whitespace; punctuation retained",
                    "summaries": summary,
                    "interpretation_note": "Duplicate presence is not automatically leakage. Check split membership, fold membership, source/article grouping, and conflicting labels separately."},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prediction-dir", default="results/context_oof")
    p.add_argument("--train", default="data/splits/train.csv")
    p.add_argument("--validation", default="data/splits/validation.csv")
    p.add_argument("--output-dir", default="results/context_oof_threshold_duplicate_audit")
    p.add_argument("--threshold-step", type=float, default=0.01)
    p.add_argument("--row-manifest", default="results/oof_purged_group_audit/row_fold_manifest.csv")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    a, b = load_prediction_pair(Path(args.prediction_dir))
    threshold_analysis(a, b, out, args.threshold_step)
    source_audit(Path(args.train), Path(args.validation), out, Path(args.row_manifest))
    print("Threshold sensitivity and duplicate audit complete.")
    print(f"OOF rows: {len(a):,}; folds: {sorted(a['fold'].astype(int).unique().tolist())}")
    print(f"Outputs: {out.resolve()}")
    print((out / "duplicate_audit_summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
