#!/usr/bin/env python3
"""Diagnose anomalous fold-matched Novel Validation results for AIGTD OOF fusion.

This script is read-only. It does not retrain models, alter predictions, or tune
thresholds. It checks row/fold joins and reports metrics separately for each
Novel subset assigned to a base-model fold, as well as each fold model over the
entire Novel set.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())
    return {
        "n": int(len(y)),
        "human": int((y == 0).sum()),
        "machine": int((y == 1).sum()),
        "machine_rate": float((y == 1).mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "precision_machine": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall_machine": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "precision_human": float(precision_score(y, pred, pos_label=0, zero_division=0)),
        "recall_human": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "roc_auc": float(roc_auc_score(y, score)) if np.unique(y).size == 2 else np.nan,
        "average_precision": float(average_precision_score(y, score)) if np.unique(y).size == 2 else np.nan,
        "score_mean": float(np.mean(score)),
        "score_std": float(np.std(score)),
        "score_min": float(np.min(score)),
        "score_max": float(np.max(score)),
    }


def add_row(rows, scope, model, fold, y, score, threshold):
    row = {"scope": scope, "model": model, "fold": fold, "threshold": threshold}
    row.update(metrics(np.asarray(y, dtype=int), np.asarray(score, dtype=float), threshold))
    rows.append(row)


def compare_join(selected: pd.DataFrame, full: pd.DataFrame, score_cols: list[str], label: str) -> list[str]:
    issues = []
    key = ["fold", "novel_row_id"]
    if selected.duplicated(key).any():
        issues.append(f"{label}: selected file has duplicate (fold, novel_row_id) keys")
        return issues
    if full.duplicated(key).any():
        issues.append(f"{label}: full fold predictions have duplicate (fold, novel_row_id) keys")
        return issues
    cols = [c for c in ["article_key", "sentence_index", "text", "label_id", "y_true"] if c in selected.columns and c in full.columns]
    # `selected` contains exactly one chosen fold prediction per Novel row,
    # whereas `full` intentionally contains all n_splits predictions per row.
    # Therefore use a LEFT join: require every selected (fold, novel_row_id)
    # key to exist in the full source, but do not treat the four unselected
    # predictions for each row as missing records.
    check = selected[key + cols + score_cols].merge(
        full[key + cols + score_cols], on=key, how="left", suffixes=("_sel", "_src"),
        indicator=True, validate="one_to_one"
    )
    missing_count = int((check["_merge"] != "both").sum())
    if missing_count:
        issues.append(f"{label}: {missing_count} selected rows missing in source predictions")
        return issues
    for c in cols:
        a, b = check[f"{c}_sel"], check[f"{c}_src"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            ok = np.isclose(pd.to_numeric(a), pd.to_numeric(b), rtol=1e-7, atol=1e-9, equal_nan=True)
        else:
            ok = a.astype("string").fillna("<NA>").to_numpy() == b.astype("string").fillna("<NA>").to_numpy()
        if not np.all(ok):
            issues.append(f"{label}: {int((~ok).sum())} mismatch(es) in {c}")
    for c in score_cols:
        a = pd.to_numeric(check[f"{c}_sel"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(check[f"{c}_src"], errors="coerce").to_numpy(float)
        if not np.allclose(a, b, rtol=1e-7, atol=1e-9, equal_nan=True):
            issues.append(f"{label}: {c} does not match source fold predictions")
    return issues


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof-dir", default="results/oof")
    ap.add_argument("--fusion-dir", default="results/oof_fusion")
    ap.add_argument("--folds-dir", default="data/oof_folds")
    ap.add_argument("--output-dir", default="results/oof_fusion_diagnostic")
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    oof_dir, fusion_dir, folds_dir, out_dir = map(Path, [args.oof_dir, args.fusion_dir, args.folds_dir, args.output_dir])
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = read_csv(fusion_dir / "novel_fusion_predictions.csv")
    svm_novel = read_csv(oof_dir / "svm_novel_fold_predictions.csv")
    rob_novel = read_csv(oof_dir / "roberta_novel_fold_predictions.csv")
    novel = read_csv(folds_dir / "novel_validation.csv")
    norm_path = fusion_dir / "score_normalization.json"
    norm = json.loads(norm_path.read_text(encoding="utf-8")) if norm_path.exists() else {}
    cfg_path = fusion_dir / "fusion_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}

    required_sel = ["novel_row_id", "fold", "assigned_fold", "y_true", "svm_decision_score", "roberta_logit_margin", "svm_z", "roberta_z"]
    missing = [c for c in required_sel if c not in selected.columns]
    if missing:
        raise ValueError(f"novel_fusion_predictions.csv missing columns: {missing}; available={list(selected.columns)}")
    for name, df in [("selected", selected), ("svm_novel", svm_novel), ("roberta_novel", rob_novel), ("novel", novel)]:
        if "novel_row_id" not in df.columns:
            raise ValueError(f"{name} missing novel_row_id")

    issues = []
    if selected["novel_row_id"].duplicated().any():
        issues.append("selected Novel rows do not have unique novel_row_id")
    if len(selected) != len(novel):
        issues.append(f"selected row count {len(selected)} != canonical Novel count {len(novel)}")
    if not (selected["fold"].astype(int).to_numpy() == selected["assigned_fold"].astype(int).to_numpy()).all():
        issues.append("selected rows contain fold != assigned_fold")

    issues += compare_join(selected, svm_novel, ["svm_decision_score", "svm_p_machine"], "SVM selected join")
    issues += compare_join(selected, rob_novel, ["roberta_logit_margin", "roberta_p_machine"], "RoBERTa selected join")

    # Validate selected labels/text against canonical Novel by novel_row_id.
    canonical_cols = [c for c in ["article_key", "sentence_index", "text", "label_id"] if c in novel.columns and c in selected.columns]
    m = selected[["novel_row_id"] + canonical_cols + ["y_true"]].merge(
        novel[["novel_row_id"] + canonical_cols], on="novel_row_id", how="outer",
        suffixes=("_sel", "_canonical"), indicator=True, validate="one_to_one"
    )
    if not (m["_merge"] == "both").all():
        issues.append("selected/canonical Novel row_id coverage mismatch")
    for c in canonical_cols:
        a, b = m[f"{c}_sel"], m[f"{c}_canonical"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            ok = np.isclose(pd.to_numeric(a), pd.to_numeric(b), rtol=0, atol=0, equal_nan=True)
        else:
            ok = a.astype("string").fillna("<NA>").to_numpy() == b.astype("string").fillna("<NA>").to_numpy()
        if not np.all(ok):
            issues.append(f"selected/canonical mismatch in {c}: {int((~ok).sum())}")

    # Per-fold metrics: (a) each model fold on ALL Novel rows; (b) same model
    # restricted to rows assigned to that fold; (c) pooled selected rows.
    rows = []
    for fold in range(args.n_splits):
        s_all = svm_novel[svm_novel["fold"].astype(int) == fold].sort_values("novel_row_id")
        r_all = rob_novel[rob_novel["fold"].astype(int) == fold].sort_values("novel_row_id")
        if len(s_all) != len(novel) or len(r_all) != len(novel):
            issues.append(f"fold {fold}: full Novel prediction count mismatch (svm={len(s_all)}, roberta={len(r_all)}, expected={len(novel)})")
        if len(s_all):
            y = s_all["y_true"].astype(int).to_numpy()
            add_row(rows, "model_fold_on_all_novel", "svm_raw", fold, y, s_all["svm_decision_score"].to_numpy(float), 0.0)
        if len(r_all):
            y = r_all["y_true"].astype(int).to_numpy()
            add_row(rows, "model_fold_on_all_novel", "roberta_logit", fold, y, r_all["roberta_logit_margin"].to_numpy(float), 0.0)

        part = selected[selected["fold"].astype(int) == fold]
        if len(part):
            y = part["y_true"].astype(int).to_numpy()
            for model_name, score_col in [
                ("svm_raw", "svm_decision_score"),
                ("svm_z", "svm_z"),
                ("roberta_logit", "roberta_logit_margin"),
                ("roberta_z", "roberta_z"),
                ("stacker", "stacker_p_machine"),
            ]:
                if score_col in part.columns:
                    score = pd.to_numeric(part[score_col], errors="coerce").to_numpy(float)
                    # AUC uses raw score. Threshold is only applied for score spaces with known thresholds.
                    if model_name == "svm_z":
                        threshold = float(cfg.get("svm_threshold_z", 0.0))
                    elif model_name == "roberta_z":
                        threshold = float(cfg.get("roberta_threshold_z", 0.0))
                    elif model_name == "stacker":
                        threshold = float(cfg.get("stacker_threshold", 0.5))
                    else:
                        threshold = 0.0
                    add_row(rows, "assigned_subset_by_fold", model_name, fold, y, score, threshold)

    y = selected["y_true"].astype(int).to_numpy()
    for model_name, score_col, threshold_key, default in [
        ("svm_z_pooled", "svm_z", "svm_threshold_z", 0.0),
        ("roberta_z_pooled", "roberta_z", "roberta_threshold_z", 0.0),
        ("stacker_pooled", "stacker_p_machine", "stacker_threshold", 0.5),
    ]:
        if score_col in selected.columns:
            add_row(rows, "pooled_assigned_novel", model_name, "all", y,
                    pd.to_numeric(selected[score_col], errors="coerce").to_numpy(float),
                    float(cfg.get(threshold_key, default)))

    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(out_dir / "fold_matched_diagnostics.csv", index=False, encoding="utf-8-sig")

    # Per-fold label/domain/score distributions for assigned Novel subsets.
    group_rows = []
    for fold, part in selected.groupby("fold", sort=True):
        row = {
            "fold": int(fold), "rows": len(part),
            "unique_articles": part["article_key"].nunique() if "article_key" in part else np.nan,
            "human": int((part["y_true"].astype(int) == 0).sum()),
            "machine": int((part["y_true"].astype(int) == 1).sum()),
            "machine_rate": float((part["y_true"].astype(int) == 1).mean()),
        }
        if "domain" in part:
            for dom, n in part["domain"].value_counts(dropna=False).items():
                row[f"domain::{dom}"] = int(n)
        for c in ["svm_decision_score", "svm_z", "roberta_logit_margin", "roberta_z", "stacker_p_machine"]:
            if c in part:
                vals = pd.to_numeric(part[c], errors="coerce")
                row[f"{c}_mean"] = float(vals.mean())
                row[f"{c}_std"] = float(vals.std(ddof=0))
        group_rows.append(row)
    pd.DataFrame(group_rows).to_csv(out_dir / "assigned_fold_distributions.csv", index=False, encoding="utf-8-sig")

    # Cross-fold AUC decomposition: within-fold AUC and pairwise positive-i vs
    # negative-j comparisons for the scores actually pooled by the selector.
    cross_rows = []
    for score_col in ["svm_z", "roberta_z", "stacker_p_machine"]:
        if score_col not in selected:
            continue
        for i in range(args.n_splits):
            pi = selected[(selected["fold"].astype(int) == i) & (selected["y_true"].astype(int) == 1)][score_col].to_numpy(float)
            for j in range(args.n_splits):
                nj = selected[(selected["fold"].astype(int) == j) & (selected["y_true"].astype(int) == 0)][score_col].to_numpy(float)
                if len(pi) and len(nj):
                    pair_auc = float((pi[:, None] > nj[None, :]).mean() + 0.5 * (pi[:, None] == nj[None, :]).mean())
                    cross_rows.append({"score": score_col, "positive_fold": i, "negative_fold": j,
                                       "n_positive": len(pi), "n_negative": len(nj), "pairwise_auc": pair_auc})
    pd.DataFrame(cross_rows).to_csv(out_dir / "cross_fold_auc_decomposition.csv", index=False, encoding="utf-8-sig")

    lines = [
        "AIGTD fold-matched Novel fusion diagnostic", "=" * 72,
        f"Canonical Novel rows: {len(novel):,}",
        f"Selected fold-matched rows: {len(selected):,}",
        f"Rows per selected fold: {selected['fold'].value_counts().sort_index().to_dict()}",
        f"Assigned folds: {selected['assigned_fold'].value_counts().sort_index().to_dict()}",
        "", "Integrity issues:",
        *( [f"- {x}" for x in issues] if issues else ["- None"] ),
        "", "Per-fold metrics are in fold_matched_diagnostics.csv.",
        "Assigned subset class/domain/score distributions are in assigned_fold_distributions.csv.",
        "Cross-fold positive-vs-negative AUC decomposition is in cross_fold_auc_decomposition.csv.",
        "", "Interpretation guide:",
        "- If each assigned_subset_by_fold AUC is high but pooled_assigned_novel AUC is low, the main issue is cross-fold score/threshold calibration or fold-to-fold distribution shift.",
        "- If assigned_subset_by_fold AUC is also near 0.5, inspect Novel fold assignment composition and base-model generalization on those selected article groups.",
        "- Compare model_fold_on_all_novel AUC to assigned_subset_by_fold AUC to isolate the effect of selecting only one fold's subset.",
        "- Do not use any thresholds or results in this diagnostic to retune on Novel labels.",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print((out_dir / "summary.txt").read_text(encoding="utf-8"))
    print(f"\nOutput: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
