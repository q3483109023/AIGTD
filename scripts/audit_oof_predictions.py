#!/usr/bin/env python3
"""Audit SVM/RoBERTa purged-OOF prediction files before fitting a stacker.

Checks exact OOF row coverage and alignment, per-fold source/text isolation,
Novel Validation fold coverage/alignment, finite score columns, and reports
threshold-independent ranking metrics. It does not train models, tune a
threshold, or modify any input file.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

KEY_OOF = ["oof_row_id"]
KEY_NOVEL = ["novel_row_id"]


def normalize_sentence(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}; found={list(df.columns)}")


def n_unique(df: pd.DataFrame, col: str) -> int:
    return int(df[col].nunique(dropna=False)) if col in df.columns else -1


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def check_identity(left: pd.DataFrame, right: pd.DataFrame, key: str, cols: list[str], label: str) -> list[str]:
    issues: list[str] = []
    merged = left[[key] + [c for c in cols if c in left.columns]].merge(
        right[[key] + [c for c in cols if c in right.columns]],
        on=key, how="outer", suffixes=("_left", "_right"), indicator=True,
        validate="one_to_one",
    )
    non_both = int((merged["_merge"] != "both").sum())
    if non_both:
        issues.append(f"{label}: {non_both} keys are not present exactly once on both sides")
    both = merged.loc[merged["_merge"] == "both"]
    for col in cols:
        lcol, rcol = f"{col}_left", f"{col}_right"
        if lcol not in both.columns or rcol not in both.columns:
            continue
        if col in {"svm_decision_score", "svm_p_machine", "roberta_logit_margin", "roberta_p_machine"}:
            a = pd.to_numeric(both[lcol], errors="coerce").to_numpy(float)
            b = pd.to_numeric(both[rcol], errors="coerce").to_numpy(float)
            ok = np.isclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True)
        else:
            a = both[lcol].astype("string").fillna("<NA>").to_numpy()
            b = both[rcol].astype("string").fillna("<NA>").to_numpy()
            ok = a == b
        bad = int((~ok).sum())
        if bad:
            issues.append(f"{label}: {bad} mismatches in column '{col}'")
    return issues


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof-dir", default="results/oof")
    ap.add_argument("--folds-dir", default="data/oof_folds")
    ap.add_argument("--output-dir", default="results/oof_prediction_audit")
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    oof_dir = Path(args.oof_dir)
    folds_dir = Path(args.folds_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = folds_dir / "manifest.json"
    fold_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(fold_manifest.get("n_splits", -1)) != args.n_splits:
        raise ValueError(
            f"n_splits mismatch: manifest={fold_manifest.get('n_splits')}, requested={args.n_splits}"
        )

    svm_oof = read_csv(oof_dir / "svm_oof_predictions.csv")
    rob_oof = read_csv(oof_dir / "roberta_oof_predictions.csv")
    svm_novel = read_csv(oof_dir / "svm_novel_fold_predictions.csv")
    rob_novel = read_csv(oof_dir / "roberta_novel_fold_predictions.csv")
    novel_base = read_csv(folds_dir / "novel_validation.csv")

    require_columns(svm_oof, ["oof_row_id", "fold", "article_key", "sentence_index", "text", "label_id", "y_true", "svm_decision_score", "svm_p_machine"], "SVM OOF")
    require_columns(rob_oof, ["oof_row_id", "fold", "article_key", "sentence_index", "text", "label_id", "y_true", "roberta_logit_margin", "roberta_p_machine"], "RoBERTa OOF")
    require_columns(svm_novel, ["novel_row_id", "fold", "article_key", "sentence_index", "text", "label_id", "y_true", "svm_decision_score", "svm_p_machine"], "SVM Novel-fold")
    require_columns(rob_novel, ["novel_row_id", "fold", "article_key", "sentence_index", "text", "label_id", "y_true", "roberta_logit_margin", "roberta_p_machine"], "RoBERTa Novel-fold")
    require_columns(novel_base, ["novel_row_id", "article_key", "sentence_index", "text", "label_id"], "Novel base")

    issues: list[str] = []
    checks: list[tuple[str, bool, str]] = []

    expected_train = int(fold_manifest["train_rows"])
    expected_novel = int(fold_manifest["novel_validation_rows"])

    # OOF exact-once coverage.
    for name, df in [("SVM", svm_oof), ("RoBERTa", rob_oof)]:
        unique_count = df["oof_row_id"].nunique()
        valid_ids = set(pd.to_numeric(df["oof_row_id"], errors="coerce").dropna().astype(int)) == set(range(expected_train))
        ok = len(df) == expected_train and unique_count == expected_train and valid_ids
        checks.append((f"{name} OOF row coverage exactly once", bool(ok), f"rows={len(df)}, unique={unique_count}, expected={expected_train}"))
        if not ok:
            issues.append(f"{name} OOF row coverage mismatch")
        dup = int(df["oof_row_id"].duplicated().sum())
        checks.append((f"{name} OOF duplicate oof_row_id", dup == 0, f"duplicates={dup}"))

    # Model OOF alignment by oof_row_id.
    alignment_cols = ["fold", "article_key", "sentence_index", "text", "label_id", "y_true"]
    issues += check_identity(svm_oof, rob_oof, "oof_row_id", alignment_cols, "SVM/RoBERTa OOF alignment")
    checks.append(("SVM/RoBERTa OOF aligned", not any("OOF alignment" in x for x in issues), "keyed by oof_row_id"))

    # Check OOF row identities against materialized validation folds.
    fold_val_parts: list[pd.DataFrame] = []
    for fold in range(args.n_splits):
        val_path = folds_dir / f"fold_{fold}" / "validation.csv"
        val_df = read_csv(val_path)
        require_columns(val_df, ["oof_row_id", "fold", "article_key", "sentence_index", "text", "label_id"], f"fold {fold} validation")
        if not (val_df["fold"].astype(int) == fold).all():
            issues.append(f"Fold {fold}: validation.csv contains incorrect fold values")
        fold_val_parts.append(val_df)
        train_df = read_csv(folds_dir / f"fold_{fold}" / "train.csv")
        require_columns(train_df, ["source_article_id", "article_key", "text"], f"fold {fold} train")
        require_columns(val_df, ["source_article_id"], f"fold {fold} validation")
        tr_source = set(train_df["source_article_id"].astype(str))
        va_source = set(val_df["source_article_id"].astype(str))
        source_overlap = len(tr_source & va_source)
        tr_text = set(train_df["text"].map(normalize_sentence))
        va_text = set(val_df["text"].map(normalize_sentence))
        text_overlap = len(tr_text & va_text)
        checks.append((f"Fold {fold} source-ID isolation", source_overlap == 0, f"overlap={source_overlap}"))
        checks.append((f"Fold {fold} normalized-text purge", text_overlap == 0, f"overlap_groups={text_overlap}"))
        if source_overlap:
            issues.append(f"Fold {fold}: source-ID overlap={source_overlap}")
        if text_overlap:
            issues.append(f"Fold {fold}: normalized-text overlap={text_overlap}")

    fold_val_all = pd.concat(fold_val_parts, ignore_index=True)
    checks.append(("Fold validation rows cover training rows", len(fold_val_all) == expected_train and fold_val_all["oof_row_id"].nunique() == expected_train, f"rows={len(fold_val_all)}, unique={fold_val_all['oof_row_id'].nunique()}"))
    if len(fold_val_all) == expected_train:
        issues += check_identity(svm_oof, fold_val_all, "oof_row_id", ["fold", "article_key", "sentence_index", "text", "label_id", "y_true"], "SVM OOF vs fold validation")
        issues += check_identity(rob_oof, fold_val_all, "oof_row_id", ["fold", "article_key", "sentence_index", "text", "label_id", "y_true"], "RoBERTa OOF vs fold validation")

    # Check numeric score validity.
    score_specs = [
        ("SVM OOF", svm_oof, ["svm_decision_score", "svm_p_machine"]),
        ("RoBERTa OOF", rob_oof, ["roberta_logit_margin", "roberta_p_machine"]),
        ("SVM Novel", svm_novel, ["svm_decision_score", "svm_p_machine"]),
        ("RoBERTa Novel", rob_novel, ["roberta_logit_margin", "roberta_p_machine"]),
    ]
    for name, df, cols in score_specs:
        ok = True
        details: list[str] = []
        for col in cols:
            arr = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            finite = bool(np.isfinite(arr).all())
            details.append(f"{col}:finite={finite}")
            if not finite:
                ok = False
        prob_col = "svm_p_machine" if "svm_p_machine" in cols else "roberta_p_machine"
        prob = pd.to_numeric(df[prob_col], errors="coerce").to_numpy(float)
        range_ok = bool(((prob >= 0) & (prob <= 1)).all())
        details.append(f"{prob_col} in [0,1]={range_ok}")
        ok = ok and range_ok
        checks.append((f"{name} finite score/range check", ok, "; ".join(details)))
        if not ok:
            issues.append(f"{name}: non-finite score or probability outside [0,1]")

    # Novel base coverage and per-fold coverage.
    require_columns(novel_base, ["novel_row_id", "article_key", "sentence_index", "text", "label_id"], "Novel base")
    base_ids = set(pd.to_numeric(novel_base["novel_row_id"], errors="coerce").dropna().astype(int))
    if len(novel_base) != expected_novel or len(base_ids) != expected_novel:
        issues.append(f"Novel base rows/IDs mismatch: rows={len(novel_base)}, unique IDs={len(base_ids)}, expected={expected_novel}")
    for name, df in [("SVM Novel", svm_novel), ("RoBERTa Novel", rob_novel)]:
        expected_rows = expected_novel * args.n_splits
        per_fold_ok = True
        detail = []
        for fold in range(args.n_splits):
            part = df[df["fold"].astype(int) == fold]
            ids = set(pd.to_numeric(part["novel_row_id"], errors="coerce").dropna().astype(int))
            good = len(part) == expected_novel and ids == base_ids and not part["novel_row_id"].duplicated().any()
            per_fold_ok = per_fold_ok and good
            detail.append(f"fold{fold}={len(part)} rows/{len(ids)} IDs")
        ok = len(df) == expected_rows and per_fold_ok
        checks.append((f"{name} has every Novel row once per fold", ok, "; ".join(detail)))
        if not ok:
            issues.append(f"{name}: Novel per-fold coverage mismatch")

    # Verify Novel predictions against the canonical Novel Validation file.
    novel_base_check = novel_base.copy()
    novel_base_check["novel_row_id"] = novel_base_check["novel_row_id"].astype(int)
    for name, df in [("SVM Novel", svm_novel), ("RoBERTa Novel", rob_novel)]:
        for fold in range(args.n_splits):
            part = df[df["fold"].astype(int) == fold].copy()
            part["novel_row_id"] = part["novel_row_id"].astype(int)
            local_issues = check_identity(
                part, novel_base_check, "novel_row_id",
                ["article_key", "sentence_index", "text", "label_id"],
                f"{name} fold {fold} vs canonical Novel Validation",
            )
            issues += local_issues
            if local_issues:
                checks.append((f"{name} fold {fold} matches canonical Novel Validation", False, "; ".join(local_issues)))
            else:
                checks.append((f"{name} fold {fold} matches canonical Novel Validation", True, f"{len(part)} rows"))
        bad_y = int((df["y_true"].astype(int) != df["label_id"].astype(int)).sum())
        checks.append((f"{name} y_true equals label_id", bad_y == 0, f"mismatches={bad_y}"))
        if bad_y:
            issues.append(f"{name}: y_true differs from label_id in {bad_y} rows")

    # Align SVM and RoBERTa Novel outputs by (fold, novel_row_id).
    svm_novel_check = svm_novel.copy()
    rob_novel_check = rob_novel.copy()
    svm_novel_check["fold"] = svm_novel_check["fold"].astype(int)
    rob_novel_check["fold"] = rob_novel_check["fold"].astype(int)
    svm_novel_check["_join_key"] = svm_novel_check["fold"].astype(str) + ":" + svm_novel_check["novel_row_id"].astype(str)
    rob_novel_check["_join_key"] = rob_novel_check["fold"].astype(str) + ":" + rob_novel_check["novel_row_id"].astype(str)
    issues += check_identity(svm_novel_check, rob_novel_check, "_join_key", ["article_key", "sentence_index", "text", "label_id", "y_true"], "SVM/RoBERTa Novel alignment")

    # Verify both models' OOF labels and fold assignment agree with raw materialized data.
    for df_name, df in [("SVM OOF", svm_oof), ("RoBERTa OOF", rob_oof)]:
        bad_y = int((df["y_true"].astype(int) != df["label_id"].astype(int)).sum())
        bad_fold = int((~df["fold"].astype(int).isin(range(args.n_splits))).sum())
        checks.append((f"{df_name} y_true equals label_id", bad_y == 0, f"mismatches={bad_y}"))
        checks.append((f"{df_name} fold values valid", bad_fold == 0, f"invalid={bad_fold}"))
        if bad_y or bad_fold:
            issues.append(f"{df_name}: label or fold value inconsistency")

    # Threshold-independent OOF ranking metrics, pooled and per fold.
    ranking_rows: list[dict[str, Any]] = []
    for model_name, df, score_col in [
        ("svm", svm_oof, "svm_decision_score"),
        ("roberta", rob_oof, "roberta_logit_margin"),
    ]:
        y = df["y_true"].astype(int).to_numpy()
        s = df[score_col].astype(float).to_numpy()
        ranking_rows.append({"split": "oof_pooled", "model": model_name, "fold": "all", "n": len(df), "roc_auc": safe_auc(y, s), "average_precision": safe_ap(y, s), "score_mean": float(np.mean(s)), "score_std": float(np.std(s))})
        for fold in range(args.n_splits):
            part = df[df["fold"].astype(int) == fold]
            yy = part["y_true"].astype(int).to_numpy()
            ss = part[score_col].astype(float).to_numpy()
            ranking_rows.append({"split": "oof_fold", "model": model_name, "fold": fold, "n": len(part), "roc_auc": safe_auc(yy, ss), "average_precision": safe_ap(yy, ss), "score_mean": float(np.mean(ss)), "score_std": float(np.std(ss))})
    pd.DataFrame(ranking_rows).to_csv(out_dir / "oof_ranking_metrics.csv", index=False, encoding="utf-8-sig")

    # Novel fold metrics and ensemble-mean ranking diagnostics (no threshold selection).
    novel_rank_rows: list[dict[str, Any]] = []
    novel_agg_parts = []
    for model_name, df, score_col in [
        ("svm", svm_novel, "svm_decision_score"),
        ("roberta", rob_novel, "roberta_logit_margin"),
    ]:
        for fold in range(args.n_splits):
            part = df[df["fold"].astype(int) == fold]
            y = part["y_true"].astype(int).to_numpy()
            s = part[score_col].astype(float).to_numpy()
            novel_rank_rows.append({"model": model_name, "fold": fold, "n": len(part), "roc_auc": safe_auc(y, s), "average_precision": safe_ap(y, s), "score_mean": float(np.mean(s)), "score_std": float(np.std(s))})
        agg = df.groupby("novel_row_id", as_index=False).agg(
            y_true=("y_true", "first"),
            score_mean=(score_col, "mean"),
            score_std=(score_col, "std"),
            fold_count=("fold", "nunique"),
        )
        y = agg["y_true"].astype(int).to_numpy()
        s = agg["score_mean"].to_numpy(float)
        novel_rank_rows.append({"model": model_name + "_five_fold_mean_score", "fold": "mean", "n": len(agg), "roc_auc": safe_auc(y, s), "average_precision": safe_ap(y, s), "score_mean": float(np.mean(s)), "score_std": float(np.std(s))})
        agg.insert(0, "model", model_name)
        novel_agg_parts.append(agg)
    pd.DataFrame(novel_rank_rows).to_csv(out_dir / "novel_fold_ranking_metrics.csv", index=False, encoding="utf-8-sig")
    if novel_agg_parts:
        pd.concat(novel_agg_parts, ignore_index=True).to_csv(out_dir / "novel_fold_score_stability.csv", index=False, encoding="utf-8-sig")

    # Duplicate/label-conflict diagnostics in OOF meta-training rows.
    oof = svm_oof[["oof_row_id", "fold", "article_key", "text", "y_true"]].copy()
    oof["norm_text"] = oof["text"].map(normalize_sentence)
    grp = oof.groupby("norm_text", dropna=False).agg(
        rows=("oof_row_id", "size"),
        folds=("fold", "nunique"),
        labels=("y_true", "nunique"),
        articles=("article_key", "nunique"),
    ).reset_index()
    duplicate_groups = grp[grp["rows"] > 1].copy()
    conflict_groups = grp[grp["labels"] > 1].copy()
    duplicate_groups.to_csv(out_dir / "oof_duplicate_text_groups.csv", index=False, encoding="utf-8-sig")
    conflict_groups.to_csv(out_dir / "oof_conflicting_text_groups.csv", index=False, encoding="utf-8-sig")

    # Cross-check same labels and key rows between model files.
    summary = [
        "AIGTD OOF prediction audit",
        "=" * 72,
        f"Expected training rows: {expected_train:,}",
        f"SVM OOF rows: {len(svm_oof):,}; unique oof_row_id: {svm_oof['oof_row_id'].nunique():,}",
        f"RoBERTa OOF rows: {len(rob_oof):,}; unique oof_row_id: {rob_oof['oof_row_id'].nunique():,}",
        f"Expected Novel rows per fold: {expected_novel:,}; folds: {args.n_splits}",
        f"SVM Novel-fold rows: {len(svm_novel):,}",
        f"RoBERTa Novel-fold rows: {len(rob_novel):,}",
        f"OOF duplicate normalized-text groups: {len(duplicate_groups):,}",
        f"OOF normalized-text groups with conflicting labels: {len(conflict_groups):,}",
        "",
        "Ranking metrics (threshold-independent):",
        pd.DataFrame(ranking_rows).to_string(index=False),
        "",
        "Novel fold ranking metrics (threshold-independent):",
        pd.DataFrame(novel_rank_rows).to_string(index=False),
        "",
        "Checks:",
    ]
    for name, ok, detail in checks:
        summary.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    summary.extend(["", "Issues:"])
    summary.extend([f"- {x}" for x in issues] if issues else ["- None"])
    summary.extend([
        "",
        "Interpretation notes:",
        "- AUC/AP are ranking diagnostics; no threshold is selected here.",
        "- SVM sigmoid(decision_function) is a monotonic score transform, not calibrated probability.",
        "- Novel five-fold mean-score metrics are diagnostic only; do not choose a final threshold from this report.",
        "- Duplicate normalized texts across different OOF validation folds are allowed by the per-fold purge design; inspect duplicate/conflict reports before meta-model fitting.",
        "- This script does not fit a stacker or tune a threshold.",
    ])
    summary_text = "\n".join(summary)
    (out_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    (out_dir / "checks.json").write_text(json.dumps({"checks": [{"name": n, "passed": bool(ok), "detail": d} for n, ok, d in checks], "issues": issues}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary_text)
    print(f"\nAudit outputs: {out_dir.resolve()}")
    if issues:
        raise SystemExit("OOF audit found issues; do not train a stacker until resolved.")


if __name__ == "__main__":
    main()
