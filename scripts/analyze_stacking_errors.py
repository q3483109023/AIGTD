#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AIGTD Stacking error analysis on the fold-matched Novel Validation file.

Expected input:
  results/oof_fusion/novel_fusion_predictions.csv

The script compares Stacking, RoBERTa, SVM and SVM-veto predictions, audits
row/key consistency, reports error groups and subgroup metrics, and computes
paired cluster-bootstrap CIs for metric differences.

Primary bootstrap cluster: source article_id.
Sensitivity bootstrap cluster: article_key.
Note: article_id clustering does not fully remove dependence induced by
identical normalized sentences shared across different source IDs. That
limitation is reported explicitly.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PREDICTIONS = {
    "stacker": "stacker_pred",
    "roberta": "roberta_pred_oof_threshold",
    "svm": "svm_pred_oof_threshold",
    "veto": "veto_pred",
}
SCORE_COLUMNS = {
    "stacker": "stacker_p_machine",
    "roberta": "roberta_p_machine",
    "svm": "svm_p_machine",
}


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    s = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", s).strip()


def word_count(value: object) -> int:
    if pd.isna(value):
        return 0
    # Alphanumeric/apostrophe token count; consistent within this report.
    return len(re.findall(r"\b[\w]+(?:['’][\w]+)*\b", str(value), flags=re.UNICODE))


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def calc_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray | None = None) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: Dict[str, float] = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_machine": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall_machine": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "f1_machine": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "precision_human": float(precision_score(y, pred, pos_label=0, zero_division=0)),
        "recall_human": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "f1_human": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "human_support": int((y == 0).sum()),
        "machine_support": int((y == 1).sum()),
        "fp_rate_human": safe_div(fp, (y == 0).sum()),
        "fn_rate_machine": safe_div(fn, (y == 1).sum()),
    }
    if score is not None:
        score = np.asarray(score, dtype=float)
        if np.isfinite(score).all() and len(np.unique(y)) == 2:
            out["roc_auc"] = float(roc_auc_score(y, score))
            out["average_precision"] = float(average_precision_score(y, score))
        else:
            out["roc_auc"] = np.nan
            out["average_precision"] = np.nan
    return out


def validate_and_load(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"预测文件不存在：{path}")
    df = pd.read_csv(path)
    errors: List[str] = []
    required = {
        "novel_row_id", "article_id", "article_key", "sentence_index",
        "text", "label_id", "y_true", "domain", "assigned_fold",
        "stacker_p_machine", "stacker_pred", "roberta_pred_oof_threshold",
        "svm_pred_oof_threshold", "veto_pred", "roberta_p_machine",
        "svm_p_machine",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"输入文件缺少必要字段：{missing}")
    if df["novel_row_id"].isna().any():
        errors.append(f"novel_row_id 缺失：{int(df['novel_row_id'].isna().sum())} 行")
    if df["novel_row_id"].duplicated().any():
        errors.append(f"novel_row_id 重复：{int(df['novel_row_id'].duplicated().sum())} 行")
    if not np.array_equal(df["y_true"].astype(int).to_numpy(), df["label_id"].astype(int).to_numpy()):
        errors.append("y_true 与 label_id 存在不一致")
    for name, col in PREDICTIONS.items():
        if col not in df:
            errors.append(f"缺少预测列：{col}")
            continue
        vals = set(df[col].dropna().astype(int).unique())
        if not vals.issubset({0, 1}) or df[col].isna().any():
            errors.append(f"{col} 存在非二元值或缺失：{vals}")
    if df["article_key"].isna().any() or df["article_id"].isna().any():
        errors.append("article_key 或 article_id 存在缺失，Bootstrap 分组可能不可靠")
    # Labels and fold IDs must be well-formed.
    y_values = set(df["y_true"].dropna().astype(int).unique())
    if not y_values.issubset({0, 1}) or df["y_true"].isna().any():
        errors.append(f"y_true 存在非二元值或缺失：{y_values}")
    if df["assigned_fold"].isna().any():
        errors.append("assigned_fold 存在缺失")
    # Each article record must map to exactly one source ID.
    a_source = df.groupby("article_key", dropna=False)["article_id"].nunique(dropna=False)
    if (a_source > 1).any():
        errors.append(f"同一 article_key 对应多个 article_id：{int((a_source > 1).sum())} 个")
    # Each article record and each source ID must have a single assigned fold.
    a_fold = df.groupby("article_key", dropna=False)["assigned_fold"].nunique(dropna=False)
    if (a_fold > 1).any():
        errors.append(f"同一 article_key 被分配到多个 Fold：{int((a_fold > 1).sum())} 个")
    s_fold = df.groupby("article_id", dropna=False)["assigned_fold"].nunique(dropna=False)
    if (s_fold > 1).any():
        errors.append(f"同一 article_id 被分配到多个 Fold：{int((s_fold > 1).sum())} 个")
    if "norm_text" in df.columns:
        computed = df["text"].map(normalize_text)
        mismatch = (computed != df["norm_text"].fillna("").astype(str)).sum()
        if mismatch:
            errors.append(f"norm_text 与按 NFKC+casefold+空白归一重算结果不一致：{int(mismatch)} 行")
    else:
        df["norm_text"] = df["text"].map(normalize_text)
    df["word_count_analysis"] = df["text"].map(word_count)
    return df, errors


def group_comparison(df: pd.DataFrame, pred_a: str, pred_b: str, name_a: str, name_b: str) -> pd.DataFrame:
    y = df["y_true"].astype(int)
    a_ok = df[pred_a].astype(int).eq(y)
    b_ok = df[pred_b].astype(int).eq(y)
    group = np.select(
        [a_ok & b_ok, a_ok & ~b_ok, ~a_ok & b_ok, ~a_ok & ~b_ok],
        ["both_correct", f"{name_a}_only_correct", f"{name_b}_only_correct", "both_wrong"],
        default="unknown",
    )
    out = df.copy()
    out["comparison_group"] = group
    out[f"{name_a}_correct"] = a_ok.astype(int)
    out[f"{name_b}_correct"] = b_ok.astype(int)
    return out


def group_summary(df: pd.DataFrame, group_col: str = "comparison_group") -> pd.DataFrame:
    rows = []
    for group_name, g in df.groupby(group_col, dropna=False, sort=False):
        y = g["y_true"].astype(int)
        rows.append({
            "comparison_group": group_name,
            "n": len(g),
            "share": len(g) / len(df) if len(df) else 0,
            "human": int((y == 0).sum()),
            "machine": int((y == 1).sum()),
            "machine_share": float((y == 1).mean()) if len(g) else np.nan,
            "mean_word_count": g["word_count_analysis"].mean(),
            "median_word_count": g["word_count_analysis"].median(),
            "mean_char_count": g["text"].fillna("").astype(str).str.len().mean(),
            "mean_punctuation_count": g["text"].fillna("").astype(str).map(lambda s: len(re.findall(r"[^\w\s]", s))).mean(),
            "mean_svm_score": g["svm_p_machine"].mean(),
            "mean_roberta_score": g["roberta_p_machine"].mean(),
            "mean_stacker_score": g["stacker_p_machine"].mean(),
            "unique_article_keys": g["article_key"].nunique(),
            "unique_source_ids": g["article_id"].nunique(),
        })
    return pd.DataFrame(rows)


def per_model_error_files(df: pd.DataFrame, outdir: Path) -> None:
    y = df["y_true"].astype(int)
    for model, pred_col in PREDICTIONS.items():
        err = df.loc[df[pred_col].astype(int).ne(y)].copy()
        err["error_type"] = np.where(err["y_true"].astype(int).eq(0), "FP_human_as_machine", "FN_machine_as_human")
        err.to_csv(outdir / f"{model}_all_errors.csv", index=False, encoding="utf-8-sig")
        err.loc[err["error_type"].eq("FP_human_as_machine")].to_csv(
            outdir / f"{model}_false_positives.csv", index=False, encoding="utf-8-sig"
        )
        err.loc[err["error_type"].eq("FN_machine_as_human")].to_csv(
            outdir / f"{model}_false_negatives.csv", index=False, encoding="utf-8-sig"
        )
        score_col = SCORE_COLUMNS.get(model)
        if score_col:
            err["distance_to_0_5"] = (err[score_col] - 0.5).abs()
        # Deterministic examples: up to 50 rows, sorted by word count then source ID.
        err.sort_values(["word_count_analysis", "article_id", "sentence_index"], inplace=True)
        err.head(50).to_csv(outdir / f"{model}_error_examples.csv", index=False, encoding="utf-8-sig")


def subgroup_metrics(df: pd.DataFrame, group_column: str, out_path: Path) -> None:
    rows = []
    for subgroup, g in df.groupby(group_column, dropna=False, sort=True):
        if len(g) == 0:
            continue
        for model, pred_col in PREDICTIONS.items():
            score_col = SCORE_COLUMNS.get(model)
            met = calc_metrics(g["y_true"].to_numpy(int), g[pred_col].to_numpy(int),
                               g[score_col].to_numpy(float) if score_col else None)
            rows.append({"group_column": group_column, "group": subgroup, "model": model, **met})
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def confusion_counts_weighted(y: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> Tuple[float, float, float, float]:
    tn = float(weights[(y == 0) & (pred == 0)].sum())
    fp = float(weights[(y == 0) & (pred == 1)].sum())
    fn = float(weights[(y == 1) & (pred == 0)].sum())
    tp = float(weights[(y == 1) & (pred == 1)].sum())
    return tn, fp, fn, tp


def macro_f1_from_counts(tn: float, fp: float, fn: float, tp: float) -> float:
    f1_pos = safe_div(2 * tp, 2 * tp + fp + fn)
    f1_neg = safe_div(2 * tn, 2 * tn + fp + fn)
    return (f1_pos + f1_neg) / 2.0


def cluster_bootstrap(df: pd.DataFrame, cluster_col: str, reps: int, seed: int, out_rows: List[dict]) -> None:
    """Paired cluster bootstrap for Stacker minus each comparator."""
    if cluster_col not in df.columns:
        return
    clusters = df[cluster_col].astype(str).to_numpy()
    unique_clusters, inverse = np.unique(clusters, return_inverse=True)
    n_clusters = len(unique_clusters)
    if n_clusters < 20:
        out_rows.append({"cluster_unit": cluster_col, "comparison": "all", "clusters": n_clusters,
                         "reps": 0, "status": "too_few_clusters"})
        return
    y = df["y_true"].to_numpy(int)
    preds = {m: df[c].to_numpy(int) for m, c in PREDICTIONS.items()}
    rng = np.random.default_rng(seed)
    # Multinomial cluster multiplicities are equivalent to sampling n_clusters clusters with replacement.
    comparisons = ["roberta", "svm", "veto"]
    diffs = {m: {"macro_f1": [], "accuracy": []} for m in comparisons}
    for _ in range(reps):
        counts = rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters))
        weights = counts[inverse].astype(float)
        total = weights.sum()
        if total <= 0:
            continue
        st_tn, st_fp, st_fn, st_tp = confusion_counts_weighted(y, preds["stacker"], weights)
        st_macro = macro_f1_from_counts(st_tn, st_fp, st_fn, st_tp)
        st_acc = (st_tn + st_tp) / total
        for other in comparisons:
            ot_tn, ot_fp, ot_fn, ot_tp = confusion_counts_weighted(y, preds[other], weights)
            ot_macro = macro_f1_from_counts(ot_tn, ot_fp, ot_fn, ot_tp)
            ot_acc = (ot_tn + ot_tp) / total
            diffs[other]["macro_f1"].append(st_macro - ot_macro)
            diffs[other]["accuracy"].append(st_acc - ot_acc)
    for other in comparisons:
        for metric in ["macro_f1", "accuracy"]:
            vals = np.asarray(diffs[other][metric], dtype=float)
            out_rows.append({
                "cluster_unit": cluster_col,
                "comparison": f"stacker_minus_{other}",
                "metric": metric,
                "clusters": n_clusters,
                "reps": len(vals),
                "mean_difference": float(vals.mean()),
                "ci95_low": float(np.quantile(vals, 0.025)),
                "ci95_high": float(np.quantile(vals, 0.975)),
                "probability_difference_gt_zero": float((vals > 0).mean()),
                "status": "ok",
            })


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze AIGTD Stacking errors on Novel Validation.")
    parser.add_argument("--input", default="results/oof_fusion/novel_fusion_predictions.csv")
    parser.add_argument("--output-dir", default="results/stacking_error_analysis")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    df, issues = validate_and_load(input_path)
    if issues:
        (outdir / "integrity_issues.txt").write_text("\n".join(issues), encoding="utf-8")
        raise ValueError("输入文件完整性检查失败：\n- " + "\n- ".join(issues))

    # Canonical order and basic identity checks.
    df = df.sort_values("novel_row_id").reset_index(drop=True)
    n = len(df)
    if n != df["novel_row_id"].nunique():
        raise ValueError("novel_row_id not unique after sorting")

    # Metrics for all models.
    metric_rows = []
    y = df["y_true"].to_numpy(int)
    for model, pred_col in PREDICTIONS.items():
        score_col = SCORE_COLUMNS.get(model)
        metrics = calc_metrics(y, df[pred_col].to_numpy(int),
                               df[score_col].to_numpy(float) if score_col else None)
        metric_rows.append({"split": "novel_validation_fold_matched", "model": model, **metrics})
    pd.DataFrame(metric_rows).to_csv(outdir / "metrics_all_models.csv", index=False, encoding="utf-8-sig")

    # Primary comparison: Stacker vs RoBERTa.
    comp = group_comparison(df, "stacker_pred", "roberta_pred_oof_threshold", "stacker", "roberta")
    group_summary(comp).to_csv(outdir / "stacker_vs_roberta_groups.csv", index=False, encoding="utf-8-sig")
    for group in ["both_correct", "stacker_only_correct", "roberta_only_correct", "both_wrong"]:
        g = comp.loc[comp["comparison_group"].eq(group)].copy()
        g.to_csv(outdir / f"{group}.csv", index=False, encoding="utf-8-sig")
        # Up to 100 representative rows, deterministic order. Full group is retained above.
        if len(g):
            sort_cols = ["stacker_p_machine", "roberta_p_machine"]
            g.sort_values(sort_cols, ascending=False).head(100).to_csv(
                outdir / f"{group}_examples_top100.csv", index=False, encoding="utf-8-sig"
            )

    # Cross-comparison with all models; useful to understand SVM/Veto too.
    per_model_error_files(df, outdir)

    # Domain and sentence-length subgroup metrics.
    df["length_bucket"] = pd.cut(
        df["word_count_analysis"],
        bins=[-1, 8, 15, 25, 40, np.inf],
        labels=["1-8", "9-15", "16-25", "26-40", "41+"],
    ).astype(str)
    subgroup_metrics(df, "domain", outdir / "metrics_by_domain.csv")
    subgroup_metrics(df, "length_bucket", outdir / "metrics_by_length.csv")

    # Duplicate and conflicting normalized text analysis.
    dup = df.groupby("norm_text", dropna=False).agg(
        occurrences=("novel_row_id", "size"),
        unique_article_keys=("article_key", "nunique"),
        unique_source_ids=("article_id", "nunique"),
        unique_labels=("y_true", "nunique"),
        text_sample=("text", "first"),
    ).reset_index()
    dup = dup.loc[dup["occurrences"] > 1].copy()
    dup.to_csv(outdir / "duplicate_text_groups.csv", index=False, encoding="utf-8-sig")
    dup.loc[dup["unique_labels"] > 1].to_csv(
        outdir / "conflicting_label_text_groups.csv", index=False, encoding="utf-8-sig"
    )
    conflict_norms = set(dup.loc[dup["unique_labels"] > 1, "norm_text"])
    df.loc[df["norm_text"].isin(conflict_norms)].to_csv(
        outdir / "conflicting_label_text_occurrences.csv", index=False, encoding="utf-8-sig"
    )

    # Article and source ID level performance summaries.
    for group_col, out_name in [("article_key", "metrics_by_article_record.csv"), ("article_id", "metrics_by_source_id.csv")]:
        rows = []
        for key, g in df.groupby(group_col, dropna=False):
            for model, pred_col in PREDICTIONS.items():
                met = calc_metrics(g["y_true"].to_numpy(int), g[pred_col].to_numpy(int))
                rows.append({group_col: key, "model": model, **met})
        pd.DataFrame(rows).to_csv(outdir / out_name, index=False, encoding="utf-8-sig")

    # Paired cluster bootstrap; source-ID is primary, article_key is sensitivity analysis.
    boot_rows: List[dict] = []
    cluster_bootstrap(df, "article_id", args.bootstrap_reps, args.seed, boot_rows)
    cluster_bootstrap(df, "article_key", args.bootstrap_reps, args.seed + 1, boot_rows)
    pd.DataFrame(boot_rows).to_csv(outdir / "cluster_bootstrap_differences.csv", index=False, encoding="utf-8-sig")

    # Sentence-level exact paired McNemar tests: supplementary, not cluster-adjusted.
    mcnemar_rows = []
    try:
        from scipy.stats import binomtest
        for other in ["roberta", "svm", "veto"]:
            pcol = PREDICTIONS[other]
            stack_ok = df["stacker_pred"].astype(int).eq(df["y_true"].astype(int))
            other_ok = df[pcol].astype(int).eq(df["y_true"].astype(int))
            b = int((stack_ok & ~other_ok).sum())
            c = int((~stack_ok & other_ok).sum())
            pval = float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue) if b + c else 1.0
            mcnemar_rows.append({"comparison": f"stacker_vs_{other}", "stacker_only_correct": b,
                                 "other_only_correct": c, "discordant_total": b + c,
                                 "exact_mcnemar_binomial_p": pval,
                                 "note": "Sentence-level paired test; does not account for within-source dependence."})
    except Exception as exc:
        mcnemar_rows.append({"comparison": "unavailable", "note": f"SciPy unavailable: {exc}"})
    pd.DataFrame(mcnemar_rows).to_csv(outdir / "paired_mcnemar_sentence_level.csv", index=False, encoding="utf-8-sig")

    # Compact human-readable summary.
    metric_df = pd.DataFrame(metric_rows).set_index("model")
    group_counts = comp["comparison_group"].value_counts()
    lines = [
        "AIGTD Stacking error analysis",
        "=" * 72,
        f"Input: {input_path}",
        f"Novel rows: {n}",
        f"Unique article_key: {df['article_key'].nunique()}",
        f"Unique source article_id: {df['article_id'].nunique()}",
        f"Assigned folds: {df['assigned_fold'].nunique()} -> {sorted(df['assigned_fold'].unique().tolist())}",
        f"Human/Machine: {(y == 0).sum()}/{(y == 1).sum()}",
        "",
        "Metrics:",
    ]
    for model in PREDICTIONS:
        r = metric_df.loc[model]
        lines.append(
            f"- {model}: Acc={r['accuracy']:.4f}, Macro-F1={r['macro_f1']:.4f}, "
            f"F1_human={r['f1_human']:.4f}, F1_machine={r['f1_machine']:.4f}, "
            f"FP={int(r['fp'])}, FN={int(r['fn'])}, ROC-AUC={r.get('roc_auc', np.nan):.4f}"
        )
    lines += ["", "Stacker vs RoBERTa correctness groups:"]
    for group in ["both_correct", "stacker_only_correct", "roberta_only_correct", "both_wrong"]:
        lines.append(f"- {group}: {int(group_counts.get(group, 0))} ({group_counts.get(group, 0)/n:.2%})")
    lines += [
        "",
        "Duplicate text audit:",
        f"- Duplicate normalized-text groups: {len(dup)}",
        f"- Conflicting-label normalized-text groups: {len(conflict_norms)}",
        f"- Rows in conflicting-label groups: {int(df['norm_text'].isin(conflict_norms).sum())}",
        "",
        "Bootstrap:",
        f"- Repetitions: {args.bootstrap_reps}; seed={args.seed}",
        "- Primary cluster: article_id (source ID); sensitivity: article_key.",
        "- Shared duplicate text across different source IDs may induce residual dependence; interpret CIs with this limitation.",
        "",
        "Interpretation notes:",
        "- All classifications use the saved prediction columns; no thresholds were retuned.",
        "- FP means human(0) predicted machine(1); FN means machine(1) predicted human(0).",
        "- Group and subgroup differences are descriptive; do not infer causal mechanisms from sentence style.",
        "- Cluster bootstrap resamples whole groups and computes paired metric differences on identical resamples.",
    ]
    (outdir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("AIGTD Stacking error analysis completed.")
    print(f"Rows: {n}; article_key: {df['article_key'].nunique()}; source IDs: {df['article_id'].nunique()}")
    print("\nNovel metrics:")
    print(pd.DataFrame(metric_rows)[["model", "accuracy", "macro_f1", "f1_human", "f1_machine", "fp", "fn", "roc_auc"]].to_string(index=False))
    print("\nStacker vs RoBERTa groups:")
    print(group_summary(comp)[["comparison_group", "n", "share", "human", "machine", "mean_word_count"]].to_string(index=False))
    print(f"\nOutput: {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
