#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare AIGTD SVM and RoBERTa predictions on identical validation samples.

Default inputs (relative to project root):
  results/svm_tfidf_predictions.csv
  results/bert_predictions.csv
  results/svm_tfidf_novel_predictions.csv
  results/bert_novel_predictions.csv

Outputs are written to results/model_comparison/.
Labels: human=0, machine=1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

KEY_COLS = ["article_key", "sentence_index"]
REQUIRED_COLS = KEY_COLS + ["text", "y_true", "y_pred"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SVM and RoBERTa predictions for AIGTD validation sets."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--full-svm", type=Path,
        default=Path("results/svm_tfidf_predictions.csv")
    )
    parser.add_argument(
        "--full-roberta", type=Path,
        default=Path("results/bert_predictions.csv")
    )
    parser.add_argument(
        "--novel-svm", type=Path,
        default=Path("results/svm_tfidf_novel_predictions.csv")
    )
    parser.add_argument(
        "--novel-roberta", type=Path,
        default=Path("results/bert_novel_predictions.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/model_comparison")
    )
    return parser.parse_args()


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def to_binary(series: pd.Series, column: str, path: Path) -> pd.Series:
    """Accept numeric 0/1 or human/machine labels."""
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
    else:
        text = series.astype(str).str.strip().str.casefold()
        mapping = {"human": 0, "machine": 1, "0": 0, "1": 1}
        values = text.map(mapping)
    if values.isna().any():
        bad = series[values.isna()].astype(str).value_counts().head(10).to_dict()
        raise ValueError(f"{path}: 列 {column} 存在无法解析的标签：{bad}")
    values = values.astype(int)
    unexpected = sorted(set(values.unique()) - {0, 1})
    if unexpected:
        raise ValueError(f"{path}: 列 {column} 出现非二分类取值：{unexpected}")
    return values


def load_predictions(path: Path, model_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到预测文件：{path}")
    df = pd.read_csv(path, dtype={"article_key": "string"}, encoding="utf-8-sig")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: 缺少必要字段：{missing}; 实际字段：{df.columns.tolist()}")

    df = df.copy()
    df["article_key"] = df["article_key"].astype("string").fillna("")
    df["sentence_index"] = pd.to_numeric(df["sentence_index"], errors="coerce")
    if df["sentence_index"].isna().any():
        raise ValueError(f"{path}: sentence_index 存在空值或非数字值")
    df["sentence_index"] = df["sentence_index"].astype(int)
    if (df["article_key"].str.len() == 0).any():
        raise ValueError(f"{path}: article_key 存在空值")

    duplicated = df.duplicated(KEY_COLS, keep=False)
    if duplicated.any():
        example = df.loc[duplicated, KEY_COLS].head(10).to_dict("records")
        raise ValueError(
            f"{path}: {int(duplicated.sum())} 行存在重复样本主键 {KEY_COLS}，示例：{example}"
        )

    df["y_true"] = to_binary(df["y_true"], "y_true", path)
    df["y_pred"] = to_binary(df["y_pred"], "y_pred", path)
    if "p_machine" in df.columns:
        df["p_machine"] = pd.to_numeric(df["p_machine"], errors="coerce")

    # Retain source metadata, but prefix model-specific prediction columns.
    keep = [c for c in df.columns if c not in {"y_pred", "p_machine", "predicted_label", "is_correct"}]
    df = df[keep + [c for c in ["y_pred", "p_machine", "predicted_label", "is_correct"] if c in df.columns]]
    rename = {"y_pred": f"{model_name}_pred"}
    if "p_machine" in df.columns:
        rename["p_machine"] = f"{model_name}_p_machine"
    if "is_correct" in df.columns:
        rename["is_correct"] = f"{model_name}_stored_is_correct"
    if "predicted_label" in df.columns:
        rename["predicted_label"] = f"{model_name}_predicted_label"
    return df.rename(columns=rename)


def align_predictions(svm: pd.DataFrame, roberta: pd.DataFrame, split_name: str, out_dir: Path) -> pd.DataFrame:
    merged = svm.merge(
        roberta,
        on=KEY_COLS,
        how="outer",
        suffixes=("_svm", "_roberta"),
        indicator=True,
        validate="one_to_one",
    )
    unmatched = merged[merged["_merge"] != "both"]
    if not unmatched.empty:
        out_dir.mkdir(parents=True, exist_ok=True)
        unmatched.to_csv(out_dir / f"{split_name}_alignment_issues.csv", index=False, encoding="utf-8-sig")
        raise ValueError(
            f"{split_name}: SVM 与 RoBERTa 样本主键不完全一致；"
            f"仅一方存在的记录 {len(unmatched)} 条。详情已写入 "
            f"{out_dir / f'{split_name}_alignment_issues.csv'}"
        )
    merged = merged.drop(columns=["_merge"])

    # The ground-truth label and text must agree for the same validation sample.
    y_svm = merged["y_true_svm"] if "y_true_svm" in merged.columns else merged["y_true"]
    y_rob = merged["y_true_roberta"] if "y_true_roberta" in merged.columns else merged["y_true"]
    if not np.array_equal(y_svm.to_numpy(), y_rob.to_numpy()):
        mismatch = merged.loc[y_svm != y_rob, KEY_COLS].head(20)
        raise ValueError(
            f"{split_name}: 两个模型同一条样本的 y_true 不一致。示例："
            f"{mismatch.to_dict('records')}"
        )

    text_svm = merged["text_svm"] if "text_svm" in merged.columns else merged["text"]
    text_rob = merged["text_roberta"] if "text_roberta" in merged.columns else merged["text"]
    text_equal = text_svm.fillna("").astype(str).to_numpy() == text_rob.fillna("").astype(str).to_numpy()
    if not bool(np.all(text_equal)):
        mismatch = merged.loc[~text_equal, KEY_COLS].head(20)
        raise ValueError(
            f"{split_name}: 相同样本主键对应的 text 不一致。示例："
            f"{mismatch.to_dict('records')}"
        )

    # Normalize the merged schema independent of suffix behavior.
    if "y_true_svm" in merged.columns:
        merged["y_true"] = merged["y_true_svm"]
    if "text_svm" in merged.columns:
        merged["text"] = merged["text_svm"]
    merged["svm_correct"] = merged["svm_pred"].eq(merged["y_true"])
    merged["roberta_correct"] = merged["roberta_pred"].eq(merged["y_true"])
    merged["same_prediction"] = merged["svm_pred"].eq(merged["roberta_pred"])

    conditions = [
        merged["svm_correct"] & merged["roberta_correct"],
        merged["svm_correct"] & ~merged["roberta_correct"],
        ~merged["svm_correct"] & merged["roberta_correct"],
        ~merged["svm_correct"] & ~merged["roberta_correct"],
    ]
    labels = ["both_correct", "svm_only_correct", "roberta_only_correct", "both_wrong"]
    merged["comparison_group"] = np.select(conditions, labels, default="unknown")
    merged["svm_error_type"] = np.select(
        [merged["y_true"].eq(0) & merged["svm_pred"].eq(1),
         merged["y_true"].eq(1) & merged["svm_pred"].eq(0)],
        ["FP_human_as_machine", "FN_machine_as_human"], default="correct"
    )
    merged["roberta_error_type"] = np.select(
        [merged["y_true"].eq(0) & merged["roberta_pred"].eq(1),
         merged["y_true"].eq(1) & merged["roberta_pred"].eq(0)],
        ["FP_human_as_machine", "FN_machine_as_human"], default="correct"
    )
    return merged


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, split_name: str, model_name: str) -> Dict[str, object]:
    f1s = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "split": split_name,
        "model": model_name,
        "n_samples": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_machine": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_machine": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_human": float(f1s[0]),
        "f1_machine": float(f1s[1]),
        "tn_human_correct": int(tn),
        "fp_human_pred_machine": int(fp),
        "fn_machine_pred_human": int(fn),
        "tp_machine_correct": int(tp),
    }


def write_split_report(merged: pd.DataFrame, split_name: str, out_root: Path) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    out_dir = out_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = merged["y_true"].to_numpy(dtype=int)
    svm_pred = merged["svm_pred"].to_numpy(dtype=int)
    rob_pred = merged["roberta_pred"].to_numpy(dtype=int)

    rows = [
        metric_row(y_true, svm_pred, split_name, "TF-IDF+LinearSVC"),
        metric_row(y_true, rob_pred, split_name, "RoBERTa"),
    ]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics_comparison.csv", index=False, float_format="%.6f", encoding="utf-8-sig")

    group_counts = merged["comparison_group"].value_counts().to_dict()
    group_order = ["both_correct", "svm_only_correct", "roberta_only_correct", "both_wrong"]
    pd.DataFrame([
        {"comparison_group": group, "count": int(group_counts.get(group, 0)), "ratio": group_counts.get(group, 0) / len(merged)}
        for group in group_order
    ]).to_csv(out_dir / "comparison_groups.csv", index=False, float_format="%.6f", encoding="utf-8-sig")

    # Detailed, human-inspectable case files.
    export_cols = [c for c in [
        "article_id_svm", "article_key", "sentence_index", "domain_svm", "article_length_svm",
        "text", "y_true", "svm_pred", "roberta_pred", "svm_p_machine", "roberta_p_machine",
        "comparison_group", "svm_error_type", "roberta_error_type",
    ] if c in merged.columns]
    case_files = {
        "both_correct.csv": merged[merged["comparison_group"] == "both_correct"],
        "svm_only_correct.csv": merged[merged["comparison_group"] == "svm_only_correct"],
        "roberta_only_correct.csv": merged[merged["comparison_group"] == "roberta_only_correct"],
        "both_wrong.csv": merged[merged["comparison_group"] == "both_wrong"],
        "model_disagreements.csv": merged[~merged["same_prediction"]],
        "svm_false_positives.csv": merged[merged["svm_error_type"] == "FP_human_as_machine"],
        "svm_false_negatives.csv": merged[merged["svm_error_type"] == "FN_machine_as_human"],
        "roberta_false_positives.csv": merged[merged["roberta_error_type"] == "FP_human_as_machine"],
        "roberta_false_negatives.csv": merged[merged["roberta_error_type"] == "FN_machine_as_human"],
    }
    for filename, frame in case_files.items():
        frame[export_cols].to_csv(out_dir / filename, index=False, encoding="utf-8-sig")

    # Per-domain metrics help reveal whether gains are concentrated in a few domains.
    domain_col = "domain_svm" if "domain_svm" in merged.columns else ("domain" if "domain" in merged.columns else None)
    if domain_col:
        domain_rows: List[Dict[str, object]] = []
        for domain, group in merged.groupby(domain_col, dropna=False):
            yt = group["y_true"].to_numpy(dtype=int)
            for name, pred_col in [("TF-IDF+LinearSVC", "svm_pred"), ("RoBERTa", "roberta_pred")]:
                domain_rows.append(metric_row(yt, group[pred_col].to_numpy(dtype=int), split_name, name) | {
                    "domain": "<missing>" if pd.isna(domain) else str(domain)
                })
        pd.DataFrame(domain_rows).to_csv(out_dir / "metrics_by_domain.csv", index=False, float_format="%.6f", encoding="utf-8-sig")

    # Save all aligned cases for reproducibility.
    merged[export_cols].to_csv(out_dir / "aligned_predictions.csv", index=False, encoding="utf-8-sig")

    report = []
    report.append(f"{split_name.upper()} 模型比较")
    report.append("=" * 58)
    report.append(f"样本数：{len(merged):,}")
    report.append(f"标签分布：human={(y_true == 0).sum():,} | machine={(y_true == 1).sum():,}")
    report.append("")
    for row in rows:
        report.append(row["model"])
        for metric in ["accuracy", "precision_machine", "recall_machine", "macro_f1", "micro_f1", "f1_human", "f1_machine"]:
            report.append(f"  {metric:<20} {row[metric]:.4f}")
        report.append(
            "  confusion matrix: "
            f"TN={row['tn_human_correct']:,}, FP={row['fp_human_pred_machine']:,}, "
            f"FN={row['fn_machine_pred_human']:,}, TP={row['tp_machine_correct']:,}"
        )
        report.append("")
    report.append("模型互补性")
    for group in group_order:
        count = int(group_counts.get(group, 0))
        report.append(f"  {group:<22} {count:>7,} ({count / len(merged):.2%})")
    report.append(f"  两模型预测不一致：{int((~merged['same_prediction']).sum()):,} ({(~merged['same_prediction']).mean():.2%})")
    report.append("")
    report.append("说明：y_pred 来自各模型现有预测文件，使用模型各自已选定的阈值；本脚本不重新搜索阈值。")
    (out_dir / "summary.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    delta = rows[1]["macro_f1"] - rows[0]["macro_f1"]
    counts = {k: int(group_counts.get(k, 0)) for k in group_order}
    counts["disagreement"] = int((~merged["same_prediction"]).sum())
    counts["roberta_macro_f1_minus_svm"] = delta  # type: ignore[assignment]
    return rows, counts


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    out_root = resolve_path(root, args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    files = {
        "full_svm": resolve_path(root, args.full_svm),
        "full_roberta": resolve_path(root, args.full_roberta),
        "novel_svm": resolve_path(root, args.novel_svm),
        "novel_roberta": resolve_path(root, args.novel_roberta),
    }

    try:
        all_rows: List[Dict[str, object]] = []
        for split_name, svm_key, rob_key in [
            ("full_validation", "full_svm", "full_roberta"),
            ("novel_validation", "novel_svm", "novel_roberta"),
        ]:
            print(f"\n正在比较 {split_name} ...")
            svm_df = load_predictions(files[svm_key], "svm")
            rob_df = load_predictions(files[rob_key], "roberta")
            print(f"  SVM 样本：{len(svm_df):,} | RoBERTa 样本：{len(rob_df):,}")
            merged = align_predictions(svm_df, rob_df, split_name, out_root)
            print(f"  对齐成功：{len(merged):,} 条样本")
            rows, counts = write_split_report(merged, split_name, out_root)
            all_rows.extend(rows)
            print(
                f"  Macro-F1：SVM={rows[0]['macro_f1']:.4f} | "
                f"RoBERTa={rows[1]['macro_f1']:.4f} | "
                f"差值={rows[1]['macro_f1'] - rows[0]['macro_f1']:+.4f}"
            )
            print(
                "  仅 SVM 正确：{svm_only_correct:,} | 仅 RoBERTa 正确：{roberta_only_correct:,} | "
                "两者都错：{both_wrong:,} | 预测不一致：{disagreement:,}".format(**counts)
            )

        pd.DataFrame(all_rows).to_csv(
            out_root / "all_metrics_comparison.csv",
            index=False,
            float_format="%.6f",
            encoding="utf-8-sig",
        )
        print(f"\n比较完成。结果目录：{out_root}")
        print("请优先查看：")
        print(f"  {out_root / 'novel_validation' / 'summary.txt'}")
        print(f"  {out_root / 'novel_validation' / 'svm_only_correct.csv'}")
        print(f"  {out_root / 'novel_validation' / 'roberta_only_correct.csv'}")
        print(f"  {out_root / 'novel_validation' / 'both_wrong.csv'}")
        return 0
    except Exception as exc:
        print(f"比较失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
