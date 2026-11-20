"""FP/FN analysis with separate Full and Novel Validation views."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

try:
    from scripts.common import cfg_path, load_config, resolve_path
    from scripts.evaluate import calculate_metrics
except ImportError:
    from common import cfg_path, load_config, resolve_path
    from evaluate import calculate_metrics


def enrich_errors(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["y_true"] = result["y_true"].astype(int)
    result["y_pred"] = result["y_pred"].astype(int)
    result["p_machine"] = pd.to_numeric(result["p_machine"], errors="coerce")
    if result["p_machine"].isna().any():
        raise ValueError("p_machine 中存在非数值或空值。")
    result["is_correct"] = result["y_true"] == result["y_pred"]
    result["error_type"] = np.select(
        [
            (result.y_true == 1) & (result.y_pred == 1),
            (result.y_true == 0) & (result.y_pred == 0),
            (result.y_true == 0) & (result.y_pred == 1),
            (result.y_true == 1) & (result.y_pred == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="UNKNOWN",
    )
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]
    result["probability_band"] = pd.cut(
        result.p_machine.clip(0, 1), bins=bins, labels=labels,
        include_lowest=True, right=True,
    )
    result["word_count"] = result.text.fillna("").astype(str).str.split().str.len()
    result["length_band"] = pd.cut(
        result.word_count,
        bins=[-1, 5, 10, 20, 35, 50, 100, np.inf],
        labels=["0-5", "6-10", "11-20", "21-35", "36-50", "51-100", "100+"],
    )
    return result


def summarize_subset(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty:
        return {"count": 0, "metrics": None, "TP": 0, "TN": 0, "FP": 0, "FN": 0}
    return {
        "count": int(len(df)),
        "metrics": calculate_metrics(df.y_true, df.y_pred),
        "TP": int((df.error_type == "TP").sum()),
        "TN": int((df.error_type == "TN").sum()),
        "FP": int((df.error_type == "FP").sum()),
        "FN": int((df.error_type == "FN").sum()),
        "error_count": int((~df.is_correct).sum()),
    }


def save_error_breakdowns(df: pd.DataFrame, results_dir: Path, suffix: str = "") -> None:
    group_cols = [col for col in ["domain", "length_band", "probability_band"] if col in df.columns]
    for col in group_cols:
        grouped = df.groupby(col, dropna=False, observed=False).agg(
            samples=("y_true", "size"),
            accuracy=("is_correct", "mean"),
            fp=("error_type", lambda values: int((values == "FP").sum())),
            fn=("error_type", lambda values: int((values == "FN").sum())),
        ).reset_index()
        grouped.to_csv(
            results_dir / f"error_by_{col}{suffix}.csv", index=False, encoding="utf-8-sig"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze FP/FN separately for Full and Novel Validation.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--predictions", default="results/predictions.csv")
    parser.add_argument("--output", default=None, help="Optional Novel Validation FP/FN CSV path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pred_path = resolve_path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(f"预测文件不存在：{pred_path}。请先运行训练脚本。")
    df = pd.read_csv(pred_path, encoding="utf-8-sig")
    required = {"y_true", "y_pred", "p_machine", "text", "is_novel_validation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"预测 CSV 缺少必要字段：{sorted(missing)}。请先使用更新后的训练/评估脚本重新生成预测文件。"
        )

    df = enrich_errors(df)
    # pandas reads this field as bool for the generated CSV; support 0/1 too.
    if df["is_novel_validation"].dtype != bool:
        value_map = {"true": True, "false": False, "1": True, "0": False}
        df["is_novel_validation"] = df["is_novel_validation"].astype(str).str.lower().map(value_map)
        if df["is_novel_validation"].isna().any():
            raise ValueError("is_novel_validation 必须只包含 True/False 或 1/0。")

    novel_df = df[df.is_novel_validation].copy()
    seen_df = df[~df.is_novel_validation].copy()
    if novel_df.empty:
        raise ValueError("预测文件中的 Novel Validation 为空。")

    results_dir = cfg_path(cfg, "results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    novel_output = resolve_path(args.output) if args.output else results_dir / "error_analysis.csv"
    novel_df[novel_df.error_type.isin(["FP", "FN"])].sort_values(
        ["error_type", "p_machine"], ascending=[True, False]
    ).to_csv(novel_output, index=False, encoding="utf-8-sig")
    df[df.error_type.isin(["FP", "FN"])].sort_values(
        ["error_type", "p_machine"], ascending=[True, False]
    ).to_csv(results_dir / "error_analysis_full.csv", index=False, encoding="utf-8-sig")
    seen_df[seen_df.error_type.isin(["FP", "FN"])].sort_values(
        ["error_type", "p_machine"], ascending=[True, False]
    ).to_csv(results_dir / "error_analysis_seen_in_train.csv", index=False, encoding="utf-8-sig")

    summary = {
        "primary_view": "novel_validation",
        "full_validation": summarize_subset(df),
        "novel_validation": summarize_subset(novel_df),
        "seen_in_train_validation": summarize_subset(seen_df),
    }
    (results_dir / "error_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    save_error_breakdowns(novel_df, results_dir, suffix="_novel")
    save_error_breakdowns(df, results_dir, suffix="_full")

    print("错误分析完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Novel Validation FP/FN 明细：{novel_output}")
    print(f"Full Validation FP/FN 明细：{results_dir / 'error_analysis_full.csv'}")


if __name__ == "__main__":
    main()
