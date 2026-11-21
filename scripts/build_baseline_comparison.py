#!/usr/bin/env python3
"""Build protocol-aware aggregate baseline comparison tables for AIGTD.

The source CSVs are aggregate metric files only. This script never reads
sentence-level predictions, raw datasets, article IDs, or model weights.
Metrics from different evaluation protocols are deliberately kept in separate
rows and are not ranked together.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


COMPARISON_COLUMNS = [
    "protocol_id",
    "dataset",
    "aggregation",
    "threshold_protocol",
    "model",
    "n",
    "accuracy",
    "macro_f1",
    "f1_human",
    "f1_machine",
    "precision_machine",
    "recall_machine",
    "roc_auc",
    "average_precision",
    "tn",
    "fp",
    "fn",
    "tp",
    "notes",
]


def _number(value: Any) -> float | int | None:
    if pd.isna(value):
        return None
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def _base_row(
    protocol_id: str,
    dataset: str,
    aggregation: str,
    threshold_protocol: str,
    model: str,
    n: Any,
    notes: str,
) -> dict[str, Any]:
    return {
        "protocol_id": protocol_id,
        "dataset": dataset,
        "aggregation": aggregation,
        "threshold_protocol": threshold_protocol,
        "model": model,
        "n": _number(n),
        "notes": notes,
    }


def _copy_metrics(row: dict[str, Any], source: pd.Series, mapping: dict[str, str]) -> None:
    for destination, source_name in mapping.items():
        if source_name in source.index:
            row[destination] = _number(source[source_name])


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def build_comparison(
    fusion_metrics: pd.DataFrame,
    context_overall: pd.DataFrame,
    context_novel: pd.DataFrame,
    svm_single: pd.DataFrame,
    bert_single: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        fusion_metrics,
        {"split", "model", "n", "accuracy", "macro_f1", "f1_human", "f1_machine",
         "precision_machine", "recall_machine", "roc_auc", "average_precision"},
        "OOF fusion metrics",
    )
    _require_columns(
        context_overall,
        {"model", "split", "n", "accuracy", "macro_f1", "f1_human", "f1_machine",
         "roc_auc", "average_precision"},
        "Context overall metrics",
    )
    _require_columns(
        context_novel,
        {"model", "split", "n", "accuracy", "macro_f1", "f1_human", "f1_machine",
         "roc_auc", "average_precision"},
        "Context Novel metrics",
    )
    for name, frame in (("SVM single-split metrics", svm_single), ("RoBERTa single-split metrics", bert_single)):
        _require_columns(
            frame,
            {"experiment", "threshold", "novel_validation_count", "novel_accuracy",
             "novel_macro_f1"},
            name,
        )

    output: list[dict[str, Any]] = []
    fusion_names = {
        "svm_oof_threshold": "SVM",
        "roberta_oof_threshold": "RoBERTa",
        "stacker_oof_crossfit": "Stacking",
        "veto_oof": "SVM Veto",
        "svm_fold_matched": "SVM",
        "roberta_fold_matched": "RoBERTa",
        "stacker_fold_matched": "Stacking",
        "svm_veto_fold_matched": "SVM Veto",
        "svm_5fold_mean_z_ranking_only": "SVM",
        "roberta_5fold_mean_z_ranking_only": "RoBERTa",
    }
    fusion_mapping = {
        "accuracy": "accuracy",
        "macro_f1": "macro_f1",
        "f1_human": "f1_human",
        "f1_machine": "f1_machine",
        "precision_machine": "precision_machine",
        "recall_machine": "recall_machine",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "tn": "tn",
        "fp": "fp",
        "fn": "fn",
        "tp": "tp",
    }
    for _, source in fusion_metrics.iterrows():
        split = str(source["split"])
        model_key = str(source["model"])
        if model_key not in fusion_names:
            continue
        if split == "oof_crossfit_diagnostic":
            protocol = "oof_crossfit"
            dataset = "Train OOF"
            threshold = "OOF-selected threshold; cross-fitted meta-model"
            aggregation = "Pooled OOF predictions; one OOF prediction per train row"
            notes = "Model-selection diagnostic; not independent test performance."
        elif split == "novel_validation_fold_matched":
            protocol = "novel_fold_matched"
            dataset = "Novel Validation (development)"
            threshold = "OOF-selected threshold and matching fold-specific model"
            aggregation = "One deterministic source-group-assigned fold per Novel row"
            notes = "Development validation; Novel has been used during model development."
        elif split == "novel_5fold_ensemble_ranking_only":
            protocol = "novel_5fold_ranking_only"
            dataset = "Novel Validation (development)"
            threshold = "Not applicable; ranking-only"
            aggregation = "Mean of five fold-standardized scores; no classification threshold"
            notes = "Ranking-only diagnostic; classification metrics intentionally blank."
        else:
            continue
        row = _base_row(
            protocol, dataset, aggregation, threshold,
            fusion_names[model_key], source["n"], notes,
        )
        _copy_metrics(row, source, fusion_mapping)
        output.append(row)

    context_mapping = {
        "accuracy": "accuracy",
        "macro_f1": "macro_f1",
        "f1_human": "f1_human",
        "f1_machine": "f1_machine",
        "precision_machine": "precision_machine",
        "recall_machine": "recall_machine",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }
    for _, source in context_overall.iterrows():
        model = str(source["model"])
        split = str(source["split"])
        if split == "pooled_oof_fixed_threshold_0.5":
            protocol = "context_pooled_oof_fixed_05"
            threshold = "Fixed probability threshold 0.5"
            aggregation = "Pooled OOF probabilities; threshold metrics and raw-score ranking metrics"
            notes = (
                "Raw pooled AUC/AP are retained as historical diagnostics; fold-specific score "
                "scales differ, so use standardized-margin AUC/AP below for cross-fold ranking."
            )
        elif split == "pooled_oof_fold_standardized_margin":
            protocol = "context_pooled_oof_standardized_ranking"
            threshold = "Not applicable; ranking-only standardized margins"
            aggregation = "Pooled OOF fold-standardized margins"
            notes = "Ranking-only diagnostic; classification metrics intentionally blank."
        else:
            continue
        row = _base_row(
            protocol, "Train OOF", aggregation, threshold,
            "Context" if model == "context" else "Sentence-only",
            source["n"], notes,
        )
        _copy_metrics(row, source, context_mapping)
        output.append(row)

    for _, source in context_novel.iterrows():
        row = _base_row(
            "context_novel_mean_probability",
            "Novel Validation (development)",
            "Mean probability over five fold models",
            "Fixed probability threshold 0.5",
            "Context" if str(source["model"]) == "context" else "Sentence-only",
            source["n"],
            "Development diagnostic; five-fold mean-probability ensemble, not fold-matched evaluation.",
        )
        _copy_metrics(row, source, context_mapping)
        output.append(row)

    for source_frame, model_name in ((svm_single, "SVM"), (bert_single, "RoBERTa")):
        for _, source in source_frame.iterrows():
            threshold = _number(source["threshold"])
            row = _base_row(
                "single_model_novel_threshold_selected",
                "Novel Validation (development)",
                "Single model trained on Train split; evaluated on Novel targets",
                f"Threshold {threshold}; selected on Novel Validation Macro-F1",
                model_name,
                source["novel_validation_count"],
                "Separate single-split protocol; threshold was selected on this development subset. "
                "Do not rank directly against OOF cross-fit or fold-matched fusion.",
            )
            row.update(
                {
                    "accuracy": _number(source.get("novel_accuracy")),
                    "macro_f1": _number(source.get("novel_macro_f1")),
                }
            )
            if "novel_precision" in source.index:
                row["precision_machine"] = _number(source["novel_precision"])
            if "novel_recall" in source.index:
                row["recall_machine"] = _number(source["novel_recall"])
            output.append(row)

    result = pd.DataFrame(output)
    if result.empty:
        raise RuntimeError("No baseline rows were assembled from the supplied aggregate files.")
    for column in COMPARISON_COLUMNS:
        if column not in result:
            result[column] = None
    return result[COMPARISON_COLUMNS].sort_values(
        ["protocol_id", "model"], kind="stable"
    ).reset_index(drop=True)


def build_context_fold_diagnostics(context_by_fold: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        context_by_fold,
        {"model", "fold", "n", "accuracy", "macro_f1", "roc_auc", "average_precision"},
        "Context fold metrics",
    )
    result = context_by_fold.copy()
    result["protocol_id"] = "context_matched_oof_per_fold"
    result["threshold_protocol"] = "Fixed probability threshold 0.5"
    result["notes"] = (
        "Strictly paired sentence_matched/context window-purged OOF fold; "
        "report every fold, including Fold 4 ranking-metric degradation."
    )
    result["model"] = result["model"].map(
        {"sentence_matched": "Sentence-only", "context": "Context"}
    ).fillna(result["model"].astype(str))
    columns = [
        "protocol_id", "fold", "model", "n", "accuracy", "macro_f1",
        "f1_human", "f1_machine", "roc_auc", "average_precision",
        "threshold_protocol", "notes",
    ]
    for column in columns:
        if column not in result:
            result[column] = None
    return result[columns].sort_values(["fold", "model"], kind="stable").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/analysis_review")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    required_paths = {
        "fusion": results_dir / "oof_fusion" / "metrics_comparison.csv",
        "context_overall": results_dir / "context_oof_analysis" / "metrics_overall.csv",
        "context_novel": results_dir / "context_oof_analysis" / "novel_metrics.csv",
        "context_by_fold": results_dir / "context_oof_analysis" / "metrics_by_fold.csv",
        "svm_single": results_dir / "svm_tfidf_metrics.csv",
        "bert_single": results_dir / "bert_metrics.csv",
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing aggregate metric files:\n- " + "\n- ".join(missing))

    frames = {
        key: pd.read_csv(path, encoding="utf-8-sig")
        for key, path in required_paths.items()
    }
    comparison = build_comparison(
        frames["fusion"],
        frames["context_overall"],
        frames["context_novel"],
        frames["svm_single"],
        frames["bert_single"],
    )
    context_folds = build_context_fold_diagnostics(frames["context_by_fold"])

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        output_dir / "baseline_comparison.csv", index=False, encoding="utf-8-sig"
    )
    context_folds.to_csv(
        output_dir / "context_fold_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Baseline rows: {len(comparison)}")
    print(f"Context fold rows: {len(context_folds)}")
    print(f"Aggregate tables written to: {output_dir.resolve()}")
    print("Different protocol_id values must not be ranked as one leaderboard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
