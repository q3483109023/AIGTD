#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose why context-window OOF performance changes on Fold 4.

The script reads the paired prediction exports. It does not retrain models,
change thresholds, or infer causality. Class 0 is Human and class 1 is Machine.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def classification_metrics(frame: pd.DataFrame, pred_col: str, prefix: str) -> dict:
    y = frame["y_true"].astype(int).to_numpy()
    p = frame[pred_col].astype(int).to_numpy()
    if len(y) == 0:
        return {f"{prefix}_n": 0}
    cm = confusion_matrix(y, p, labels=[0, 1])
    return {
        f"{prefix}_n": int(len(frame)),
        f"{prefix}_accuracy": float(accuracy_score(y, p)),
        f"{prefix}_macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        f"{prefix}_human_f1": float(f1_score(y, p, pos_label=0, zero_division=0)),
        f"{prefix}_machine_f1": float(f1_score(y, p, pos_label=1, zero_division=0)),
        f"{prefix}_human_as_machine": int(cm[0, 1]),
        f"{prefix}_machine_as_human": int(cm[1, 0]),
    }


def safe_qcut(values: pd.Series, q: int, prefix: str) -> pd.Series:
    try:
        result = pd.qcut(values, q=q, duplicates="drop")
        return result.astype(str).replace("nan", "missing")
    except ValueError:
        return pd.Series([f"{prefix}_all"] * len(values), index=values.index)


def add_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "text" in out:
        out["target_chars"] = out["text"].fillna("").astype(str).str.len()
        out["target_words"] = out["text"].fillna("").astype(str).str.split().str.len()
    if "context_text" in out:
        contexts = out["context_text"].fillna("").astype(str)
        prev, nxt = [], []
        for value in contexts:
            m = re.search(r"\[PREVIOUS\]\s*(.*?)\s*\[TARGET\]", value, flags=re.S)
            n = re.search(r"\[NEXT\]\s*(.*)$", value, flags=re.S)
            prev_value = m.group(1).strip() if m else ""
            next_value = n.group(1).strip() if n else ""
            prev.append("" if prev_value == "[START]" else prev_value)
            nxt.append("" if next_value == "[END]" else next_value)
        out["previous_chars"] = pd.Series(prev, index=out.index).str.len()
        out["next_chars"] = pd.Series(nxt, index=out.index).str.len()
        out["has_previous"] = out["previous_chars"] > 0
        out["has_next"] = out["next_chars"] > 0
        out["context_chars"] = contexts.str.len()
    return out


def load_predictions(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = {
        "sentence": root / "sentence_matched_oof_predictions.csv",
        "context": root / "context_oof_predictions.csv",
    }
    a, b = (pd.read_csv(paths[k], low_memory=False) for k in ("sentence", "context"))
    required = {"oof_row_id", "fold", "y_true", "context_p_machine"}
    for name, df in (("sentence", a), ("context", b)):
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{name} prediction file missing columns: {missing}")
        if df["oof_row_id"].isna().any() or df["oof_row_id"].duplicated().any():
            raise ValueError(f"{name} oof_row_id must be non-null and unique")
        df["context_p_machine"] = pd.to_numeric(df["context_p_machine"], errors="coerce")
        if not np.isfinite(df["context_p_machine"].to_numpy(float)).all():
            raise ValueError(f"{name} probabilities contain non-finite values")
        if not df["context_p_machine"].between(0, 1).all():
            raise ValueError(f"{name} probabilities must be in [0, 1]")
    a = a.sort_values("oof_row_id").reset_index(drop=True)
    b = b.sort_values("oof_row_id").reset_index(drop=True)
    if not np.array_equal(a["oof_row_id"], b["oof_row_id"]):
        raise ValueError("Sentence and Context OOF row IDs do not match")
    for col in ("fold", "y_true"):
        if not np.array_equal(a[col], b[col]):
            raise ValueError(f"Sentence and Context disagree on {col}")
    return a, b


def paired_fold_frame(a: pd.DataFrame, b: pd.DataFrame, fold: int) -> pd.DataFrame:
    left = a.loc[a["fold"].astype(int).eq(fold)].copy()
    right = b.loc[b["fold"].astype(int).eq(fold)].copy()
    if len(left) != len(right) or not np.array_equal(left["oof_row_id"], right["oof_row_id"]):
        raise ValueError(f"Fold {fold} paired rows mismatch")
    meta_cols = [
        c for c in ("article_key", "article_id", "source_article_id",
                    "sentence_index", "text", "context_text", "label_id")
        if c in left.columns and c in right.columns
    ]
    for col in meta_cols:
        if not left[col].fillna("<NA>").astype(str).equals(right[col].fillna("<NA>").astype(str)):
            raise ValueError(f"Fold {fold} metadata differs between models: {col}")
    out = left.copy()
    out = out.rename(columns={"context_p_machine": "sentence_p_machine"})
    out["context_p_machine"] = right["context_p_machine"].to_numpy(float)
    out["sentence_pred"] = (out["sentence_p_machine"] >= 0.5).astype(int)
    out["context_pred"] = (out["context_p_machine"] >= 0.5).astype(int)
    out["sentence_correct"] = out["sentence_pred"].eq(out["y_true"].astype(int))
    out["context_correct"] = out["context_pred"].eq(out["y_true"].astype(int))
    out["outcome"] = np.select(
        [
            out["sentence_correct"] & out["context_correct"],
            ~out["sentence_correct"] & ~out["context_correct"],
            ~out["sentence_correct"] & out["context_correct"],
            out["sentence_correct"] & ~out["context_correct"],
        ],
        ["both_correct", "both_wrong", "context_only_correct", "sentence_only_correct"],
        default="unknown",
    )
    out["sentence_confidence"] = np.maximum(out["sentence_p_machine"], 1 - out["sentence_p_machine"])
    out["context_confidence"] = np.maximum(out["context_p_machine"], 1 - out["context_p_machine"])
    out["confidence_change"] = out["context_confidence"] - out["sentence_confidence"]
    out["probability_change"] = out["context_p_machine"] - out["sentence_p_machine"]
    return add_context_features(out)


def summarize(frame: pd.DataFrame, out: Path, fold: int) -> dict:
    summary = {
        "fold": fold,
        "rows": int(len(frame)),
        **classification_metrics(frame, "sentence_pred", "sentence"),
        **classification_metrics(frame, "context_pred", "context"),
    }
    summary["macro_f1_delta_context_minus_sentence"] = (
        summary["context_macro_f1"] - summary["sentence_macro_f1"]
    )
    summary["accuracy_delta_context_minus_sentence"] = (
        summary["context_accuracy"] - summary["sentence_accuracy"]
    )
    outcome_counts = frame["outcome"].value_counts().to_dict()
    summary["outcome_counts"] = {str(k): int(v) for k, v in outcome_counts.items()}
    summary["outcome_rates"] = {str(k): float(v / len(frame)) for k, v in outcome_counts.items()}
    (out / f"fold_{fold}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def grouped_report(frame: pd.DataFrame, group_col: str, min_rows: int) -> pd.DataFrame:
    if group_col not in frame.columns:
        return pd.DataFrame()
    rows = []
    for key, g in frame.groupby(group_col, dropna=False):
        if len(g) < min_rows:
            continue
        row = {"group_by": group_col, "group": str(key), "rows": int(len(g))}
        row.update(classification_metrics(g, "sentence_pred", "sentence"))
        row.update(classification_metrics(g, "context_pred", "context"))
        row["macro_f1_delta"] = row["context_macro_f1"] - row["sentence_macro_f1"]
        row["accuracy_delta"] = row["context_accuracy"] - row["sentence_accuracy"]
        row["context_only_correct"] = int(g["outcome"].eq("context_only_correct").sum())
        row["sentence_only_correct"] = int(g["outcome"].eq("sentence_only_correct").sum())
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["macro_f1_delta", "rows"], ascending=[True, False])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="results/context_oof")
    p.add_argument("--output-dir", default="results/context_oof_fold4_diagnosis")
    p.add_argument("--fold", type=int, default=4)
    p.add_argument("--min-group-rows", type=int, default=5)
    p.add_argument("--confidence-examples", type=int, default=500)
    args = p.parse_args()
    root, out = Path(args.input_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    a, b = load_predictions(root)
    frame = paired_fold_frame(a, b, args.fold)
    if frame.empty:
        raise ValueError(f"No OOF rows found for fold {args.fold}")

    summary = summarize(frame, out, args.fold)
    pd.DataFrame([summary]).drop(columns=["outcome_counts", "outcome_rates"]).to_csv(
        out / f"fold_{args.fold}_metrics.csv", index=False
    )
    frame["true_class"] = frame["y_true"].map({0: "Human", 1: "Machine"}).fillna("unknown")
    frame["target_length_bin"] = safe_qcut(
        frame["target_chars"] if "target_chars" in frame else pd.Series(0, index=frame.index),
        4, "length"
    )
    frame["sentence_confidence_bin"] = pd.cut(
        frame["sentence_confidence"], bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.000001],
        labels=["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"],
        include_lowest=True
    ).astype(str)
    frame["context_confidence_bin"] = pd.cut(
        frame["context_confidence"], bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.000001],
        labels=["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"],
        include_lowest=True
    ).astype(str)

    outcome = frame["outcome"].value_counts().rename_axis("outcome").reset_index(name="rows")
    outcome["rate"] = outcome["rows"] / len(frame)
    outcome.to_csv(out / f"fold_{args.fold}_outcome_counts.csv", index=False)

    class_rows = []
    for label, g in frame.groupby("true_class", sort=True):
        row = {"true_class": label, "rows": int(len(g))}
        row["sentence_class_recall"] = float(g["sentence_correct"].mean())
        row["context_class_recall"] = float(g["context_correct"].mean())
        row["class_recall_delta"] = row["context_class_recall"] - row["sentence_class_recall"]
        row["context_only_correct"] = int(g["outcome"].eq("context_only_correct").sum())
        row["sentence_only_correct"] = int(g["outcome"].eq("sentence_only_correct").sum())
        class_rows.append(row)
    pd.DataFrame(class_rows).to_csv(out / f"fold_{args.fold}_by_true_class.csv", index=False)

    subgroup_specs = [
        ("target_length_bin", "target_length_bin"),
        ("sentence_confidence_bin", "sentence_confidence_bin"),
        ("context_confidence_bin", "context_confidence_bin"),
        ("has_previous", "has_previous"),
        ("has_next", "has_next"),
    ]
    subgroup_frames = []
    for name, col in subgroup_specs:
        if col not in frame.columns:
            continue
        gdf = grouped_report(frame, col, min_rows=1)
        if not gdf.empty:
            gdf.insert(0, "analysis", name)
            subgroup_frames.append(gdf)
    if subgroup_frames:
        pd.concat(subgroup_frames, ignore_index=True).to_csv(
            out / f"fold_{args.fold}_subgroups.csv", index=False
        )

    for col in ("article_key", "article_id", "source_article_id"):
        report = grouped_report(frame, col, args.min_group_rows)
        if not report.empty:
            report.to_csv(out / f"fold_{args.fold}_by_{col}.csv", index=False)

    # Keep every disagreement, with enough source text/context to inspect manually.
    disagreements = frame.loc[frame["outcome"].isin(
        ["context_only_correct", "sentence_only_correct", "both_wrong"]
    )].copy()
    disagreements["diagnostic_priority"] = np.where(
        disagreements["outcome"].eq("sentence_only_correct"), 0,
        np.where(disagreements["outcome"].eq("context_only_correct"), 1, 2)
    )
    disagreements = disagreements.sort_values(
        ["diagnostic_priority", "confidence_change"], ascending=[True, False]
    )
    preferred = [
        "oof_row_id", "fold", "article_key", "article_id", "source_article_id",
        "sentence_index", "true_class", "y_true", "outcome",
        "sentence_p_machine", "sentence_pred", "sentence_correct",
        "context_p_machine", "context_pred", "context_correct",
        "probability_change", "sentence_confidence", "context_confidence",
        "confidence_change", "target_chars", "target_words", "previous_chars",
        "next_chars", "text", "context_text",
    ]
    cols = [c for c in preferred if c in disagreements.columns]
    disagreements[cols].to_csv(out / f"fold_{args.fold}_disagreements.csv", index=False)

    # Compact, high-confidence cases where context flips a correct prediction to wrong.
    harmful = frame.loc[frame["outcome"].eq("sentence_only_correct")].copy()
    harmful = harmful.sort_values("confidence_change", ascending=False).head(args.confidence_examples)
    helpful = frame.loc[frame["outcome"].eq("context_only_correct")].copy()
    helpful = helpful.sort_values("confidence_change", ascending=False).head(args.confidence_examples)
    for name, df in (("context_harmed", harmful), ("context_helped", helpful)):
        keep = [c for c in preferred if c in df.columns]
        df[keep].to_csv(out / f"fold_{args.fold}_{name}_examples.csv", index=False)

    print(f"Fold {args.fold} diagnosis complete: {len(frame):,} paired rows")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs: {out.resolve()}")


if __name__ == "__main__":
    main()
