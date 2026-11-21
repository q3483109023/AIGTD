#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit article-level contributions to cross-Train/Validation label conflicts.

Read-only audit. Recomputes conflicts from the canonical split CSVs using
Unicode NFKC + casefold + collapsed whitespace (punctuation retained).
The aggregate JSON contains neither sentence text nor article identifiers.
Identifier-level CSVs are LOCAL ONLY and must not be committed.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", str(value)).casefold()
    ).strip()


def labels_to_ids(frame: pd.DataFrame, name: str) -> pd.Series:
    col = next((c for c in ("label_id", "label", "target")
                if c in frame.columns and frame[c].notna().all()), None)
    if col is None:
        raise ValueError(f"{name} needs a complete label_id, label, or target column.")

    def convert(value: Any) -> int:
        if isinstance(value, str):
            token = value.strip().casefold()
            if token == "human":
                return 0
            if token == "machine":
                return 1
            number = float(token)
        else:
            number = float(value)
        if not number.is_integer() or int(number) not in (0, 1):
            raise ValueError(f"Invalid label {value!r}; expected human/0 or machine/1.")
        return int(number)

    return frame[col].map(convert).astype("int8")


def entity_detail(rows: pd.DataFrame, col: str) -> pd.DataFrame:
    part = rows.loc[rows[col].notna()].copy()
    if part.empty:
        return pd.DataFrame(columns=[
            col, "conflict_groups", "conflict_rows", "train_rows", "validation_rows"
        ])
    part[col] = part[col].astype(str)
    result = part.groupby(col, dropna=True).agg(
        conflict_groups=("_group_id", "nunique"),
        conflict_rows=("_group_id", "size"),
        train_rows=("_split", lambda s: int((s == "train").sum())),
        validation_rows=("_split", lambda s: int((s == "validation").sum())),
    ).reset_index()
    return result.sort_values(
        ["conflict_groups", "conflict_rows", col],
        ascending=[False, False, True], kind="stable"
    ).reset_index(drop=True)


def histogram(values: pd.Series) -> dict[str, int]:
    return {str(int(k)): int(v) for k, v in values.value_counts().sort_index().items()}


def entity_summary(detail: pd.DataFrame, col: str) -> dict[str, Any]:
    if detail.empty:
        return {
            "entities_involved": 0, "max_conflict_groups_per_entity": 0,
            "groups_per_entity_histogram": {}, "top_1_group_share": 0.0,
            "top_5_groups_share": 0.0, "top_10_groups_share": 0.0,
        }
    counts = detail["conflict_groups"].sort_values(ascending=False).to_numpy()
    total = int(counts.sum())
    return {
        "entities_involved": int(len(detail)),
        "max_conflict_groups_per_entity": int(counts[0]),
        "groups_per_entity_histogram": histogram(detail["conflict_groups"]),
        "top_1_group_share": float(counts[:1].sum() / total),
        "top_5_groups_share": float(counts[:5].sum() / total),
        "top_10_groups_share": float(counts[:10].sum() / total),
    }


def analyze(train: pd.DataFrame, validation: pd.DataFrame):
    for name, frame in (("train", train), ("validation", validation)):
        required = {"text", "article_key"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    id_cols = [c for c in ("article_key", "article_id", "source_article_id")
               if c in train.columns and c in validation.columns]
    if "article_id" not in id_cols and "source_article_id" not in id_cols:
        raise ValueError("Both splits need article_id or source_article_id.")
    pieces = []
    for name, frame in (("train", train), ("validation", validation)):
        part = frame.copy()
        part["_split"] = name
        part["_label_id"] = labels_to_ids(frame, name).to_numpy()
        part["_norm_text"] = part["text"].map(normalize_text)
        if (part["_norm_text"] == "").any():
            raise ValueError(f"{name} contains empty normalized text.")
        pieces.append(part)
    combined = pd.concat(pieces, ignore_index=True, sort=False)

    records = []
    group_texts = []
    for norm_text, group in combined.groupby("_norm_text", sort=False):
        if group["_label_id"].nunique() < 2:
            continue
        if set(group["_split"]) != {"train", "validation"}:
            continue
        tr = group.loc[group["_split"] == "train"]
        va = group.loc[group["_split"] == "validation"]
        tr_labels = sorted(tr["_label_id"].unique().astype(int).tolist())
        va_labels = sorted(va["_label_id"].unique().astype(int).tolist())
        if len(tr_labels) == 1 and len(va_labels) == 1:
            category = "both_splits_consistent_but_disagree"
        elif len(tr_labels) > 1 and len(va_labels) == 1:
            category = "train_internal_conflict_only"
        elif len(tr_labels) == 1 and len(va_labels) > 1:
            category = "validation_internal_conflict_only"
        else:
            category = "both_splits_internal_conflict"
        record = {
            "_norm_text": norm_text,
            "train_rows": int(len(tr)),
            "validation_rows": int(len(va)),
            "all_rows": int(len(group)),
            "category": category,
        }
        for col in id_cols:
            record[f"distinct_{col}"] = int(group[col].dropna().astype(str).nunique())
        records.append(record)
        group_texts.append(norm_text)

    if not records:
        raise ValueError("No cross-split duplicate-text groups with conflicting labels found.")

    group_frame = pd.DataFrame(records)
    group_to_id = {value: i for i, value in enumerate(group_texts)}
    conflict_rows = combined.loc[combined["_norm_text"].isin(group_to_id)].copy()
    conflict_rows["_group_id"] = conflict_rows["_norm_text"].map(group_to_id)

    details = {col: entity_detail(conflict_rows, col) for col in id_cols}
    id_col = "article_id" if "article_id" in id_cols else "source_article_id"
    summary = {
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "cross_split_conflicting_text_groups": int(len(group_frame)),
        "cross_split_conflicting_rows": int(len(conflict_rows)),
        "train_rows_in_conflicting_groups": int((conflict_rows["_split"] == "train").sum()),
        "validation_rows_in_conflicting_groups": int((conflict_rows["_split"] == "validation").sum()),
        "distinct_article_keys_involved": int(conflict_rows["article_key"].dropna().astype(str).nunique()),
        "distinct_article_ids_involved": int(conflict_rows[id_col].dropna().astype(str).nunique()),
        "group_label_pattern_counts": {
            str(k): int(v) for k, v in group_frame["category"].value_counts().items()
        },
        "article_keys_per_conflict_group_histogram": histogram(group_frame["distinct_article_key"]),
        "article_ids_per_conflict_group_histogram": histogram(group_frame[f"distinct_{id_col}"]),
        "groups_spanning_multiple_article_keys": int((group_frame["distinct_article_key"] > 1).sum()),
        "groups_spanning_multiple_article_ids": int((group_frame[f"distinct_{id_col}"] > 1).sum()),
        "per_entity_concentration": {
            col: entity_summary(detail, col) for col, detail in details.items()
        },
        "normalization": "Unicode NFKC + casefold + collapsed whitespace; punctuation retained",
        "label_mapping": {"human": 0, "machine": 1},
        "privacy": {
            "aggregate_contains_sentence_text": False,
            "aggregate_contains_article_identifiers": False,
            "identifier_detail_csvs_are_local_only": True,
            "source_csvs_modified": False,
        },
    }
    return summary, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/splits/train.csv")
    parser.add_argument("--validation", default="data/splits/validation.csv")
    parser.add_argument("--aggregate-output",
                        default="results/analysis_review/article_conflict_distribution_aggregate.json")
    parser.add_argument("--detail-dir",
                        default="results/duplicate_data_audit/article_distribution_local_only")
    parser.add_argument("--expected-groups", type=int, default=38)
    args = parser.parse_args()

    train_path, val_path = Path(args.train), Path(args.validation)
    for path in (train_path, val_path):
        if not path.exists():
            raise FileNotFoundError(path)
    train = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False)
    validation = pd.read_csv(val_path, encoding="utf-8-sig", low_memory=False)
    summary, details = analyze(train, validation)
    actual = summary["cross_split_conflicting_text_groups"]
    summary["handover_count_check"] = {
        "expected": args.expected_groups, "actual": actual, "matches": actual == args.expected_groups
    }

    aggregate_path = Path(args.aggregate_output)
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    detail_dir = Path(args.detail_dir)
    detail_dir.mkdir(parents=True, exist_ok=True)
    for col, frame in details.items():
        frame.to_csv(detail_dir / f"{col}_conflict_distribution_LOCAL_ONLY.csv", index=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nText-free aggregate: {aggregate_path.resolve()}")
    print(f"LOCAL ONLY identifier details: {detail_dir.resolve()}")
    if actual != args.expected_groups:
        print("WARNING: conflict-group count differs from the handover; verify split versions and normalization.")


if __name__ == "__main__":
    main()
