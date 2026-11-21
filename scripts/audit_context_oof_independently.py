#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independently audit materialized Context OOF folds and Novel mapping.

Rebuilds context windows and expected purged rows from canonical split CSVs
and the row-fold manifest. It deliberately does not import the preparation
script. Reports counts/checks only: no sentence text or article identifiers.
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


def normalize(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", str(value)).casefold()
    ).strip()


def require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def build_windows(frame: pd.DataFrame, window: int, name: str) -> pd.DataFrame:
    """Independent reconstruction matching the documented window=1 protocol."""
    require_columns(frame, {"article_key", "sentence_index", "text"}, name)
    if window != 1:
        raise ValueError("This auditor currently supports context_window=1 only.")
    out = frame.copy().reset_index(drop=True)
    out["_norm_target"] = out["text"].map(normalize)
    if (out["_norm_target"] == "").any():
        raise ValueError(f"{name} contains empty normalized text.")
    if out.duplicated(["article_key", "sentence_index"]).any():
        raise ValueError(f"{name} contains duplicate article_key/sentence_index rows.")

    lookup: dict[tuple[str, int], tuple[str, tuple[str, ...]]] = {}
    for article_key, group in out.groupby("article_key", sort=False):
        group = group.sort_values("sentence_index", kind="stable")
        records = list(
            group[["sentence_index", "text", "_norm_target"]]
            .itertuples(index=False, name=None)
        )
        by_index = {int(item[0]): item for item in records}
        for index, _target_raw, target_norm in records:
            left = [by_index[j] for j in (int(index) - 1,) if j in by_index]
            right = [by_index[j] for j in (int(index) + 1,) if j in by_index]
            left_text = str(left[0][1]) if left else "[START]"
            right_text = str(right[0][1]) if right else "[END]"
            parts = tuple(
                part for part in (
                    normalize(left_text) if left else "",
                    target_norm,
                    normalize(right_text) if right else "",
                ) if part
            )
            serialized = (
                f"[PREVIOUS] {left_text} [TARGET] {target_norm} [NEXT] {right_text}"
            )
            lookup[(str(article_key), int(index))] = (serialized, parts)

    keys = [(str(r.article_key), int(r.sentence_index)) for r in out.itertuples()]
    out["context_text"] = [lookup[key][0] for key in keys]
    out["_window_components"] = [lookup[key][1] for key in keys]
    return out


def union_components(frame: pd.DataFrame) -> set[str]:
    result: set[str] = set()
    for values in frame["_window_components"]:
        result.update(values)
    return result


def compare_exports(expected: pd.DataFrame, actual: pd.DataFrame, key: str,
                    fields: list[str], name: str) -> dict[str, Any]:
    require_columns(expected, {key, *fields}, f"{name} expected")
    require_columns(actual, {key, *fields}, f"{name} actual")
    if expected[key].duplicated().any() or actual[key].duplicated().any():
        raise ValueError(f"{name} has duplicate {key}.")
    left = expected.set_index(key, drop=False)
    right = actual.set_index(key, drop=False)
    left_keys, right_keys = set(left.index), set(right.index)
    common = sorted(left_keys & right_keys)
    mismatches = {}
    for field in fields:
        a = left.loc[common, field].fillna("<NA>").astype(str)
        b = right.loc[common, field].fillna("<NA>").astype(str)
        mismatches[field] = int((a.to_numpy() != b.to_numpy()).sum())
    return {
        "expected_rows": int(len(expected)),
        "actual_rows": int(len(actual)),
        "missing_rows": int(len(left_keys - right_keys)),
        "extra_rows": int(len(right_keys - left_keys)),
        "field_mismatches": mismatches,
        "pass": (
            left_keys == right_keys
            and not any(mismatches.values())
        ),
    }


def string_set(frame: pd.DataFrame, col: str) -> set[str]:
    return set(frame[col].dropna().astype(str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/splits/train.csv")
    parser.add_argument("--validation", default="data/splits/validation.csv")
    parser.add_argument("--row-manifest", default="results/oof_purged_group_audit/row_fold_manifest.csv")
    parser.add_argument("--folds-dir", default="data/context_oof_folds")
    parser.add_argument("--output-dir", default="results/analysis_review")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--context-window", type=int, default=1)
    parser.add_argument("--expected-novel-rows", type=int, default=11682)
    args = parser.parse_args()

    if args.context_window != 1:
        raise ValueError("Only context_window=1 is currently supported.")
    train_path, val_path, manifest_path = map(
        Path, (args.train, args.validation, args.row_manifest)
    )
    for path in (train_path, val_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)
    folds_dir = Path(args.folds_dir)
    if not folds_dir.exists():
        raise FileNotFoundError(folds_dir)

    train = pd.read_csv(train_path, encoding="utf-8-sig", low_memory=False).reset_index(drop=True)
    validation = pd.read_csv(val_path, encoding="utf-8-sig", low_memory=False).reset_index(drop=True)
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", low_memory=False)
    require_columns(train, {"article_key", "article_id", "sentence_index", "text", "label_id"}, "train")
    require_columns(validation, {"article_key", "sentence_index", "text", "label_id"}, "validation")
    require_columns(manifest, {"row_id", "article_key", "sentence_index", "source_article_id", "fold"}, "manifest")

    if len(manifest) != len(train):
        raise ValueError("Manifest and train row counts differ.")
    manifest = manifest.sort_values("row_id", kind="stable").reset_index(drop=True)
    row_ids = pd.to_numeric(manifest["row_id"]).astype(int).to_numpy()
    if not np.array_equal(row_ids, np.arange(len(train))):
        raise ValueError("Manifest row_id must cover 0..N-1 exactly.")
    for col in ("article_key", "sentence_index"):
        if not np.array_equal(manifest[col].astype(str).to_numpy(), train[col].astype(str).to_numpy()):
            raise ValueError(f"Manifest/train order mismatch in {col}.")
    if not np.array_equal(manifest["source_article_id"].astype(str).to_numpy(),
                          train["article_id"].astype(str).to_numpy()):
        raise ValueError("Manifest source IDs do not match train.article_id.")

    folds = sorted(pd.to_numeric(manifest["fold"]).astype(int).unique().tolist())
    if folds != list(range(args.n_splits)):
        raise ValueError(f"Expected folds 0..{args.n_splits - 1}; got {folds}.")

    train = train.copy()
    train["oof_row_id"] = np.arange(len(train), dtype=int)
    train["source_article_id"] = manifest["source_article_id"].astype(str).to_numpy()
    train["fold"] = pd.to_numeric(manifest["fold"]).astype(int).to_numpy()
    train_ctx = build_windows(train, args.context_window, "train")
    val_ctx = build_windows(validation, args.context_window, "validation")

    train_targets = set(train_ctx["_norm_target"])
    expected_novel = val_ctx.loc[~val_ctx["_norm_target"].isin(train_targets)].copy()
    expected_novel["novel_row_id"] = np.arange(len(expected_novel), dtype=int)
    novel_file = folds_dir / "novel_validation.csv"
    if not novel_file.exists():
        raise FileNotFoundError(novel_file)
    actual_novel = pd.read_csv(novel_file, encoding="utf-8-sig", low_memory=False)

    novel_fields = [
        field for field in (
            "article_key", "sentence_index", "text", "label_id", "article_id", "context_text"
        ) if field in expected_novel.columns and field in actual_novel.columns
    ]
    novel_check = compare_exports(
        expected_novel, actual_novel, "novel_row_id", novel_fields, "Novel Validation"
    )
    novel_ids_valid = (
        "novel_row_id" in actual_novel.columns
        and np.array_equal(
            pd.to_numeric(actual_novel["novel_row_id"]).astype(int).to_numpy(),
            np.arange(len(actual_novel)),
        )
    )
    novel_check["novel_row_id_is_contiguous_in_export_order"] = bool(novel_ids_valid)
    novel_check["pass"] = bool(novel_check["pass"] and novel_ids_valid)
    expected_novel_components = union_components(expected_novel)

    fold_reports = []
    all_oof_val_ids = []
    for fold in folds:
        fold_dir = folds_dir / f"fold_{fold}"
        train_file, val_file = fold_dir / "train.csv", fold_dir / "validation.csv"
        if not train_file.exists() or not val_file.exists():
            raise FileNotFoundError(f"Fold {fold} is missing train.csv or validation.csv.")
        actual_train = pd.read_csv(train_file, encoding="utf-8-sig", low_memory=False)
        actual_val = pd.read_csv(val_file, encoding="utf-8-sig", low_memory=False)

        expected_val = train_ctx.loc[train_ctx["fold"].eq(fold)].copy()
        candidates = train_ctx.loc[~train_ctx["fold"].eq(fold)].copy()
        protected = union_components(expected_val) | expected_novel_components
        keep = ~candidates["_window_components"].map(
            lambda parts: any(part in protected for part in parts)
        )
        expected_train = candidates.loc[keep].copy()

        fields = [
            c for c in (
                "fold", "article_key", "source_article_id", "sentence_index",
                "text", "label_id", "context_text"
            ) if c in expected_train.columns and c in actual_train.columns
        ]
        train_check = compare_exports(
            expected_train, actual_train, "oof_row_id", fields, f"Fold {fold} train"
        )
        val_fields = [
            c for c in (
                "fold", "article_key", "source_article_id", "sentence_index",
                "text", "label_id", "context_text"
            ) if c in expected_val.columns and c in actual_val.columns
        ]
        val_check = compare_exports(
            expected_val, actual_val, "oof_row_id", val_fields, f"Fold {fold} validation"
        )

        all_oof_val_ids.extend(pd.to_numeric(actual_val["oof_row_id"]).astype(int).tolist())
        # These component unions are rebuilt from canonical input rows, while
        # export equality above ensures the saved rows/windows match them.
        actual_train_by_id = expected_train.loc[
            expected_train["oof_row_id"].isin(set(pd.to_numeric(actual_train["oof_row_id"]).astype(int)))
        ]
        actual_val_by_id = expected_val.loc[
            expected_val["oof_row_id"].isin(set(pd.to_numeric(actual_val["oof_row_id"]).astype(int)))
        ]
        train_components = union_components(actual_train_by_id)
        val_components = union_components(actual_val_by_id)
        val_overlap = train_components & val_components
        novel_overlap = train_components & expected_novel_components
        source_overlap = string_set(actual_train, "source_article_id") & string_set(actual_val, "source_article_id")
        article_overlap = string_set(actual_train, "article_key") & string_set(actual_val, "article_key")
        train_labels = pd.to_numeric(actual_train["label_id"]).astype(int)
        val_fold_values = sorted(pd.to_numeric(actual_val["fold"]).astype(int).unique().tolist())

        fold_pass = bool(
            train_check["pass"] and val_check["pass"]
            and not val_overlap and not novel_overlap
            and not source_overlap and not article_overlap
            and set(train_labels.unique()) == {0, 1}
            and val_fold_values == [fold]
        )
        fold_reports.append({
            "fold": int(fold),
            "candidate_train_rows": int(len(candidates)),
            "expected_purged_rows": int(len(candidates) - len(expected_train)),
            "expected_train_rows": int(len(expected_train)),
            "actual_train_rows": int(len(actual_train)),
            "validation_rows": int(len(actual_val)),
            "train_export_exact_match": train_check,
            "validation_export_exact_match": val_check,
            "train_validation_component_overlap": int(len(val_overlap)),
            "train_novel_window_component_overlap": int(len(novel_overlap)),
            "source_id_overlap": int(len(source_overlap)),
            "article_key_overlap": int(len(article_overlap)),
            "train_human_rows": int((train_labels == 0).sum()),
            "train_machine_rows": int((train_labels == 1).sum()),
            "validation_fold_values_match": val_fold_values == [fold],
            "pass": fold_pass,
        })

    oof_coverage = (
        len(all_oof_val_ids) == len(train)
        and len(set(all_oof_val_ids)) == len(train)
        and set(all_oof_val_ids) == set(range(len(train)))
    )
    overall_pass = bool(
        novel_check["pass"] and oof_coverage and all(row["pass"] for row in fold_reports)
    )
    report = {
        "audit": "independent_context_oof_fold_and_novel_mapping_audit",
        "context_window": args.context_window,
        "n_splits": args.n_splits,
        "canonical_train_rows": int(len(train)),
        "canonical_validation_rows": int(len(validation)),
        "recomputed_novel_rows": int(len(expected_novel)),
        "exported_novel_rows": int(len(actual_novel)),
        "expected_novel_rows_from_handover": args.expected_novel_rows,
        "novel_count_matches_handover": bool(len(expected_novel) == args.expected_novel_rows),
        "novel_export_mapping_check": novel_check,
        "recomputed_novel_target_components": int(len(expected_novel_targets := set(expected_novel["_norm_target"]))),
        "oof_validation_rows_cover_each_train_row_once": bool(oof_coverage),
        "fold_reports": fold_reports,
        "overall_pass": overall_pass,
        "privacy": {
            "sentence_text_written_to_report": False,
            "article_identifiers_written_to_report": False,
            "row_level_audit_exports_created": False,
            "source_data_modified": False,
        },
        "limitations": [
            "The audit verifies exported fold CSVs against the current canonical split CSVs and row manifest.",
            "It does not establish semantic label correctness or independent-test generalization.",
        ],
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "context_oof_independent_integrity_report.json"
    folds_path = output / "context_oof_independent_fold_checks.csv"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([
        {k: v for k, v in row.items() if k not in {"train_export_exact_match", "validation_export_exact_match"}}
        for row in fold_reports
    ]).to_csv(folds_path, index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nAggregate report: {report_path.resolve()}")
    print(f"Fold check table: {folds_path.resolve()}")
    if not overall_pass:
        raise SystemExit("Independent Context OOF audit FAILED; inspect the aggregate report.")


if __name__ == "__main__":
    main()
