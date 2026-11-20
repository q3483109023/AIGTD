#!/usr/bin/env python3
"""Materialize source-ID grouped, per-fold exact-text-purged OOF datasets.

Consumes the audited row_fold_manifest.csv produced by audit_purged_oof.py.
Does not modify the original train/validation splits.
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


def normalize_sentence(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def choose_col(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Cannot find {what}; tried {candidates}; columns={list(df.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/splits/train.csv")
    parser.add_argument("--validation", default="data/splits/validation.csv")
    parser.add_argument("--row-manifest", default="results/oof_purged_group_audit/row_fold_manifest.csv")
    parser.add_argument("--output-dir", default="data/oof_folds")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    train_path = Path(args.train)
    val_path = Path(args.validation)
    manifest_path = Path(args.row_manifest)
    out_dir = Path(args.output_dir)
    for path in (train_path, val_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.n_splits < 2:
        raise ValueError("--n-splits must be >= 2")

    df = pd.read_csv(train_path, encoding="utf-8-sig").reset_index(drop=True)
    val_external = pd.read_csv(val_path, encoding="utf-8-sig").reset_index(drop=True)
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")

    article_col = choose_col(df, ["article_key"], "article_key")
    text_col = choose_col(df, ["text"], "sentence text")
    label_col = choose_col(df, ["label_id"], "numeric label_id")
    sentence_col = choose_col(df, ["sentence_index"], "sentence_index")
    source_in_df = choose_col(df, ["article_id", "source_article_id", "source_id"], "source article ID")

    required_manifest = {"row_id", "article_key", "source_article_id", "fold", "sentence_index"}
    missing = required_manifest - set(manifest.columns)
    if missing:
        raise ValueError(f"row_fold_manifest missing columns: {sorted(missing)}")
    if len(manifest) != len(df):
        raise ValueError(f"Manifest row count {len(manifest)} != train rows {len(df)}")

    manifest = manifest.sort_values("row_id", kind="stable").reset_index(drop=True)
    expected_row_ids = np.arange(len(df))
    if not np.array_equal(manifest["row_id"].astype(int).to_numpy(), expected_row_ids):
        raise ValueError("row_fold_manifest row_id must be exactly 0..N-1")
    if not (manifest["article_key"].astype(str).to_numpy() == df[article_col].astype(str).to_numpy()).all():
        raise ValueError("Manifest article_key order does not match train.csv. Re-run audit on this exact train.csv.")
    if not (manifest["sentence_index"].astype(int).to_numpy() == df[sentence_col].astype(int).to_numpy()).all():
        raise ValueError("Manifest sentence_index order does not match train.csv.")
    if not (manifest["source_article_id"].astype(str).to_numpy() == df[source_in_df].astype(str).to_numpy()).all():
        raise ValueError("Manifest source_article_id does not match train.csv article ID.")

    folds = sorted(manifest["fold"].astype(int).unique().tolist())
    if folds != list(range(args.n_splits)):
        raise ValueError(f"Expected folds 0..{args.n_splits-1}, got {folds}")

    df["oof_row_id"] = expected_row_ids
    df["source_article_id"] = manifest["source_article_id"].astype(str).to_numpy()
    df["fold"] = manifest["fold"].astype(int).to_numpy()
    df["_norm_text"] = df[text_col].map(normalize_sentence)
    if (df["_norm_text"] == "").any():
        raise ValueError("Found empty normalized text in train.csv")

    out_dir.mkdir(parents=True, exist_ok=True)
    fold_summaries: list[dict[str, Any]] = []
    assigned_val_rows: list[int] = []

    for fold in folds:
        val_mask = df["fold"].eq(fold)
        fold_val = df.loc[val_mask].copy()
        candidate_train = df.loc[~val_mask].copy()
        val_norms = set(fold_val["_norm_text"])
        purge_mask = candidate_train["_norm_text"].isin(val_norms)
        purged = candidate_train.loc[purge_mask]
        fold_train = candidate_train.loc[~purge_mask].copy()

        # Hard checks: source IDs, article records, and normalized target text must be disjoint.
        overlap_text = set(fold_train["_norm_text"]) & val_norms
        overlap_source = set(fold_train["source_article_id"]) & set(fold_val["source_article_id"])
        overlap_article = set(fold_train["article_key"].astype(str)) & set(fold_val["article_key"].astype(str))
        if overlap_text or overlap_source or overlap_article:
            raise RuntimeError(
                f"Fold {fold} leakage: text={len(overlap_text)}, source={len(overlap_source)}, article={len(overlap_article)}"
            )

        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        drop_internal = ["_norm_text"]
        fold_train.drop(columns=drop_internal).to_csv(fold_dir / "train.csv", index=False, encoding="utf-8-sig")
        fold_val.drop(columns=drop_internal).to_csv(fold_dir / "validation.csv", index=False, encoding="utf-8-sig")
        purged.drop(columns=drop_internal).to_csv(fold_dir / "purged_train_rows.csv", index=False, encoding="utf-8-sig")
        assigned_val_rows.extend(fold_val["oof_row_id"].astype(int).tolist())

        cand_h = int((candidate_train[label_col].astype(int) == 0).sum())
        cand_m = int((candidate_train[label_col].astype(int) == 1).sum())
        purged_h = int((purged[label_col].astype(int) == 0).sum())
        purged_m = int((purged[label_col].astype(int) == 1).sum())
        fold_summaries.append({
            "fold": fold,
            "validation_rows": len(fold_val),
            "validation_articles": fold_val[article_col].nunique(),
            "validation_source_ids": fold_val["source_article_id"].nunique(),
            "validation_human": int((fold_val[label_col].astype(int) == 0).sum()),
            "validation_machine": int((fold_val[label_col].astype(int) == 1).sum()),
            "candidate_train_rows": len(candidate_train),
            "candidate_train_human": cand_h,
            "candidate_train_machine": cand_m,
            "purged_rows": len(purged),
            "purged_human": purged_h,
            "purged_machine": purged_m,
            "purged_row_pct": len(purged) / max(len(candidate_train), 1),
            "purged_human_pct_of_candidate_human": purged_h / max(cand_h, 1),
            "purged_machine_pct_of_candidate_machine": purged_m / max(cand_m, 1),
            "final_train_rows": len(fold_train),
            "final_train_articles": fold_train[article_col].nunique(),
            "final_train_source_ids": fold_train["source_article_id"].nunique(),
            "final_train_human": int((fold_train[label_col].astype(int) == 0).sum()),
            "final_train_machine": int((fold_train[label_col].astype(int) == 1).sum()),
            "post_purge_text_overlap_groups": len(overlap_text),
            "post_purge_source_overlap": len(overlap_source),
        })
        print(
            f"Fold {fold}: train={len(fold_train):,}; val={len(fold_val):,}; "
            f"purged={len(purged):,} ({len(purged)/max(len(candidate_train),1):.2%}); "
            f"text_overlap={len(overlap_text)}; source_overlap={len(overlap_source)}"
        )

    if len(assigned_val_rows) != len(df) or len(set(assigned_val_rows)) != len(df):
        raise RuntimeError("OOF validation rows do not cover every training row exactly once.")
    pd.DataFrame(fold_summaries).to_csv(out_dir / "fold_purge_balance.csv", index=False, encoding="utf-8-sig")

    # Export Novel Validation, computed against the entire OOF training source.
    if not {article_col, text_col, sentence_col}.issubset(val_external.columns):
        raise ValueError("validation.csv lacks article_key/text/sentence_index fields")
    train_norms = set(df["_norm_text"])
    val_external["_norm_text"] = val_external[text_col].map(normalize_sentence)
    val_external["is_novel_validation"] = val_external["_norm_text"].map(lambda x: bool(x) and x not in train_norms)
    novel_df = val_external.loc[val_external["is_novel_validation"]].copy()
    seen_count = int((~val_external["is_novel_validation"]).sum())
    if (novel_df["_norm_text"].isin(train_norms)).any():
        raise RuntimeError("Novel Validation export contains training text overlap.")
    novel_df.insert(0, "novel_row_id", np.arange(len(novel_df), dtype=int))
    novel_df.drop(columns=["_norm_text"]).to_csv(out_dir / "novel_validation.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "input_train": str(train_path),
        "input_validation": str(val_path),
        "row_manifest": str(manifest_path),
        "n_splits": args.n_splits,
        "train_rows": len(df),
        "validation_rows": len(val_external),
        "novel_validation_rows": len(novel_df),
        "seen_validation_rows": seen_count,
        "fold_strategy": "source_article_id grouped; per-fold exact normalized target-text purge",
        "normalization": "Unicode NFKC + casefold + whitespace normalization; punctuation retained",
        "checks": {
            "every_train_row_in_exactly_one_oof_validation_fold": True,
            "source_ids_disjoint_per_fold": True,
            "article_keys_disjoint_per_fold": True,
            "normalized_target_text_disjoint_per_fold_after_purge": True,
            "novel_validation_disjoint_from_all_oof_train_text": True,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nOOF fold datasets materialized.")
    print(f"Training rows covered exactly once: {len(assigned_val_rows):,}")
    print(f"Novel Validation exported: {len(novel_df):,}; seen: {seen_count:,}")
    print(f"Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
