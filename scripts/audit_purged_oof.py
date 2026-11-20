#!/usr/bin/env python3
"""Audit source-ID grouped OOF folds with per-fold exact-text purging.

Why this strategy:
- Keep all article records sharing source_article_id in the same validation fold.
- For each fold, remove from that fold's training rows every normalized sentence
  that occurs in that fold's validation rows.
- This avoids transitive connected-component collapse caused by common sentences,
  while ensuring no exact normalized target sentence is shared between the
  training and validation rows of the same fold.

This script ONLY audits and writes manifests/statistics. It does NOT train models
or modify the original data. It is intended first for sentence-only SVM/RoBERTa OOF.
For context-window models, stricter article/window-level purging is required.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def normalize_text(value: Any) -> str:
    """Match project normalization: NFKC + casefold + collapsed whitespace."""
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def to_label(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("Found missing label value.")
    s = str(value).strip().casefold()
    if s in {"human", "0", "0.0"}:
        return 0
    if s in {"machine", "1", "1.0"}:
        return 1
    try:
        v = int(float(s))
    except Exception as exc:
        raise ValueError(f"Unrecognized label: {value!r}") from exc
    if v not in (0, 1):
        raise ValueError(f"Label must be human=0/machine=1, got {value!r}")
    return v


def choose_col(df: pd.DataFrame, names: list[str], description: str) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(
        f"Cannot find {description}. Tried {names}; columns={list(df.columns)}"
    )


def stable_greedy_group_folds(group_stats: pd.DataFrame, n_splits: int, seed: int) -> dict[str, int]:
    """Greedy multi-dimensional bin packing by source ID groups."""
    dims = ["sentence_count", "article_count", "human_count", "machine_count"]
    totals = group_stats[dims].sum().to_numpy(dtype=float)
    targets = np.maximum(totals / n_splits, 1.0)
    loads = np.zeros((n_splits, len(dims)), dtype=float)
    rng = np.random.default_rng(seed)

    work = group_stats.copy()
    # Seeded tie-break only; main order is descending sentence volume.
    work["_tie"] = rng.random(len(work))
    work = work.sort_values(
        ["sentence_count", "article_count", "_tie"],
        ascending=[False, False, True],
        kind="mergesort",
    )

    assignment: dict[str, int] = {}
    for _, row in work.iterrows():
        vec = row[dims].to_numpy(dtype=float)
        scores: list[float] = []
        for fold in range(n_splits):
            candidate = loads.copy()
            candidate[fold] += vec
            # Minimize total normalized squared deviation from equal fold targets.
            score = float(np.sum(((candidate - targets) / targets) ** 2))
            scores.append(score)
        best = np.flatnonzero(np.isclose(scores, np.min(scores), rtol=0, atol=1e-12))
        chosen = int(best[0])
        assignment[str(row["source_group_key"])] = chosen
        loads[chosen] += vec
    return assignment


def safe_div(a: int | float, b: int | float) -> float:
    return float(a / b) if b else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/splits/train.csv")
    parser.add_argument("--output-dir", default="results/oof_purged_group_audit")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.n_splits < 2:
        raise ValueError("--n-splits must be >= 2")

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)
    article_col = choose_col(df, ["article_key"], "unique article record key")
    source_col = choose_col(
        df, ["article_id", "source_article_id", "source_id"], "source article ID"
    )
    text_col = choose_col(df, ["text", "sentence", "sent"], "sentence text")
    label_col = choose_col(df, ["label_id", "label", "y_true"], "label")

    if df[article_col].isna().any():
        raise ValueError(f"{article_col} contains missing values.")
    df[article_col] = df[article_col].astype(str)
    # article_key naturally repeats once per sentence row. It must map to exactly
    # one source ID, which is checked below; it is not expected to be row-unique.

    # article_key must map to exactly one source ID. Missing IDs get a private key
    # so unrelated missing-ID records are never accidentally grouped together.
    source_map: dict[str, str] = {}
    for key, g in df.groupby(article_col, sort=False):
        vals = [str(x).strip() for x in g[source_col].dropna().unique() if str(x).strip()]
        if len(vals) > 1:
            raise ValueError(f"{article_col}={key} maps to multiple source IDs: {vals[:5]}")
        source_map[str(key)] = vals[0] if vals else f"__MISSING_SOURCE_ID__::{key}"

    df["_source_group_key"] = df[article_col].map(source_map)
    df["_norm_text"] = df[text_col].map(normalize_text)
    df["_label_id"] = df[label_col].map(to_label)
    if (df["_norm_text"] == "").any():
        n_empty = int((df["_norm_text"] == "").sum())
        raise ValueError(f"Found {n_empty} empty normalized sentence(s); inspect data first.")

    # Check each article record has a unique, continuous sentence_index if provided.
    sentence_index_col = "sentence_index" if "sentence_index" in df.columns else None
    if sentence_index_col:
        dup_idx = df.duplicated([article_col, sentence_index_col]).sum()
        if dup_idx:
            raise ValueError(
                f"Found {dup_idx} duplicate ({article_col}, {sentence_index_col}) rows."
            )

    # Build group-level stats using source IDs only (not duplicate-text connectivity).
    article_stats = (
        df.groupby(["_source_group_key", article_col], sort=False)
        .agg(sentence_count=(text_col, "size"),
             human_count=("_label_id", lambda x: int((x == 0).sum())),
             machine_count=("_label_id", lambda x: int((x == 1).sum())))
        .reset_index()
    )
    group_stats = (
        article_stats.groupby("_source_group_key", sort=False)
        .agg(article_count=(article_col, "nunique"),
             sentence_count=("sentence_count", "sum"),
             human_count=("human_count", "sum"),
             machine_count=("machine_count", "sum"))
        .reset_index()
        .rename(columns={"_source_group_key": "source_group_key"})
    )
    if len(group_stats) < args.n_splits:
        raise ValueError(
            f"Only {len(group_stats)} source groups; cannot create {args.n_splits} folds."
        )

    fold_map = stable_greedy_group_folds(group_stats, args.n_splits, args.seed)
    group_stats["fold"] = group_stats["source_group_key"].map(fold_map).astype(int)

    # Article-level manifest.
    article_manifest = pd.DataFrame({
        "article_key": list(source_map.keys()),
        "source_article_id": list(source_map.values()),
    })
    article_manifest["fold"] = article_manifest["source_article_id"].map(fold_map).astype(int)
    article_manifest.to_csv(out_dir / "article_fold_manifest.csv", index=False, encoding="utf-8-sig")

    # Row-level manifest for later OOF generation; no text is copied into this file.
    row_manifest = df[[article_col]].copy()
    row_manifest["row_id"] = np.arange(len(df), dtype=int)
    row_manifest["source_article_id"] = df["_source_group_key"].to_numpy()
    row_manifest["fold"] = df["_source_group_key"].map(fold_map).astype(int).to_numpy()
    if sentence_index_col:
        row_manifest["sentence_index"] = df[sentence_index_col].to_numpy()
    row_manifest.to_csv(out_dir / "row_fold_manifest.csv", index=False, encoding="utf-8-sig")

    fold_rows: list[dict[str, Any]] = []
    purge_detail_rows: list[dict[str, Any]] = []
    validation_norm_to_folds: dict[str, set[int]] = {}

    for fold in range(args.n_splits):
        val_mask = df["_source_group_key"].map(fold_map).to_numpy() == fold
        val_df = df.loc[val_mask]
        candidate_train_df = df.loc[~val_mask]

        val_norms = set(val_df["_norm_text"].tolist())
        for norm in val_norms:
            validation_norm_to_folds.setdefault(norm, set()).add(fold)

        purge_mask = candidate_train_df["_norm_text"].isin(val_norms)
        purged_df = candidate_train_df.loc[purge_mask]
        clean_train_df = candidate_train_df.loc[~purge_mask]

        # Exact post-purge validation checks.
        remaining_overlap = set(clean_train_df["_norm_text"]) & val_norms
        source_overlap = set(clean_train_df["_source_group_key"]) & set(val_df["_source_group_key"])
        article_overlap = set(clean_train_df[article_col]) & set(val_df[article_col])
        if remaining_overlap or source_overlap or article_overlap:
            raise RuntimeError(
                f"Fold {fold}: leakage check failed: normalized_text={len(remaining_overlap)}, "
                f"source_id={len(source_overlap)}, article_key={len(article_overlap)}"
            )

        def counts(part: pd.DataFrame, prefix: str) -> dict[str, Any]:
            return {
                f"{prefix}_rows": int(len(part)),
                f"{prefix}_articles": int(part[article_col].nunique()),
                f"{prefix}_source_ids": int(part["_source_group_key"].nunique()),
                f"{prefix}_human": int((part["_label_id"] == 0).sum()),
                f"{prefix}_machine": int((part["_label_id"] == 1).sum()),
            }

        record: dict[str, Any] = {"fold": fold}
        record.update(counts(val_df, "validation"))
        record.update(counts(candidate_train_df, "candidate_train"))
        record.update(counts(purged_df, "purged_from_train"))
        record.update(counts(clean_train_df, "final_train"))
        record["purged_train_row_pct"] = safe_div(len(purged_df), len(candidate_train_df))
        record["purged_unique_texts"] = int(purged_df["_norm_text"].nunique())
        record["validation_unique_texts"] = int(len(val_norms))
        record["post_purge_text_overlap_groups"] = int(len(remaining_overlap))
        record["post_purge_source_id_overlap"] = int(len(source_overlap))
        fold_rows.append(record)

        if not purged_df.empty:
            purged_counts = purged_df.groupby("_norm_text").size()
            val_counts = val_df.groupby("_norm_text").size()
            for norm, n_train in purged_counts.items():
                digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()
                purge_detail_rows.append({
                    "fold": fold,
                    "normalized_text_sha1": digest,
                    "text_sample": norm[:240],
                    "validation_occurrences": int(val_counts.get(norm, 0)),
                    "training_rows_removed": int(n_train),
                })

    fold_balance = pd.DataFrame(fold_rows)
    fold_balance.to_csv(out_dir / "fold_purge_balance.csv", index=False, encoding="utf-8-sig")
    if purge_detail_rows:
        pd.DataFrame(purge_detail_rows).to_csv(
            out_dir / "purge_details.csv", index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame(columns=["fold", "normalized_text_sha1", "text_sample",
                              "validation_occurrences", "training_rows_removed"]).to_csv(
            out_dir / "purge_details.csv", index=False, encoding="utf-8-sig"
        )

    # Global repeated-text / conflicting-label summaries.
    norm_stats = df.groupby("_norm_text").agg(
        occurrences=("_label_id", "size"),
        unique_labels=("_label_id", "nunique"),
        article_records=(article_col, "nunique"),
        source_ids=("_source_group_key", "nunique"),
    )
    duplicate_groups = norm_stats[norm_stats["occurrences"] > 1]
    conflict_groups = norm_stats[norm_stats["unique_labels"] > 1]
    val_cross_fold_duplicate_groups = sum(
        1 for folds in validation_norm_to_folds.values() if len(folds) > 1
    )

    lines = [
        "AIGTD source-ID grouped OOF with per-fold exact-text purge audit",
        "=" * 72,
        f"Input: {input_path}",
        f"Sentence rows: {len(df):,}",
        f"Article records: {df[article_col].nunique():,}",
        f"Unique source-ID groups: {df['_source_group_key'].nunique():,}",
        f"Folds: {args.n_splits}; seed: {args.seed}",
        "",
        "Grouping strategy:",
        "- Keep all records sharing source_article_id in one fold.",
        "- For each fold, purge from its training rows every normalized sentence",
        "  that occurs in that fold's validation rows.",
        "- Do NOT transitively connect all articles through shared sentence links.",
        "",
        f"Repeated normalized text groups (all data): {len(duplicate_groups):,}",
        f"Rows in repeated normalized text groups: {int(duplicate_groups['occurrences'].sum()) if len(duplicate_groups) else 0:,}",
        f"Conflicting-label normalized text groups: {len(conflict_groups):,}",
        f"Normalized text groups appearing in validation portions of >1 fold: {val_cross_fold_duplicate_groups:,}",
        "",
        "Per-fold balance and purge:",
    ]
    for _, r in fold_balance.iterrows():
        lines.append(
            f"Fold {int(r['fold'])}: val={int(r['validation_rows']):,} rows / "
            f"{int(r['validation_articles']):,} articles; candidate_train={int(r['candidate_train_rows']):,}; "
            f"purged={int(r['purged_from_train_rows']):,} ({r['purged_train_row_pct']:.2%}); "
            f"final_train={int(r['final_train_rows']):,}; "
            f"final_train human/machine={int(r['final_train_human']):,}/{int(r['final_train_machine']):,}; "
            f"remaining_text_overlap={int(r['post_purge_text_overlap_groups'])}"
        )

    val_sizes = fold_balance["validation_rows"].to_numpy(dtype=float)
    train_sizes = fold_balance["final_train_rows"].to_numpy(dtype=float)
    lines.extend([
        "",
        f"Validation sentence-share max-min gap: {safe_div(val_sizes.max()-val_sizes.min(), len(df)):.2%}",
        f"Final training sentence-count min/max: {int(train_sizes.min()):,}/{int(train_sizes.max()):,}",
        "",
        "Automatic checks:",
        f"- Every article_key assigned to exactly one fold: PASS",
        f"- Every source_article_id assigned to exactly one fold: PASS",
        f"- Per-fold exact normalized sentence overlap after purge: 0 (PASS)",
        f"- Per-fold source ID overlap after split: 0 (PASS)",
        "",
        "Important interpretation:",
        "- This is a fold/purge feasibility audit, not model OOF predictions.",
        "- It avoids giant transitive components created by common repeated sentences.",
        "- Training rows are purged separately for each fold; the same sentence may",
        "  appear in validation portions of multiple folds, but is removed from the",
        "  corresponding fold's training rows each time.",
        "- Suitable first for sentence-only SVM/RoBERTa. For context-window models,",
        "  a matching sentence must also be purged if it appears inside training",
        "  context windows; use article/window-level purge and audit separately.",
        "- Conflicting-label duplicates remain inherently ambiguous for sentence-only",
        "  models; report their counts and consider inverse-frequency weighting in",
        "  meta-model fitting as a separate ablation, not an automatic deletion.",
        "- Do not use this manifest for training until fold balance and purge rates",
        "  have been reviewed.",
    ])
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nOutputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
