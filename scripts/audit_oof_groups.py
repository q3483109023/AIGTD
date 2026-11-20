#!/usr/bin/env python3
"""Audit leakage-safe grouping candidates before generating AIGTD OOF predictions.

Groups article records together when either:
1) they share the same source article_id, or
2) they contain at least one identical normalized sentence.

This is an AUDIT / fold-planning script only. It does not train models or modify
raw/processed/split datasets. It writes a group manifest and a deterministic,
greedy 5-fold proposal for inspection.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: str) -> str:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def normalize_text(value: Any) -> str:
    """Match the project's novelty normalization: NFKC + casefold + whitespace."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def first_existing(columns: set[str], candidates: list[str], what: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"Cannot find {what} column. Tried {candidates}; available columns={sorted(columns)}"
    )


def label_to_id(value: Any) -> int | None:
    if pd.isna(value):
        return None
    s = str(value).strip().casefold()
    if s in {"human", "0", "0.0"}:
        return 0
    if s in {"machine", "1", "1.0"}:
        return 1
    try:
        n = int(float(s))
        return n if n in (0, 1) else None
    except Exception:
        return None


def greedy_assign_folds(summary: pd.DataFrame, n_splits: int) -> dict[str, int]:
    """Deterministic greedy bin packing, balancing rows, articles, and labels."""
    totals = summary[["sentence_count", "article_count", "human_count", "machine_count"]].sum()
    targets = totals / n_splits
    loads = np.zeros((n_splits, 4), dtype=float)
    assignment: dict[str, int] = {}

    ordered = summary.sort_values(
        ["sentence_count", "article_count"], ascending=False
    )
    values = ["sentence_count", "article_count", "human_count", "machine_count"]

    for _, row in ordered.iterrows():
        vector = np.array([float(row[c]) for c in values], dtype=float)
        denom = np.maximum(targets.to_numpy(dtype=float), 1.0)
        scores = []
        for fold in range(n_splits):
            candidate = loads.copy()
            candidate[fold] += vector
            # Relative squared imbalance across the four workload dimensions.
            score = np.sum(((candidate - targets.to_numpy(dtype=float)) / denom) ** 2)
            scores.append(score)
        chosen = int(np.argmin(scores))
        assignment[str(row["oof_group_id"])] = chosen
        loads[chosen] += vector
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/splits/train.csv")
    parser.add_argument("--output-dir", default="results/oof_group_audit")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Training split not found: {input_path}")
    if args.n_splits < 2:
        raise ValueError("n-splits must be >= 2")

    df = pd.read_csv(input_path)
    columns = set(df.columns)
    article_col = first_existing(columns, ["article_key"], "unique article record key")
    source_id_col = first_existing(
        columns, ["article_id", "source_article_id", "source_id"], "source article ID"
    )
    text_col = first_existing(columns, ["text", "sentence", "sent"], "sentence text")
    label_col = first_existing(columns, ["label", "label_id", "y_true"], "label")

    df[article_col] = df[article_col].astype(str)
    if df[article_col].duplicated().all():
        raise ValueError("article_key is not usable: every value is duplicated.")

    # Build one record per article_key and collect row-level normalized text.
    article_keys = sorted(df[article_col].dropna().unique().tolist())
    if not article_keys:
        raise ValueError("No article_key values found.")

    article_source: dict[str, str] = {}
    article_rows: dict[str, int] = defaultdict(int)
    article_human: dict[str, int] = defaultdict(int)
    article_machine: dict[str, int] = defaultdict(int)
    source_to_articles: dict[str, list[str]] = defaultdict(list)
    norm_sentence_to_articles: dict[str, list[str]] = defaultdict(list)

    for key, group in df.groupby(article_col, sort=False):
        key = str(key)
        source_values = [
            str(v).strip() for v in group[source_id_col].dropna().unique().tolist()
            if str(v).strip()
        ]
        if len(source_values) > 1:
            raise ValueError(
                f"article_key={key} maps to multiple source IDs: {source_values[:5]}"
            )
        source_id = source_values[0] if source_values else ""
        article_source[key] = source_id
        article_rows[key] = len(group)
        for value in group[label_col].tolist():
            y = label_to_id(value)
            if y == 0:
                article_human[key] += 1
            elif y == 1:
                article_machine[key] += 1

        if source_id:
            source_to_articles[source_id].append(key)

        seen_norms = set()
        for text in group[text_col].tolist():
            norm = normalize_text(text)
            if norm:
                seen_norms.add(norm)
        for norm in seen_norms:
            norm_sentence_to_articles[norm].append(key)

    uf = UnionFind(article_keys)

    # Constraint 1: same source article ID cannot cross folds.
    for keys in source_to_articles.values():
        if len(keys) > 1:
            first = keys[0]
            for key in keys[1:]:
                uf.union(first, key)

    # Constraint 2: exact normalized sentence overlap cannot cross folds.
    duplicate_norm_groups = 0
    duplicate_norm_article_links = 0
    for keys in norm_sentence_to_articles.values():
        unique_keys = sorted(set(keys))
        if len(unique_keys) > 1:
            duplicate_norm_groups += 1
            duplicate_norm_article_links += len(unique_keys)
            first = unique_keys[0]
            for key in unique_keys[1:]:
                uf.union(first, key)

    components: dict[str, list[str]] = defaultdict(list)
    for key in article_keys:
        components[uf.find(key)].append(key)

    ordered_components = sorted(
        components.values(), key=lambda xs: (-len(xs), min(xs))
    )
    key_to_group: dict[str, str] = {}
    component_records = []
    for i, keys in enumerate(ordered_components, start=1):
        group_id = f"oof_group_{i:05d}"
        total_rows = sum(article_rows[k] for k in keys)
        human = sum(article_human[k] for k in keys)
        machine = sum(article_machine[k] for k in keys)
        source_ids = sorted({article_source[k] for k in keys if article_source[k]})
        for key in keys:
            key_to_group[key] = group_id
        component_records.append({
            "oof_group_id": group_id,
            "article_count": len(keys),
            "sentence_count": total_rows,
            "human_count": human,
            "machine_count": machine,
            "source_id_count": len(source_ids),
            "article_keys": "|".join(sorted(keys)),
            "source_article_ids": "|".join(source_ids),
        })

    component_df = pd.DataFrame(component_records)
    total_articles = len(article_keys)
    total_sentences = len(df)
    largest = int(component_df.iloc[0]["article_count"])
    largest_rows = int(component_df.iloc[0]["sentence_count"])

    # Candidate fold plan is for inspection only; it is not yet an approved split.
    fold_assignment = greedy_assign_folds(component_df, args.n_splits)
    component_df["proposed_fold"] = component_df["oof_group_id"].map(fold_assignment)

    key_manifest = pd.DataFrame({
        "article_key": article_keys,
        "source_article_id": [article_source[k] for k in article_keys],
        "oof_group_id": [key_to_group[k] for k in article_keys],
    })
    key_manifest = key_manifest.merge(
        component_df[["oof_group_id", "proposed_fold", "article_count", "sentence_count"]],
        on="oof_group_id", how="left"
    )

    fold_rows = []
    for fold in range(args.n_splits):
        group_ids = set(component_df.loc[component_df["proposed_fold"] == fold, "oof_group_id"])
        keys = set(key_manifest.loc[key_manifest["oof_group_id"].isin(group_ids), "article_key"])
        sub = df[df[article_col].isin(keys)]
        labels = [label_to_id(x) for x in sub[label_col].tolist()]
        fold_rows.append({
            "fold": fold,
            "article_count": len(keys),
            "sentence_count": len(sub),
            "human_count": sum(y == 0 for y in labels),
            "machine_count": sum(y == 1 for y in labels),
            "group_count": len(group_ids),
        })
    fold_df = pd.DataFrame(fold_rows)

    component_df.to_csv(output_dir / "component_summary.csv", index=False, encoding="utf-8-sig")
    key_manifest.to_csv(output_dir / "article_group_manifest.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(output_dir / "proposed_fold_balance.csv", index=False, encoding="utf-8-sig")

    fold_sentence_share = fold_df["sentence_count"] / max(total_sentences, 1)
    imbalance = float(fold_sentence_share.max() - fold_sentence_share.min())
    report = [
        "AIGTD OOF grouping audit",
        "=" * 60,
        f"Input: {input_path}",
        f"Sentence rows: {total_sentences:,}",
        f"Unique article_key records: {total_articles:,}",
        f"Unique source article IDs: {len(set(x for x in article_source.values() if x)):,}",
        f"Normalized duplicate sentence groups spanning multiple article records: {duplicate_norm_groups:,}",
        f"Article-record occurrences in those duplicate groups (sum over groups): {duplicate_norm_article_links:,}",
        f"Connected components after joining same source ID OR shared normalized sentence: {len(component_df):,}",
        f"Largest component: {largest:,} article records / {largest_rows:,} sentences "
        f"({largest / max(total_articles, 1):.2%} of articles; {largest_rows / max(total_sentences, 1):.2%} of sentences)",
        "",
        f"Proposed {args.n_splits}-fold greedy balance (all linked records stay together):",
        fold_df.to_string(index=False),
        f"Sentence-share max-min gap: {imbalance:.2%}",
        "",
        "Top 20 connected components:",
        component_df.head(20)[[
            "oof_group_id", "article_count", "sentence_count",
            "human_count", "machine_count", "source_id_count", "proposed_fold"
        ]].to_string(index=False),
        "",
        "Interpretation warnings:",
        "- proposed_fold is a deterministic planning proposal, not an approved OOF split.",
        "- Connecting all repeated sentences can create very large components; inspect balance before training.",
        "- If one component dominates, strict sentence-disjoint OOF may be infeasible with ordinary 5-fold CV.",
        "- Do not split a connected component merely to force equal fold sizes; that reintroduces leakage.",
        "- This script checks exact normalized sentence overlap, not semantic paraphrases.",
    ]
    (output_dir / "summary.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))
    print(f"\nFiles written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
