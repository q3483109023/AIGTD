#!/usr/bin/env python3
"""Audit duplicate sentences and possible content leakage in AIGTD.

Expected project layout:
  configs/experiment.yaml
  data/processed/test_sentences.csv
  data/splits/train.csv
  data/splits/validation.csv

The script does not modify source data or existing splits. It writes reports to
results/duplicate_audit/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

try:
    from scripts.common import cfg_path, load_config
except ImportError:  # Support: python scripts/audit_duplicates.py
    from common import cfg_path, load_config


def normalize_sentence(value: object) -> str:
    """Conservative normalization for exact-text duplicate detection.

    NFKC, case folding and whitespace normalization are applied. Punctuation
    and word order are preserved to avoid treating merely similar sentences
    as exact duplicates.
    """
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_csv_checked(path: Path, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {split} 数据文件：{path}\n"
            "请先成功运行：python scripts/preprocess.py"
        )
    df = pd.read_csv(path, keep_default_na=False, encoding="utf-8-sig")
    required = {"article_id", "sentence_index", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少必要字段：{sorted(missing)}")
    if "article_key" not in df.columns:
        raise ValueError(
            f"{path} 缺少 article_key。请确认已使用修复重复 article_id 后的 preprocess.py，"
            "然后重新运行预处理。"
        )
    df = df.copy()
    df["split"] = split
    df["article_id"] = df["article_id"].astype(str)
    df["article_key"] = df["article_key"].astype(str)
    df["sentence_index"] = pd.to_numeric(df["sentence_index"], errors="coerce")
    if df["sentence_index"].isna().any():
        raise ValueError(f"{path} 的 sentence_index 存在非数值。")
    df["sentence_index"] = df["sentence_index"].astype(int)
    df["text"] = df["text"].astype(str)
    if "label" not in df.columns:
        df["label"] = ""
    else:
        df["label"] = df["label"].fillna("").astype(str).str.strip().str.lower()
    if "domain" not in df.columns:
        df["domain"] = "unknown"
    df["record_uid"] = split + "::" + df["article_key"]
    df["normalized_text"] = df["text"].map(normalize_sentence)
    df["sentence_hash"] = df["normalized_text"].map(sha256_text)
    return df


def split_counts(group: pd.DataFrame) -> str:
    counts = group["split"].value_counts().to_dict()
    return " | ".join(f"{name}={counts.get(name, 0)}" for name in ("train", "validation", "test"))


def build_sentence_duplicate_reports(all_df: pd.DataFrame, out_dir: Path) -> Dict[str, int]:
    duplicate_groups: List[Dict[str, object]] = []
    for normalized, group in all_df.groupby("normalized_text", sort=False, dropna=False):
        if not normalized or len(group) < 2:
            continue
        record_counts = group.groupby("record_uid", sort=False).size()
        labels = sorted({x for x in group["label"].tolist() if x in {"human", "machine"}})
        splits = sorted(set(group["split"]))
        duplicate_groups.append({
            "sentence_hash": sha256_text(normalized),
            "text_example": group.iloc[0]["text"],
            "total_occurrences": int(len(group)),
            "extra_occurrences": int(len(group) - 1),
            "article_record_count": int(group["record_uid"].nunique()),
            "source_article_id_count": int(group["article_id"].nunique()),
            "split_count": int(len(splits)),
            "splits": ",".join(splits),
            "split_occurrences": split_counts(group),
            "within_same_article_duplicate": bool(record_counts.max() > 1),
            "across_article_records": bool(group["record_uid"].nunique() > 1),
            "cross_split_duplicate": bool(len(splits) > 1),
            "labels_present": ",".join(labels),
            "label_conflict": bool(len(labels) > 1),
            "article_ids_sample": ",".join(group["article_id"].drop_duplicates().astype(str).head(30)),
            "article_keys_sample": ",".join(group["record_uid"].drop_duplicates().astype(str).head(30)),
        })

    dup_df = pd.DataFrame(duplicate_groups)
    if not dup_df.empty:
        dup_df = dup_df.sort_values(
            ["cross_split_duplicate", "label_conflict", "total_occurrences"],
            ascending=[False, False, False],
        )
    dup_df.to_csv(out_dir / "duplicate_sentence_groups.csv", index=False, encoding="utf-8-sig")

    if dup_df.empty:
        cross_split = pd.DataFrame()
        conflicts = pd.DataFrame()
        train_val = pd.DataFrame()
        dev_test = pd.DataFrame()
    else:
        cross_split = dup_df[dup_df["cross_split_duplicate"]].copy()
        conflicts = dup_df[dup_df["label_conflict"]].copy()
        train_val = dup_df[
            dup_df["splits"].map(lambda x: {"train", "validation"}.issubset(set(x.split(","))))
        ].copy()
        dev_test = dup_df[
            dup_df["splits"].map(lambda x: "test" in x.split(",") and bool({"train", "validation"} & set(x.split(","))))
        ].copy()

    cross_split.to_csv(out_dir / "cross_split_duplicate_sentences.csv", index=False, encoding="utf-8-sig")
    train_val.to_csv(out_dir / "train_validation_sentence_overlap.csv", index=False, encoding="utf-8-sig")
    dev_test.to_csv(out_dir / "development_test_sentence_overlap.csv", index=False, encoding="utf-8-sig")
    conflicts.to_csv(out_dir / "conflicting_duplicate_labels.csv", index=False, encoding="utf-8-sig")

    # Detailed occurrences are useful for tracing a duplicate to exact records.
    if not dup_df.empty:
        duplicated_hashes = set(dup_df["sentence_hash"].astype(str))
        detail = all_df[all_df["sentence_hash"].isin(duplicated_hashes)].copy()
        cols = [
            "split", "article_id", "article_key", "sentence_index", "label",
            "domain", "sentence_hash", "text",
        ]
        detail[cols].sort_values(["sentence_hash", "split", "article_id", "article_key", "sentence_index"]).to_csv(
            out_dir / "duplicate_sentence_occurrences.csv", index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame(columns=[
            "split", "article_id", "article_key", "sentence_index", "label", "domain", "sentence_hash", "text"
        ]).to_csv(out_dir / "duplicate_sentence_occurrences.csv", index=False, encoding="utf-8-sig")

    labeled_df = all_df[all_df["split"].isin(["train", "validation"]) & all_df["label"].isin(["human", "machine"])]
    label_conflict_groups = 0
    if not dup_df.empty:
        label_conflict_groups = int(dup_df["label_conflict"].sum())

    return {
        "nonempty_sentence_rows_all_splits": int(len(all_df)),
        "distinct_normalized_sentences_all_splits": int(all_df.loc[all_df["normalized_text"] != "", "normalized_text"].nunique()),
        "duplicate_sentence_groups_all_splits": int(len(dup_df)),
        "rows_in_duplicate_groups_all_splits": int(dup_df["total_occurrences"].sum()) if not dup_df.empty else 0,
        "extra_duplicate_occurrences_all_splits": int(dup_df["extra_occurrences"].sum()) if not dup_df.empty else 0,
        "within_article_duplicate_groups": int(dup_df["within_same_article_duplicate"].sum()) if not dup_df.empty else 0,
        "cross_article_record_duplicate_groups": int(dup_df["across_article_records"].sum()) if not dup_df.empty else 0,
        "cross_split_duplicate_groups": int(dup_df["cross_split_duplicate"].sum()) if not dup_df.empty else 0,
        "train_validation_duplicate_groups": int(len(train_val)),
        "development_test_duplicate_groups": int(len(dev_test)),
        "duplicate_groups_with_human_machine_label_conflict": label_conflict_groups,
        "labeled_sentence_rows_train_validation": int(len(labeled_df)),
    }


def build_article_reports(all_df: pd.DataFrame, out_dir: Path) -> Dict[str, int]:
    article_rows: List[Dict[str, object]] = []
    for record_uid, group in all_df.groupby("record_uid", sort=False):
        group = group.sort_values("sentence_index", kind="stable")
        normalized_sentences = group["normalized_text"].tolist()
        sequence_text = "\n< SENTENCE_BOUNDARY >\n".join(normalized_sentences)
        # Also build a set signature to help identify same sentence sets with changed order.
        set_text = "\n< SENTENCE_BOUNDARY >\n".join(sorted(set(normalized_sentences)))
        article_rows.append({
            "split": group.iloc[0]["split"],
            "record_uid": record_uid,
            "article_id": group.iloc[0]["article_id"],
            "article_key": group.iloc[0]["article_key"],
            "domain": group.iloc[0].get("domain", "unknown"),
            "sentence_count": int(len(group)),
            "ordered_article_hash": sha256_text(sequence_text),
            "sentence_set_hash": sha256_text(set_text),
        })
    articles = pd.DataFrame(article_rows)
    if articles.empty:
        articles.to_csv(out_dir / "article_content_signatures.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(out_dir / "cross_split_exact_article_duplicates.csv", index=False, encoding="utf-8-sig")
        return {"article_records_audited": 0, "cross_split_exact_article_duplicates": 0, "cross_split_sentence_set_duplicates": 0}

    articles.to_csv(out_dir / "article_content_signatures.csv", index=False, encoding="utf-8-sig")
    dup_ordered = articles[articles.duplicated("ordered_article_hash", keep=False)].copy()
    dup_set = articles[articles.duplicated("sentence_set_hash", keep=False)].copy()
    dup_ordered = dup_ordered[dup_ordered.groupby("ordered_article_hash")["split"].transform("nunique") > 1]
    dup_set = dup_set[dup_set.groupby("sentence_set_hash")["split"].transform("nunique") > 1]
    dup_ordered.to_csv(out_dir / "cross_split_exact_article_duplicates.csv", index=False, encoding="utf-8-sig")
    dup_set.to_csv(out_dir / "cross_split_article_sentence_set_duplicates.csv", index=False, encoding="utf-8-sig")
    return {
        "article_records_audited": int(len(articles)),
        "cross_split_exact_article_duplicates": int(dup_ordered["ordered_article_hash"].nunique()),
        "cross_split_sentence_set_duplicates": int(dup_set["sentence_set_hash"].nunique()),
    }


def build_id_overlap_report(all_df: pd.DataFrame, out_dir: Path) -> Dict[str, int]:
    dev = all_df[all_df["split"].isin(["train", "validation"])]
    test = all_df[all_df["split"] == "test"]
    dev_ids = set(dev["article_id"].astype(str))
    test_ids = set(test["article_id"].astype(str))
    shared_ids = sorted(dev_ids & test_ids)

    rows: List[Dict[str, object]] = []
    for article_id in shared_ids:
        d = dev[dev["article_id"] == article_id]
        t = test[test["article_id"] == article_id]
        d_sent = set(d["normalized_text"]) - {""}
        t_sent = set(t["normalized_text"]) - {""}
        shared = d_sent & t_sent
        union = d_sent | t_sent
        d_signatures = set()
        t_signatures = set()
        for frame, target in ((d, d_signatures), (t, t_signatures)):
            for _, group in frame.groupby("record_uid", sort=False):
                ordered = group.sort_values("sentence_index", kind="stable")["normalized_text"].tolist()
                target.add(sha256_text("\n< SENTENCE_BOUNDARY >\n".join(ordered)))
        rows.append({
            "article_id": article_id,
            "development_article_records": int(d["record_uid"].nunique()),
            "test_article_records": int(t["record_uid"].nunique()),
            "development_unique_sentences": int(len(d_sent)),
            "test_unique_sentences": int(len(t_sent)),
            "shared_unique_sentences": int(len(shared)),
            "sentence_jaccard": round(len(shared) / len(union), 6) if union else 0.0,
            "exact_ordered_article_matches": int(len(d_signatures & t_signatures)),
            "development_sentence_examples": " || ".join(list(d_sent)[:5]),
            "test_sentence_examples": " || ".join(list(t_sent)[:5]),
        })
    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(["exact_ordered_article_matches", "shared_unique_sentences", "sentence_jaccard"], ascending=False)
    report.to_csv(out_dir / "train_test_shared_article_ids.csv", index=False, encoding="utf-8-sig")
    return {
        "development_unique_article_ids": int(len(dev_ids)),
        "test_unique_article_ids": int(len(test_ids)),
        "train_test_shared_article_ids": int(len(shared_ids)),
        "shared_ids_with_any_exact_sentence_overlap": int((report["shared_unique_sentences"] > 0).sum()) if not report.empty else 0,
        "shared_ids_with_exact_ordered_article_match": int((report["exact_ordered_article_matches"] > 0).sum()) if not report.empty else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit duplicate sentences and possible AIGTD data leakage.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = cfg_path(cfg, "processed_dir", "data/processed")
    splits_dir = cfg_path(cfg, "splits_dir", "data/splits")
    results_dir = cfg_path(cfg, "results_dir", "results")
    out_dir = results_dir / "duplicate_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = read_csv_checked(splits_dir / "train.csv", "train")
    validation = read_csv_checked(splits_dir / "validation.csv", "validation")
    test = read_csv_checked(processed_dir / "test_sentences.csv", "test")
    all_df = pd.concat([train, validation, test], ignore_index=True)

    sentence_stats = build_sentence_duplicate_reports(all_df, out_dir)
    article_stats = build_article_reports(all_df, out_dir)
    id_stats = build_id_overlap_report(all_df, out_dir)

    summary = {
        "inputs": {
            "train_split": str(splits_dir / "train.csv"),
            "validation_split": str(splits_dir / "validation.csv"),
            "unlabeled_test": str(processed_dir / "test_sentences.csv"),
        },
        "sentence_duplicate_audit": sentence_stats,
        "article_content_audit": article_stats,
        "train_test_article_id_audit": id_stats,
        "normalization": "Unicode NFKC + casefold + whitespace normalization; punctuation retained",
        "interpretation": [
            "Exact sentence overlap between train and validation may inflate validation metrics.",
            "Exact sentence overlap between development data and test is reported for awareness; test labels are unavailable.",
            "A repeated article_id is not treated as proof of leakage; article text signatures and sentence overlap are compared.",
            "This script only reports duplicates and does not delete or move any sample.",
        ],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "AIGTD 重复文本与潜在数据泄漏审计",
        "=" * 48,
        f"训练 split 句子：{len(train):,}",
        f"验证 split 句子：{len(validation):,}",
        f"无标签测试句子：{len(test):,}",
        "",
        "一、重复句子",
        f"重复文本组数：{sentence_stats['duplicate_sentence_groups_all_splits']:,}",
        f"重复组中的句子总出现次数：{sentence_stats['rows_in_duplicate_groups_all_splits']:,}",
        f"超过首次出现的重复次数：{sentence_stats['extra_duplicate_occurrences_all_splits']:,}",
        f"跨文章记录重复的文本组：{sentence_stats['cross_article_record_duplicate_groups']:,}",
        f"同一文章内部重复的文本组：{sentence_stats['within_article_duplicate_groups']:,}",
        f"跨数据 split 重复的文本组：{sentence_stats['cross_split_duplicate_groups']:,}",
        f"训练/验证之间重复的文本组：{sentence_stats['train_validation_duplicate_groups']:,}",
        f"开发集/测试集之间重复的文本组：{sentence_stats['development_test_duplicate_groups']:,}",
        f"同时出现 human 和 machine 标签的重复文本组：{sentence_stats['duplicate_groups_with_human_machine_label_conflict']:,}",
        "",
        "二、文章内容重复",
        f"检查的文章记录数：{article_stats['article_records_audited']:,}",
        f"跨 split 完全相同（顺序一致）的文章组：{article_stats['cross_split_exact_article_duplicates']:,}",
        f"跨 split 句子集合相同（忽略顺序）的文章组：{article_stats['cross_split_sentence_set_duplicates']:,}",
        "",
        "三、训练/测试相同 article_id 检查",
        f"开发集不同 article_id：{id_stats['development_unique_article_ids']:,}",
        f"测试集不同 article_id：{id_stats['test_unique_article_ids']:,}",
        f"共享 article_id：{id_stats['train_test_shared_article_ids']:,}",
        f"共享 ID 中存在完全相同句子的 ID 数：{id_stats['shared_ids_with_any_exact_sentence_overlap']:,}",
        f"共享 ID 中存在完全相同整篇文章的 ID 数：{id_stats['shared_ids_with_exact_ordered_article_match']:,}",
        "",
        "四、报告文件",
        *[f"- {p.name}" for p in sorted(out_dir.iterdir()) if p.is_file() and p.name != "summary.txt"],
        "",
        "说明：本脚本只审计和报告，不删除数据、不重划分数据。请先检查 train_validation_sentence_overlap.csv、",
        "conflicting_duplicate_labels.csv、cross_split_exact_article_duplicates.csv 和 train_test_shared_article_ids.csv。",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\nAIGTD 重复文本与潜在数据泄漏审计完成")
    print(f"重复文本组：{sentence_stats['duplicate_sentence_groups_all_splits']:,}")
    print(f"额外重复出现次数：{sentence_stats['extra_duplicate_occurrences_all_splits']:,}")
    print(f"训练/验证重复文本组：{sentence_stats['train_validation_duplicate_groups']:,}")
    print(f"开发集/测试重复文本组：{sentence_stats['development_test_duplicate_groups']:,}")
    print(f"跨 split 完全相同文章组：{article_stats['cross_split_exact_article_duplicates']:,}")
    print(f"Human/Machine 标签冲突重复组：{sentence_stats['duplicate_groups_with_human_machine_label_conflict']:,}")
    print(f"训练/测试共享 article_id：{id_stats['train_test_shared_article_ids']:,}")
    print(f"详细报告：{out_dir}")


if __name__ == "__main__":
    main()
