"""Parse the user-provided AIGTD train/test JSON schemas and create article-level splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

try:
    from scripts.common import (
        cfg_path, ensure_project_dirs, load_config, parse_test_json, parse_train_json,
        resolve_path, set_seed,
    )
except ImportError:
    from common import (
        cfg_path, ensure_project_dirs, load_config, parse_test_json, parse_train_json,
        resolve_path, set_seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess AIGTD train and unlabeled test JSON files.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--train", default=None, help="Optional override for training JSON path")
    parser.add_argument("--test", default=None, help="Optional override for test JSON path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    seed = int(cfg.get("project", {}).get("random_seed", 42))
    set_seed(seed)

    train_path = resolve_path(args.train) if args.train else cfg_path(cfg, "train_json", "data/raw/AIGTD_train_sampled_5000.json")
    test_path = resolve_path(args.test) if args.test else cfg_path(cfg, "test_json", "data/raw/AIGTD_test_without_labels.json")
    processed_dir = cfg_path(cfg, "processed_dir", "data/processed")
    splits_dir = cfg_path(cfg, "splits_dir", "data/splits")
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_df, train_articles = parse_train_json(train_path)
    train_csv = processed_dir / "train_sentences.csv"
    train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")

    # GroupShuffleSplit keeps all sentences from an article in exactly one split.
    val_size = float(cfg.get("split", {}).get("validation_size", 0.20))
    if not 0.05 <= val_size <= 0.5:
        raise ValueError("split.validation_size 建议设置在 0.05 到 0.50 之间。")
    groups = train_df["article_id"].astype(str).to_numpy()
    if len(set(groups)) < 2:
        raise ValueError("训练数据至少需要两篇文章才能按文章划分训练集和验证集。")
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=int(cfg.get("split", {}).get("random_seed", seed)),
    )
    train_idx, val_idx = next(splitter.split(train_df, train_df["label_id"], groups=groups))
    train_split = train_df.iloc[train_idx].reset_index(drop=True)
    val_split = train_df.iloc[val_idx].reset_index(drop=True)
    if train_split["label_id"].nunique() < 2 or val_split["label_id"].nunique() < 2:
        raise RuntimeError(
            "随机文章划分后，训练集或验证集缺少某个类别。请检查每篇文章的标签组成，"
            "或调整验证比例/划分策略。"
        )

    # Keep every occurrence of the same source article_id in one split.
    # article_key distinguishes record occurrences for context construction.
    train_ids = set(train_split["article_id"].astype(str))
    val_ids = set(val_split["article_id"].astype(str))
    train_article_keys = set(train_split["article_key"].astype(str))
    val_article_keys = set(val_split["article_key"].astype(str))
    overlap = train_ids & val_ids
    if overlap:
        raise RuntimeError(f"文章级划分失败，发现交集：{list(overlap)[:10]}")

    train_split.to_csv(splits_dir / "train.csv", index=False, encoding="utf-8-sig")
    val_split.to_csv(splits_dir / "validation.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "train_source": str(train_path),
        "validation_size_requested": val_size,
        "random_seed": int(cfg.get("split", {}).get("random_seed", seed)),
        "train_article_record_count": len(train_article_keys),
        "validation_article_record_count": len(val_article_keys),
        "train_unique_source_id_count": len(train_ids),
        "validation_unique_source_id_count": len(val_ids),
        "train_sentence_count": len(train_split),
        "validation_sentence_count": len(val_split),
        "train_article_ids": sorted(train_ids),
        "validation_article_ids": sorted(val_ids),
        "train_article_keys": sorted(train_article_keys),
        "validation_article_keys": sorted(val_article_keys),
        "article_id_overlap_count": len(overlap),
        "article_key_overlap_count": len(train_article_keys & val_article_keys),
        "split_method": "GroupShuffleSplit by source article_id; article_key identifies each JSON record",
    }
    with (splits_dir / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    test_summary = "未处理（未找到测试文件）"
    if test_path.exists():
        test_df, test_articles = parse_test_json(test_path)
        test_df.to_csv(processed_dir / "test_sentences.csv", index=False, encoding="utf-8-sig")
        test_ids = set(test_df["article_id"].astype(str))
        train_test_overlap = train_ids | val_ids
        shared_ids = test_ids & train_test_overlap
        test_record_count = test_df["article_key"].nunique() if "article_key" in test_df else len(test_ids)
        test_summary = (
            f"测试文章记录 {test_record_count:,} 条，测试 source article_id {len(test_ids):,} 个，"
            f"测试句子 {len(test_df):,} 条；与训练来源 article_id 重叠 {len(shared_ids)} 个。"
        )
        if shared_ids:
            print(
                "[警告] train/test 存在相同 source article_id。由于 ID 本身可能重复，"
                "仅凭 ID 交集不能判定泄漏，请进一步比较文章内容：",
                list(shared_ids)[:10],
            )
    else:
        print(f"[警告] 测试集文件不存在，跳过测试集解析：{test_path}")

    print("\n预处理完成")
    print(f"训练源：{train_path}")
    print(
        f"文章记录数：{train_df['article_key'].nunique():,}；"
        f"不同 source article_id 数：{train_df['article_id'].nunique():,}；句子数：{len(train_df):,}"
    )
    print(
        f"训练 split：{len(train_article_keys):,} 条文章记录 / {len(train_ids):,} 个 source ID / "
        f"{len(train_split):,} 句"
    )
    print(
        f"验证 split：{len(val_article_keys):,} 条文章记录 / {len(val_ids):,} 个 source ID / "
        f"{len(val_split):,} 句"
    )
    print(f"Human/Machine：{int((train_df.label_id == 0).sum()):,} / {int((train_df.label_id == 1).sum()):,}")
    print(f"测试集：{test_summary}")
    print(f"输出：{processed_dir}、{splits_dir}")


if __name__ == "__main__":
    main()
