"""Inspect the concrete AIGTD JSON schema and generate a standalone HTML report."""
from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.common import load_config, parse_test_json, parse_train_json, resolve_path, cfg_path
except ImportError:
    from common import load_config, parse_test_json, parse_train_json, resolve_path, cfg_path


def describe_lengths(series: pd.Series) -> dict:
    values = series.astype(str).str.split().str.len()
    if values.empty:
        return {"count": 0}
    return {
        "count": int(values.count()),
        "mean": round(float(values.mean()), 3),
        "median": round(float(values.median()), 3),
        "p90": round(float(values.quantile(0.90)), 3),
        "max": int(values.max()),
        "min": int(values.min()),
    }


def make_table(frame: pd.DataFrame, title: str) -> str:
    if frame.empty:
        return f"<h3>{html.escape(title)}</h3><p>无数据</p>"
    return f"<h3>{html.escape(title)}</h3>{frame.to_html(index=False, border=0, classes='data-table', escape=True)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AIGTD JSON datasets and write an offline HTML report.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--train", default=None)
    parser.add_argument("--test", default=None)
    parser.add_argument("--output", default="results/dataset_inspection.html")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_path = resolve_path(args.train) if args.train else cfg_path(cfg, "train_json", "data/raw/AIGTD_train_sampled_5000.json")
    test_path = resolve_path(args.test) if args.test else cfg_path(cfg, "test_json", "data/raw/AIGTD_test_without_labels.json")
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    train_df, train_articles = parse_train_json(train_path)
    test_df, test_articles = (parse_test_json(test_path) if test_path.exists() else (pd.DataFrame(), []))

    train_article_df = pd.DataFrame(train_articles)
    test_article_df = pd.DataFrame(test_articles)
    label_counts = train_df["label"].value_counts().rename_axis("label").reset_index(name="sentence_count")
    label_counts["proportion"] = (label_counts["sentence_count"] / len(train_df)).round(4)
    domain_counts = train_df.groupby(["domain", "label"]).size().unstack(fill_value=0).reset_index()
    article_group_col = "article_key" if "article_key" in train_df.columns else "article_id"
    article_sizes = train_df.groupby(article_group_col).size()
    duplicate_sentence_count = int(train_df.duplicated(subset=["text"]).sum())
    article_id_counts = train_df[["article_id", article_group_col]].drop_duplicates().groupby("article_id").size()
    duplicate_article_id_count = int((article_id_counts > 1).sum())
    duplicate_article_id_extra_records = int((article_id_counts - 1).clip(lower=0).sum())
    duplicate_id_details = (
        pd.DataFrame(train_articles)
        .groupby("article_id", as_index=False)
        .agg(record_count=("article_key", "count"))
        .query("record_count > 1")
        .sort_values("record_count", ascending=False)
        .head(30)
    )
    cross_train_test_ids = sorted(set(train_df.article_id.astype(str)) & set(test_df.article_id.astype(str))) if not test_df.empty else []
    label_mix = train_df.groupby(article_group_col)["label"].nunique()
    mixed_articles = int((label_mix == 2).sum())
    non_mixed_articles = int((label_mix < 2).sum())

    overview = pd.DataFrame([
        ["训练文件", str(train_path)],
        ["训练文章记录数", f"{train_df[article_group_col].nunique():,}"],
        ["不同 source article_id 数", f"{train_df.article_id.nunique():,}"],
        ["训练句子数", f"{len(train_df):,}"],
        ["训练领域数", f"{train_df.domain.nunique():,}"],
        ["平均每篇句子数", f"{article_sizes.mean():.2f}"],
        ["句子词数分布", json.dumps(describe_lengths(train_df.text), ensure_ascii=False)],
        ["同时含 human/machine 的文章", f"{mixed_articles:,}"],
        ["只含单一标签的文章", f"{non_mixed_articles:,}"],
        ["重复句子（按完全相同文本）", f"{duplicate_sentence_count:,}"],
        ["具有重复 source article_id 的 ID 数", f"{duplicate_article_id_count:,}"],
        ["重复 source article_id 额外记录数", f"{duplicate_article_id_extra_records:,}"],
        ["训练/测试 article_id 交集", f"{len(cross_train_test_ids):,}"],
    ], columns=["检查项", "结果"])

    test_overview = pd.DataFrame([
        ["测试文件", str(test_path) if test_path.exists() else "未找到"],
        ["测试文章记录数", f"{test_df.article_key.nunique():,}" if not test_df.empty and "article_key" in test_df else (f"{test_df.article_id.nunique():,}" if not test_df.empty else "-")],
        ["测试不同 source article_id 数", f"{test_df.article_id.nunique():,}" if not test_df.empty else "-"],
        ["测试句子数", f"{len(test_df):,}" if not test_df.empty else "-"],
        ["测试领域数", f"{test_df.domain.nunique():,}" if not test_df.empty else "-"],
        ["测试句子词数分布", json.dumps(describe_lengths(test_df.text), ensure_ascii=False) if not test_df.empty else "-"],
    ], columns=["检查项", "结果"])

    train_sample_cols = [c for c in ["article_id", "article_key", "sentence_index", "domain", "text", "label"] if c in train_df.columns]
    test_sample_cols = [c for c in ["article_id", "article_key", "sentence_index", "domain", "text"] if c in test_df.columns]
    sample_train = train_df[train_sample_cols].head(12)
    sample_test = test_df[test_sample_cols].head(12) if not test_df.empty else pd.DataFrame()
    warnings = []
    if non_mixed_articles:
        warnings.append(f"有 {non_mixed_articles} 篇训练文章并非同时包含 human 和 machine，请核对任务数据假设。")
    if duplicate_sentence_count:
        warnings.append(f"发现 {duplicate_sentence_count} 条重复句子；重复文本不自动删除，请检查是否跨文章或跨划分重复。")
    if duplicate_article_id_count:
        warnings.append(
            f"训练集中有 {duplicate_article_id_count} 个 source article_id 对应多条 JSON 文章记录；"
            "程序已使用 article_key 区分记录，并在数据切分时将相同 source article_id 放在同一分组。"
        )
    if cross_train_test_ids:
        warnings.append(
            f"训练与测试有 {len(cross_train_test_ids)} 个相同 source article_id。"
            "由于训练文件本身存在重复 ID，仅凭 ID 相同不能判定数据泄漏，建议进一步比较文章文本内容。"
        )
    if train_df.text.str.len().max() > 10000:
        warnings.append("存在超长句子，请检查 tokenizer 截断设置。")
    warning_html = "".join(f"<li>{html.escape(x)}</li>" for x in warnings) or "<li>未发现上述基础结构问题。</li>"

    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AIGTD 数据集审计报告</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;line-height:1.6;margin:32px auto;padding:0 20px;max-width:1180px;color:#202938;background:#f7f8fa}}
h1,h2{{color:#17324d}} .panel{{background:white;border:1px solid #e2e7ee;border-radius:12px;padding:20px;margin:18px 0;overflow-x:auto}}
.data-table{{border-collapse:collapse;width:100%;font-size:14px}} .data-table th,.data-table td{{border-bottom:1px solid #e8edf2;padding:9px 11px;text-align:left;vertical-align:top;word-break:break-word}}
.data-table th{{background:#f1f5f9}} code{{background:#eef2f7;padding:2px 5px;border-radius:4px}} .muted{{color:#64748b}}
</style></head><body>
<h1>AIGTD 数据集审计报告</h1><p class="muted">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · 训练格式：<code>id + sent_and_label + domain</code> · 测试格式：<code>id + sent + domain</code></p>
<div class="panel"><h2>1. 数据概况</h2>{make_table(overview, '训练集概况')}{make_table(test_overview, '无标签测试集概况')}</div>
<div class="panel"><h2>2. 标签分布</h2>{make_table(label_counts, '训练集 Human/Machine 分布')}{make_table(domain_counts, '领域与标签分布')}</div>
<div class="panel"><h2>3. 文章分布</h2>{make_table(train_article_df.groupby('domain', as_index=False).agg(article_count=('article_id','count'), sentence_count=('num_sentences','sum'), mean_sentences=('num_sentences','mean')).round(2), '训练集各领域文章统计')}</div>
<div class="panel"><h2>4. 结构与质量检查</h2><ul>{warning_html}</ul>{make_table(duplicate_id_details, '重复 source article_id 示例（最多 30 个）')}<p>注意：审计报告只提示潜在问题，不会自动删除重复文本或改变标签。</p></div>
<div class="panel"><h2>5. 训练数据样本</h2>{make_table(sample_train, '前 12 条句子记录')}</div>
<div class="panel"><h2>6. 测试数据样本</h2>{make_table(sample_test, '前 12 条无标签句子记录')}</div>
<div class="panel"><h2>7. 数据结构约定</h2><p>训练集每篇文章通过 <code>sent_and_label</code> 提供 <code>[sentence, label]</code>；测试集通过 <code>sent</code> 提供纯句子列表。内部统一转换为逐句表格，保留原始 <code>article_id</code>，并为每条 JSON 文章记录生成唯一 <code>article_key</code>，同时保留 <code>sentence_index</code> 和 <code>domain</code>。测试集不会生成或推断真实标签。</p></div>
</body></html>"""
    output.write_text(html_text, encoding="utf-8")

    print("AIGTD 数据审计完成")
    print(
        f"训练：{train_df[article_group_col].nunique():,} 条文章记录，"
        f"{train_df.article_id.nunique():,} 个不同 source article_id，{len(train_df):,} 句"
    )
    print(f"标签：\n{train_df.label.value_counts().to_string()}")
    if not test_df.empty:
        test_group_col = "article_key" if "article_key" in test_df.columns else "article_id"
        print(
            f"测试：{test_df[test_group_col].nunique():,} 条文章记录，"
            f"{test_df.article_id.nunique():,} 个不同 source article_id，{len(test_df):,} 句（无标签）"
        )
    print(f"HTML 报告：{output}")
    if warnings:
        print("潜在问题：")
        for w in warnings:
            print(" -", w)


if __name__ == "__main__":
    main()
