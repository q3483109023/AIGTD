"""Shared data parsing, configuration, path, and reproducibility utilities for AIGTD."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_TO_ID = {"human": 0, "machine": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "experiment.yaml"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def cfg_path(cfg: Dict[str, Any], key: str, default: str) -> Path:
    return resolve_path(cfg.get("paths", {}).get(key, default))


def ensure_project_dirs(cfg: Optional[Dict[str, Any]] = None) -> None:
    if cfg is None:
        cfg = {}
    defaults = {
        "processed_dir": "data/processed",
        "splits_dir": "data/splits",
        "results_dir": "results",
        "checkpoints_dir": "models/checkpoints",
        "best_model_dir": "models/best_model",
    }
    for key, default in defaults.items():
        cfg_path(cfg, key, default).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _load_json_records(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在：{path}\n请检查 configs/experiment.yaml 中的路径。")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        # Accommodate common wrappers without silently interpreting arbitrary objects.
        for key in ("data", "records", "articles"):
            if key in obj and isinstance(obj[key], list):
                obj = obj[key]
                break
    if not isinstance(obj, list):
        raise ValueError(f"{path} 的顶层 JSON 必须是文章对象组成的数组，当前类型：{type(obj).__name__}")
    for i, record in enumerate(obj):
        if not isinstance(record, dict):
            raise ValueError(f"第 {i} 条记录不是 JSON object。")
    return obj


def _validate_article_id(record: Dict[str, Any], index: int, path: Path) -> str:
    if "id" not in record or record["id"] is None:
        raise ValueError(f"{path} 第 {index} 篇文章缺少 id 字段。")
    return str(record["id"])


def parse_train_json(path: str | Path) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Parse training format: [{id, sent_and_label:[[text,label],...], domain}, ...]."""
    path = Path(path)
    records = _load_json_records(path)
    rows: List[Dict[str, Any]] = []
    article_meta: List[Dict[str, Any]] = []
    invalid: List[str] = []

    for article_idx, article in enumerate(records):
        article_id = _validate_article_id(article, article_idx, path)
        # The source `id` is not guaranteed to be unique in this sampled dataset.
        # Keep it unchanged and create a unique key for each JSON article record.
        article_key = f"record_{article_idx:08d}"
        pairs = article.get("sent_and_label")
        if not isinstance(pairs, list):
            invalid.append(f"文章 {article_id} 缺少 list 类型的 sent_and_label 字段")
            continue
        domain = str(article.get("domain") or "unknown")
        parsed_article_rows = []
        for sent_idx, pair in enumerate(pairs):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                invalid.append(f"文章 {article_id} 第 {sent_idx} 个句子标签项不是 [sentence, label] 结构")
                continue
            text, label = pair
            if not isinstance(text, str):
                invalid.append(f"文章 {article_id} 第 {sent_idx} 个句子不是字符串")
                continue
            text = text.strip()
            if not text:
                invalid.append(f"文章 {article_id} 第 {sent_idx} 个句子为空")
                continue
            normalized_label = str(label).strip().lower()
            if normalized_label not in LABEL_TO_ID:
                invalid.append(f"文章 {article_id} 第 {sent_idx} 个标签无效：{label!r}（只允许 human/machine）")
                continue
            row = {
                "article_id": article_id,
                "article_key": article_key,
                "sentence_index": sent_idx,
                "text": text,
                "label": normalized_label,
                "label_id": LABEL_TO_ID[normalized_label],
                "domain": domain,
                "article_length": len(pairs),
                "article_order": article_idx,
            }
            rows.append(row)
            parsed_article_rows.append(row)
        article_meta.append({
            "article_id": article_id,
            "article_key": article_key,
            "article_order": article_idx,
            "domain": domain,
            "num_sentences": len(parsed_article_rows),
            "num_human": sum(r["label_id"] == 0 for r in parsed_article_rows),
            "num_machine": sum(r["label_id"] == 1 for r in parsed_article_rows),
        })

    if invalid:
        preview = "\n".join(f"- {x}" for x in invalid[:30])
        suffix = f"\n……另有 {len(invalid) - 30} 项" if len(invalid) > 30 else ""
        raise ValueError(
            f"训练数据解析发现 {len(invalid)} 个结构/内容问题，已停止，避免静默丢弃样本：\n{preview}{suffix}"
        )
    if not rows:
        raise ValueError("训练集没有解析出任何有效句子。")
    return pd.DataFrame(rows), article_meta


def parse_test_json(path: str | Path) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Parse test format: [{id, sent:[text,...], domain}, ...], with no labels."""
    path = Path(path)
    records = _load_json_records(path)
    rows: List[Dict[str, Any]] = []
    article_meta: List[Dict[str, Any]] = []
    invalid: List[str] = []

    for article_idx, article in enumerate(records):
        article_id = _validate_article_id(article, article_idx, path)
        # The source `id` may repeat; article_key identifies this record occurrence.
        article_key = f"record_{article_idx:08d}"
        sentences = article.get("sent")
        if not isinstance(sentences, list):
            invalid.append(f"文章 {article_id} 缺少 list 类型的 sent 字段")
            continue
        domain = str(article.get("domain") or "unknown")
        parsed_count = 0
        for sent_idx, text in enumerate(sentences):
            if not isinstance(text, str):
                invalid.append(f"文章 {article_id} 第 {sent_idx} 个句子不是字符串")
                continue
            text = text.strip()
            if not text:
                invalid.append(f"文章 {article_id} 第 {sent_idx} 个句子为空")
                continue
            rows.append({
                "article_id": article_id,
                "article_key": article_key,
                "sentence_index": sent_idx,
                "text": text,
                "domain": domain,
                "article_length": len(sentences),
                "article_order": article_idx,
            })
            parsed_count += 1
        article_meta.append({
            "article_id": article_id,
            "article_key": article_key,
            "article_order": article_idx,
            "domain": domain,
            "num_sentences": parsed_count,
        })

    if invalid:
        preview = "\n".join(f"- {x}" for x in invalid[:30])
        suffix = f"\n……另有 {len(invalid) - 30} 项" if len(invalid) > 30 else ""
        raise ValueError(
            f"测试数据解析发现 {len(invalid)} 个结构/内容问题，已停止，避免静默丢弃样本：\n{preview}{suffix}"
        )
    return pd.DataFrame(rows), article_meta


def build_context_text(df: pd.DataFrame, window: int = 1) -> List[str]:
    """Build target-centered context without crossing article boundaries."""
    if window < 0:
        raise ValueError("context window 不能小于 0。")
    required = {"article_id", "sentence_index", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"构造上下文缺少字段：{sorted(missing)}")

    # Group by the unique record key when available. Grouping only by source
    # article_id would accidentally join sentences from separate records that
    # happen to share the same source ID.
    group_col = "article_key" if "article_key" in df.columns else "article_id"
    result: Dict[Tuple[str, int], str] = {}
    for record_key, group in df.groupby(group_col, sort=False):
        group = group.sort_values("sentence_index", kind="stable")
        texts = group["text"].astype(str).tolist()
        indices = group["sentence_index"].astype(int).tolist()
        for pos, (sentence_index, target) in enumerate(zip(indices, texts)):
            left = texts[max(0, pos - window):pos]
            right = texts[pos + 1:pos + 1 + window]
            left_text = " ".join(left) if left else "[START]"
            right_text = " ".join(right) if right else "[END]"
            combined = f"Previous context: {left_text} [TARGET] {target} [NEXT] {right_text}"
            result[(str(record_key), int(sentence_index))] = combined

    return [
        result[(str(getattr(r, group_col)), int(r.sentence_index))]
        for r in df.itertuples()
    ]


def append_metrics(metrics_row: Dict[str, Any], results_dir: str | Path) -> None:
    """Append experiment metrics to CSV, tolerating an empty existing file."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    path = results_dir / "metrics.csv"
    frame = pd.DataFrame([metrics_row])

    old = pd.DataFrame()

    if path.exists():
        try:
            # 空文件或只有空白字符时，不尝试作为 CSV 读取
            if path.stat().st_size > 0:
                old = pd.read_csv(path, encoding="utf-8-sig")

        except pd.errors.EmptyDataError:
            # 兼容已经存在但没有有效 CSV 内容的文件
            old = pd.DataFrame()

        # 同一实验重新运行时，覆盖旧实验记录
        if "experiment" in old.columns:
            old = old[
                old["experiment"].astype(str)
                != str(metrics_row.get("experiment", ""))
            ]

    # 合并历史实验与当前实验结果
    frame = pd.concat(
        [old, frame],
        ignore_index=True,
        sort=False
    )

    # 写入 CSV
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )

def check_group_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    train_ids = set(train_df["article_id"].astype(str))
    val_ids = set(val_df["article_id"].astype(str))
    overlap = train_ids & val_ids
    if overlap:
        raise RuntimeError(f"发现文章级数据泄漏：train/validation 重叠 {len(overlap)} 个 article_id，例如 {list(overlap)[:10]}")
