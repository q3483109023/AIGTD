"""Shared AIGTD data, path, reproducibility, and validation-split utilities."""
from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_TO_ID = {"human": 0, "machine": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "experiment.yaml"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def cfg_path(cfg: Dict[str, Any], key: str, default: str) -> Path:
    return resolve_path(cfg.get("paths", {}).get(key, default))


def ensure_project_dirs(cfg: Optional[Dict[str, Any]] = None) -> None:
    cfg = cfg or {}
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


def normalize_sentence(value: Any) -> str:
    """Match audit_duplicates.py: Unicode NFKC + casefold + whitespace normalization.

    Punctuation and word order are intentionally retained. Empty/NA values map
    to an empty string and are not considered novel evaluation examples.
    """
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def mark_novel_validation(
    validation_df: pd.DataFrame,
    train_df: pd.DataFrame,
    text_col: str = "text",
) -> pd.DataFrame:
    """Add `is_novel_validation`: normalized sentence does not occur in train.

    Context is not altered. The flag only selects which target rows contribute
    to Novel Validation metrics, so context-model inputs remain full sentences
    in their original articles.
    """
    if text_col not in validation_df.columns or text_col not in train_df.columns:
        raise ValueError(f"训练集和验证集都必须包含文本字段 `{text_col}`。")
    train_norm = {
        normalize_sentence(text)
        for text in train_df[text_col].tolist()
        if normalize_sentence(text)
    }
    result = validation_df.copy()
    normalized_val = result[text_col].map(normalize_sentence)
    result["is_novel_validation"] = normalized_val.map(
        lambda value: bool(value) and value not in train_norm
    )
    return result


def novel_validation_summary(
    validation_df: pd.DataFrame,
    train_df: pd.DataFrame,
    text_col: str = "text",
) -> Dict[str, Any]:
    marked = mark_novel_validation(validation_df, train_df, text_col=text_col)
    mask = marked["is_novel_validation"].to_numpy(dtype=bool)
    return {
        "full_validation_count": int(len(marked)),
        "novel_validation_count": int(mask.sum()),
        "seen_in_train_validation_count": int((~mask).sum()),
        "novel_validation_ratio": float(mask.mean()) if len(mask) else 0.0,
    }


def _load_json_records(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在：{path}\n请检查 configs/experiment.yaml 中的路径。")
    with path.open("r", encoding="utf-8") as file:
        obj = json.load(file)
    if isinstance(obj, dict):
        for key in ("data", "records", "articles"):
            if key in obj and isinstance(obj[key], list):
                obj = obj[key]
                break
    if not isinstance(obj, list):
        raise ValueError(
            f"{path} 的顶层 JSON 必须是文章对象组成的数组，当前类型：{type(obj).__name__}"
        )
    for index, record in enumerate(obj):
        if not isinstance(record, dict):
            raise ValueError(f"第 {index} 条记录不是 JSON object。")
    return obj


def _validate_article_id(record: Dict[str, Any], index: int, path: Path) -> str:
    if "id" not in record or record["id"] is None:
        raise ValueError(f"{path} 第 {index} 篇文章缺少 id 字段。")
    return str(record["id"])


def parse_train_json(path: str | Path) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Parse [{id, sent_and_label:[[sentence,label],...], domain}, ...]."""
    path = Path(path)
    records = _load_json_records(path)
    rows: List[Dict[str, Any]] = []
    article_meta: List[Dict[str, Any]] = []
    invalid: List[str] = []

    for article_idx, article in enumerate(records):
        article_id = _validate_article_id(article, article_idx, path)
        article_key = f"record_{article_idx:08d}"
        pairs = article.get("sent_and_label")
        if not isinstance(pairs, list):
            invalid.append(f"文章 {article_id} 缺少 list 类型的 sent_and_label 字段")
            continue
        domain = str(article.get("domain") or "unknown")
        parsed_rows: List[Dict[str, Any]] = []
        for sent_idx, pair in enumerate(pairs):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                invalid.append(
                    f"文章 {article_id} 第 {sent_idx} 个句子标签项不是 [sentence, label] 结构"
                )
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
                invalid.append(
                    f"文章 {article_id} 第 {sent_idx} 个标签无效：{label!r}（只允许 human/machine）"
                )
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
            parsed_rows.append(row)
        article_meta.append({
            "article_id": article_id,
            "article_key": article_key,
            "article_order": article_idx,
            "domain": domain,
            "num_sentences": len(parsed_rows),
            "num_human": sum(row["label_id"] == 0 for row in parsed_rows),
            "num_machine": sum(row["label_id"] == 1 for row in parsed_rows),
        })

    if invalid:
        preview = "\n".join(f"- {item}" for item in invalid[:30])
        suffix = f"\n……另有 {len(invalid) - 30} 项" if len(invalid) > 30 else ""
        raise ValueError(
            f"训练数据解析发现 {len(invalid)} 个结构/内容问题，已停止，避免静默丢弃样本：\n"
            f"{preview}{suffix}"
        )
    if not rows:
        raise ValueError("训练集没有解析出任何有效句子。")
    return pd.DataFrame(rows), article_meta


def parse_test_json(path: str | Path) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Parse [{id, sent:[sentence,...], domain}, ...] without labels."""
    path = Path(path)
    records = _load_json_records(path)
    rows: List[Dict[str, Any]] = []
    article_meta: List[Dict[str, Any]] = []
    invalid: List[str] = []

    for article_idx, article in enumerate(records):
        article_id = _validate_article_id(article, article_idx, path)
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
        preview = "\n".join(f"- {item}" for item in invalid[:30])
        suffix = f"\n……另有 {len(invalid) - 30} 项" if len(invalid) > 30 else ""
        raise ValueError(
            f"测试数据解析发现 {len(invalid)} 个结构/内容问题，已停止，避免静默丢弃样本：\n"
            f"{preview}{suffix}"
        )
    return pd.DataFrame(rows), article_meta


def build_context_text(df: pd.DataFrame, window: int = 1) -> List[str]:
    """Build target-centered context without crossing JSON article records."""
    if window < 0:
        raise ValueError("context window 不能小于 0。")
    required = {"article_id", "sentence_index", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"构造上下文缺少字段：{sorted(missing)}")
    group_col = "article_key" if "article_key" in df.columns else "article_id"
    result: Dict[Tuple[str, int], str] = {}
    for record_key, group in df.groupby(group_col, sort=False):
        group = group.sort_values("sentence_index", kind="stable")
        texts = group["text"].astype(str).tolist()
        indices = group["sentence_index"].astype(int).tolist()
        for position, (sentence_index, target) in enumerate(zip(indices, texts)):
            left = texts[max(0, position - window):position]
            right = texts[position + 1:position + 1 + window]
            left_text = " ".join(left) if left else "[START]"
            right_text = " ".join(right) if right else "[END]"
            result[(str(record_key), int(sentence_index))] = (
                f"Previous context: {left_text} [TARGET] {target} [NEXT] {right_text}"
            )
    return [
        result[(str(getattr(row, group_col)), int(row.sentence_index))]
        for row in df.itertuples()
    ]


def append_metrics(metrics_row: Dict[str, Any], results_dir: str | Path) -> None:
    """Append metrics, safely handling a missing, empty, or stale CSV file."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "metrics.csv"
    frame = pd.DataFrame([metrics_row])
    old = pd.DataFrame()

    if path.exists() and path.stat().st_size > 0:
        try:
            old = pd.read_csv(path, encoding="utf-8-sig")
        except pd.errors.EmptyDataError:
            old = pd.DataFrame()

    # Re-running the same experiment replaces its prior row instead of
    # accumulating duplicate/stale results.
    if not old.empty and "experiment" in old.columns:
        old = old[old["experiment"].astype(str) != str(metrics_row.get("experiment", ""))]

    if old.empty:
        combined = frame
    else:
        combined = pd.concat([old, frame], ignore_index=True, sort=False)
    combined.to_csv(path, index=False, encoding="utf-8-sig")


def check_group_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Fail if source article IDs overlap between training and validation."""
    if "article_id" not in train_df or "article_id" not in val_df:
        raise ValueError("训练集和验证集均必须包含 article_id 字段。")
    train_ids = set(train_df["article_id"].astype(str))
    val_ids = set(val_df["article_id"].astype(str))
    overlap = train_ids & val_ids
    if overlap:
        examples = sorted(overlap)[:10]
        raise RuntimeError(
            f"发现文章级数据泄漏：train/validation 重叠 {len(overlap)} 个 article_id，"
            f"例如 {examples}"
        )
