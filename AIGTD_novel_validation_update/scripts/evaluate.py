"""AIGTD unified evaluation: Full Validation + Novel Validation.

Novel Validation contains validation target sentences whose normalized text
(NFKC + casefold + whitespace normalization; punctuation retained) does not
occur anywhere in the training split.

The optimized threshold is selected ONLY on Novel Validation Macro-F1.
Full Validation metrics are reported at that same threshold as a secondary
reference. The full article/context sequence is never filtered or modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    from scripts.common import (
        append_metrics,
        cfg_path,
        load_config,
        mark_novel_validation,
        resolve_path,
    )
except ImportError:  # Support: python scripts/evaluate.py
    from common import append_metrics, cfg_path, load_config, mark_novel_validation, resolve_path


def calculate_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    """Calculate the five project metrics using human=0, machine=1.

    Macro-F1 explicitly averages over both labels [0, 1], even if one label is
    absent from a subset. Precision/Recall use machine=1 as the positive class.
    In single-label binary classification, Micro-F1 equals Accuracy.
    """
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    if len(y_true) == 0:
        raise ValueError("不能对空标签数组计算指标。")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 与 y_pred 长度不一致。")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("y_true 只允许 0/1（human=0, machine=1）。")
    if not set(np.unique(y_pred)).issubset({0, 1}):
        raise ValueError("y_pred 只允许 0/1（human=0, machine=1）。")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="micro", zero_division=0)),
    }


def _prefixed_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in calculate_metrics(y_true, y_pred).items()}


def search_threshold(
    y_true: Iterable[int],
    p_machine: Iterable[float],
    novel_mask: Optional[Iterable[bool]] = None,
    min_threshold: float = 0.30,
    max_threshold: float = 0.80,
    step: float = 0.01,
) -> Tuple[float, pd.DataFrame]:
    """Search threshold by Novel Validation Macro-F1.

    Each candidate threshold is also evaluated on Full Validation, but the
    sorting/selection criterion is novel_macro_f1, then novel_accuracy, then
    closeness to 0.5.
    """
    y_full = np.asarray(list(y_true), dtype=int)
    probabilities = np.asarray(list(p_machine), dtype=float)
    if len(y_full) != len(probabilities) or len(y_full) == 0:
        raise ValueError("y_true 和 p_machine 长度必须相同且非空。")
    if not np.isfinite(probabilities).all():
        raise ValueError("p_machine 包含 NaN 或 Inf。")
    if novel_mask is None:
        mask = np.ones(len(y_full), dtype=bool)
    else:
        mask = np.asarray(list(novel_mask), dtype=bool)
    if len(mask) != len(y_full):
        raise ValueError("novel_mask 长度必须与 y_true 相同。")
    if not mask.any():
        raise ValueError("Novel Validation 为空：验证集中所有句子都已在训练集中出现。")
    if len(np.unique(y_full[mask])) < 2:
        raise ValueError(
            "Novel Validation 只包含一个真实类别，无法可靠地用其 Macro-F1 选择阈值。"
            "请检查划分或新颖样本标签分布。"
        )
    if not (0.0 < min_threshold < max_threshold < 1.0) or step <= 0:
        raise ValueError("阈值范围或步长不合法。")

    probabilities = np.clip(probabilities, 0.0, 1.0)
    thresholds = np.arange(min_threshold, max_threshold + step / 2.0, step)
    records = []
    y_novel = y_full[mask]
    p_novel = probabilities[mask]
    for threshold in thresholds:
        pred_full = (probabilities >= threshold).astype(int)
        pred_novel = pred_full[mask]
        record = {"threshold": float(threshold)}
        record.update(_prefixed_metrics(y_novel, pred_novel, "novel"))
        record.update(_prefixed_metrics(y_full, pred_full, "full"))
        records.append(record)

    table = pd.DataFrame(records)
    table["distance_to_half"] = (table["threshold"] - 0.5).abs()
    best = table.sort_values(
        ["novel_macro_f1", "novel_accuracy", "distance_to_half"],
        ascending=[False, False, True],
        kind="stable",
    ).iloc[0]
    table = table.drop(columns=["distance_to_half"])
    return float(best["threshold"]), table


def plot_confusion(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    output_path: str | Path,
    title: str,
) -> None:
    cm = confusion_matrix(list(y_true), list(y_pred), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["human", "machine"], yticklabels=["human", "machine"],
        ylabel="True label", xlabel="Predicted label", title=title,
    )
    midpoint = cm.max() / 2.0 if cm.size else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col, row, f"{cm[row, col]:,}", ha="center", va="center",
                color="white" if cm[row, col] > midpoint else "black",
            )
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_threshold(table: pd.DataFrame, output_path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(table["threshold"], table["novel_macro_f1"], label="Novel Macro-F1", linewidth=2.2)
    ax.plot(table["threshold"], table["novel_accuracy"], label="Novel Accuracy", alpha=0.85)
    ax.plot(table["threshold"], table["novel_precision"], label="Novel Precision", alpha=0.75)
    ax.plot(table["threshold"], table["novel_recall"], label="Novel Recall", alpha=0.75)
    ax.plot(
        table["threshold"], table["full_macro_f1"],
        label="Full Macro-F1 (same threshold)", linestyle="--", alpha=0.8,
    )
    ax.set_xlabel("Machine probability / score threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_validation(
    frame: pd.DataFrame,
    probabilities: Iterable[float],
    experiment: str,
    results_dir: str | Path,
    threshold_min: float = 0.30,
    threshold_max: float = 0.80,
    threshold_step: float = 0.01,
    optimize_threshold: bool = True,
    default_threshold: float = 0.50,
    train_df: Optional[pd.DataFrame] = None,
) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
    """Evaluate on Full + Novel Validation and select threshold on Novel only.

    `train_df` is preferred and used to construct the novel mask. If omitted,
    frame must already contain the boolean `is_novel_validation` column.
    """
    if "label_id" in frame.columns:
        y_true = frame["label_id"].astype(int).to_numpy()
    elif "y_true" in frame.columns:
        y_true = frame["y_true"].astype(int).to_numpy()
    else:
        raise ValueError("验证数据缺少 label_id 或 y_true。")

    probabilities = np.asarray(list(probabilities), dtype=float)
    if len(probabilities) != len(frame):
        raise ValueError(
            f"预测数量 {len(probabilities)} 与验证句子数量 {len(frame)} 不一致。"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("预测分数包含 NaN 或 Inf。")
    probabilities = np.clip(probabilities, 0.0, 1.0)

    if train_df is not None:
        marked_frame = mark_novel_validation(frame, train_df)
    elif "is_novel_validation" in frame.columns:
        marked_frame = frame.copy()
    else:
        raise ValueError(
            "无法计算 Novel Validation：请传入 train_df，或在 frame 中提供 is_novel_validation。"
        )
    novel_mask = marked_frame["is_novel_validation"].astype(bool).to_numpy()
    if not novel_mask.any():
        raise ValueError("Novel Validation 为空，无法执行 Novel Macro-F1 阈值选择。")

    y_novel = y_true[novel_mask]
    if len(np.unique(y_novel)) < 2:
        raise ValueError(
            "Novel Validation 只包含一个真实类别，无法可靠选择阈值。"
            "请先检查 novel 样本的标签分布。"
        )

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if optimize_threshold:
        threshold, threshold_table = search_threshold(
            y_true=y_true,
            p_machine=probabilities,
            novel_mask=novel_mask,
            min_threshold=threshold_min,
            max_threshold=threshold_max,
            step=threshold_step,
        )
    else:
        threshold = float(default_threshold)
        pred = (probabilities >= threshold).astype(int)
        record = {"threshold": threshold}
        record.update(_prefixed_metrics(y_novel, pred[novel_mask], "novel"))
        record.update(_prefixed_metrics(y_true, pred, "full"))
        threshold_table = pd.DataFrame([record])

    y_pred = (probabilities >= threshold).astype(int)
    full_metrics = calculate_metrics(y_true, y_pred)
    novel_metrics = calculate_metrics(y_novel, y_pred[novel_mask])

    # Unprefixed keys intentionally represent the primary metric set (Novel).
    # Prefixed keys preserve both evaluation views in metrics.csv/json.
    metrics: Dict[str, Any] = {
        **full_metrics,
        **{f"full_{key}": value for key, value in full_metrics.items()},
        **{f"novel_{key}": value for key, value in novel_metrics.items()},
        "primary_metric_set": "novel_validation",
        "threshold_selection_metric": "novel_macro_f1",
        "full_validation_count": int(len(frame)),
        "novel_validation_count": int(novel_mask.sum()),
        "seen_in_train_validation_count": int((~novel_mask).sum()),
        "novel_validation_ratio": float(novel_mask.mean()),
        "full_human_count": int((y_true == 0).sum()),
        "full_machine_count": int((y_true == 1).sum()),
        "novel_human_count": int((y_novel == 0).sum()),
        "novel_machine_count": int((y_novel == 1).sum()),
    }
    # For compatibility with existing logs and printed summaries, the
    # unprefixed metric names are changed to the primary Novel Validation view.
    metrics.update(novel_metrics)

    predictions = marked_frame.copy().reset_index(drop=True)
    predictions["y_true"] = y_true
    predictions["p_machine"] = probabilities
    predictions["y_pred"] = y_pred
    predictions["predicted_label"] = np.where(y_pred == 1, "machine", "human")
    predictions["is_correct"] = y_true == y_pred
    predictions["evaluation_subset"] = np.where(novel_mask, "novel", "seen_in_train")
    if "label" not in predictions.columns:
        predictions["label"] = np.where(y_true == 1, "machine", "human")

    safe_name = experiment.replace("/", "_").replace(" ", "_")
    predictions.to_csv(
        results_dir / f"{safe_name}_predictions.csv", index=False, encoding="utf-8-sig"
    )
    predictions.to_csv(results_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    predictions.loc[novel_mask].to_csv(
        results_dir / f"{safe_name}_novel_predictions.csv", index=False, encoding="utf-8-sig"
    )
    threshold_table.to_csv(
        results_dir / f"{safe_name}_thresholds.csv", index=False, encoding="utf-8-sig"
    )

    metrics_record: Dict[str, Any] = {
        "experiment": experiment,
        "threshold": threshold,
        **metrics,
    }
    append_metrics(metrics_record, results_dir)
    pd.DataFrame([metrics_record]).to_csv(
        results_dir / f"{safe_name}_metrics.csv", index=False, encoding="utf-8-sig"
    )

    metrics_json = {
        "experiment": experiment,
        "threshold": threshold,
        "threshold_selection_metric": "novel_macro_f1",
        "normalization": "Unicode NFKC + casefold + whitespace normalization; punctuation retained",
        **metrics,
    }
    (results_dir / f"{safe_name}_metrics.json").write_text(
        json.dumps(metrics_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_confusion(
        y_true, y_pred, results_dir / "confusion_matrix_full.png",
        f"{experiment} - Full Validation (t={threshold:.3f})",
    )
    plot_confusion(
        y_novel, y_pred[novel_mask], results_dir / "confusion_matrix_novel.png",
        f"{experiment} - Novel Validation (t={threshold:.3f})",
    )
    # Keep the original canonical filename pointing to the primary Novel view.
    plot_confusion(
        y_novel, y_pred[novel_mask], results_dir / "confusion_matrix.png",
        f"{experiment} - Novel Validation (t={threshold:.3f})",
    )
    plot_threshold(
        threshold_table, results_dir / "threshold_analysis.png",
        f"{experiment}: threshold search on Novel Validation",
    )

    print(
        f"[{experiment}] Full Validation: {len(frame):,} | "
        f"Novel Validation: {int(novel_mask.sum()):,} "
        f"({novel_mask.mean():.2%}) | seen-in-train: {int((~novel_mask).sum()):,}"
    )
    return threshold, metrics, predictions


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate predictions and report Full/Novel Validation metrics."
    )
    parser.add_argument("--predictions", default="results/predictions.csv")
    parser.add_argument("--train", default="data/splits/train.csv", help="training split CSV for novelty matching")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--no-threshold-search", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pred_path = resolve_path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(f"预测文件不存在：{pred_path}")
    frame = pd.read_csv(pred_path, encoding="utf-8-sig")
    results_dir = cfg_path(cfg, "results_dir", "results")
    train_path = resolve_path(args.train)
    train_df = None
    if train_path.exists():
        train_df = pd.read_csv(train_path, encoding="utf-8-sig")

    if "p_machine" in frame.columns and ("y_true" in frame.columns or "label_id" in frame.columns):
        threshold_cfg = cfg.get("threshold", {})
        threshold, metrics, _ = evaluate_validation(
            frame=frame,
            probabilities=frame["p_machine"].to_numpy(),
            experiment="manual_evaluation",
            results_dir=results_dir,
            threshold_min=float(threshold_cfg.get("min", 0.30)),
            threshold_max=float(threshold_cfg.get("max", 0.80)),
            threshold_step=float(threshold_cfg.get("step", 0.01)),
            optimize_threshold=not args.no_threshold_search,
            default_threshold=float(threshold_cfg.get("default", 0.50)),
            train_df=train_df,
        )
        print(f"Novel Validation best threshold: {threshold:.4f}")
        print(pd.Series(metrics).round(4).to_string())
    elif "y_true" in frame.columns and "y_pred" in frame.columns:
        print("该 CSV 只有离散预测，无法搜索阈值；以下为整表指标：")
        print(pd.Series(calculate_metrics(frame["y_true"], frame["y_pred"])).round(4).to_string())
    else:
        raise ValueError("CSV 至少需要 y_true 与 p_machine，或 y_true 与 y_pred。")


if __name__ == "__main__":
    _cli()
