"""Unified evaluation for AIGTD. Binary convention: human=0, machine=1."""
from __future__ import annotations

import argparse
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
    from scripts.common import PROJECT_ROOT, append_metrics, load_config, resolve_path
except ImportError:  # running as `python scripts/evaluate.py`
    from common import PROJECT_ROOT, append_metrics, load_config, resolve_path


def calculate_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    if len(y_true) == 0:
        raise ValueError("不能对空标签数组计算指标。")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def search_threshold(
    y_true: Iterable[int],
    p_machine: Iterable[float],
    min_threshold: float = 0.30,
    max_threshold: float = 0.80,
    step: float = 0.01,
) -> Tuple[float, pd.DataFrame]:
    y_true = np.asarray(list(y_true), dtype=int)
    probabilities = np.asarray(list(p_machine), dtype=float)
    if len(y_true) != len(probabilities) or len(y_true) == 0:
        raise ValueError("y_true 和 p_machine 长度必须相同且非空。")
    if not (0.0 < min_threshold < max_threshold < 1.0) or step <= 0:
        raise ValueError("阈值范围或步长不合法。")
    thresholds = np.arange(min_threshold, max_threshold + step / 2, step)
    records = []
    for threshold in thresholds:
        pred = (probabilities >= threshold).astype(int)
        records.append({"threshold": float(threshold), **calculate_metrics(y_true, pred)})
    table = pd.DataFrame(records)
    # Macro-F1 primary; tie-break by accuracy, then proximity to 0.5.
    table["distance_to_half"] = (table["threshold"] - 0.5).abs()
    best = table.sort_values(
        ["macro_f1", "accuracy", "distance_to_half"],
        ascending=[False, False, True],
        kind="stable",
    ).iloc[0]
    table = table.drop(columns=["distance_to_half"])
    return float(best["threshold"]), table


def plot_confusion(y_true: Iterable[int], y_pred: Iterable[int], output_path: str | Path, title: str) -> None:
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
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > midpoint else "black")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_threshold(table: pd.DataFrame, output_path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(table["threshold"], table["macro_f1"], label="Macro-F1")
    ax.plot(table["threshold"], table["accuracy"], label="Accuracy", alpha=0.8)
    ax.plot(table["threshold"], table["precision"], label="Precision", alpha=0.8)
    ax.plot(table["threshold"], table["recall"], label="Recall", alpha=0.8)
    ax.set_xlabel("Machine probability threshold")
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
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """Save validation predictions, metrics, threshold table, and plots."""
    if "label_id" in frame:
        y_true = frame["label_id"].astype(int).to_numpy()
    elif "y_true" in frame:
        y_true = frame["y_true"].astype(int).to_numpy()
    else:
        raise ValueError("验证数据缺少 label_id 或 y_true。")
    probabilities = np.asarray(list(probabilities), dtype=float)
    if len(probabilities) != len(frame):
        raise ValueError(f"预测数量 {len(probabilities)} 与验证句子数量 {len(frame)} 不一致。")
    if not np.isfinite(probabilities).all():
        raise ValueError("预测概率包含 NaN 或 Inf。")
    probabilities = np.clip(probabilities, 0.0, 1.0)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if optimize_threshold:
        threshold, threshold_table = search_threshold(
            y_true, probabilities, threshold_min, threshold_max, threshold_step
        )
    else:
        threshold = float(default_threshold)
        threshold_table = pd.DataFrame([
            {"threshold": threshold, **calculate_metrics(y_true, (probabilities >= threshold).astype(int))}
        ])
    y_pred = (probabilities >= threshold).astype(int)
    metrics = calculate_metrics(y_true, y_pred)

    predictions = frame.copy().reset_index(drop=True)
    predictions["y_true"] = y_true
    predictions["p_machine"] = probabilities
    predictions["y_pred"] = y_pred
    predictions["predicted_label"] = np.where(y_pred == 1, "machine", "human")
    predictions["is_correct"] = y_true == y_pred
    if "label" not in predictions.columns:
        predictions["label"] = np.where(y_true == 1, "machine", "human")

    safe_name = experiment.replace("/", "_").replace(" ", "_")
    predictions_path = results_dir / f"{safe_name}_predictions.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    # Keep the requested canonical file pointing at the most recently evaluated model.
    predictions.to_csv(results_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    threshold_table.to_csv(results_dir / f"{safe_name}_thresholds.csv", index=False, encoding="utf-8-sig")

    metrics_record: Dict[str, Any] = {"experiment": experiment, "threshold": threshold, **metrics}
    append_metrics(metrics_record, results_dir)
    pd.DataFrame([metrics_record]).to_csv(results_dir / f"{safe_name}_metrics.csv", index=False, encoding="utf-8-sig")
    plot_confusion(y_true, y_pred, results_dir / "confusion_matrix.png", f"{experiment} (threshold={threshold:.2f})")
    plot_threshold(threshold_table, results_dir / "threshold_analysis.png", f"{experiment}: validation threshold search")
    return threshold, metrics, predictions


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CSV containing y_true and p_machine / y_pred.")
    parser.add_argument("--predictions", default="results/predictions.csv", help="prediction CSV")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--no-threshold-search", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    path = resolve_path(args.predictions)
    frame = pd.read_csv(path)
    results_dir = resolve_path(cfg.get("paths", {}).get("results_dir", "results"))
    if "p_machine" in frame.columns and ("y_true" in frame.columns or "label_id" in frame.columns):
        threshold_cfg = cfg.get("threshold", {})
        threshold, metrics, _ = evaluate_validation(
            frame,
            frame["p_machine"].to_numpy(),
            experiment="manual_evaluation",
            results_dir=results_dir,
            threshold_min=float(threshold_cfg.get("min", 0.30)),
            threshold_max=float(threshold_cfg.get("max", 0.80)),
            threshold_step=float(threshold_cfg.get("step", 0.01)),
            optimize_threshold=not args.no_threshold_search,
            default_threshold=float(threshold_cfg.get("default", 0.50)),
        )
        print(f"Best threshold: {threshold:.4f}")
        print(pd.Series(metrics).to_string())
    elif "y_true" in frame.columns and "y_pred" in frame.columns:
        metrics = calculate_metrics(frame["y_true"], frame["y_pred"])
        print(pd.Series(metrics).round(4).to_string())
    else:
        raise ValueError("CSV 至少需要 y_true 与 p_machine，或 y_true 与 y_pred 两组字段。")


if __name__ == "__main__":
    _cli()
