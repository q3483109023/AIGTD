#!/usr/bin/env python3
"""Train and evaluate leakage-aware OOF stacking and SVM-veto fusion.

Design:
- Base OOF rows are produced by exactly one fold model per training row.
- Fold-specific score standardization is fitted on that fold's OOF validation
  score distribution and reused for the matching fold's Novel predictions.
- Meta-level cross-fitting holds out one original OOF fold at a time and purges
  exact normalized text overlap from meta-training rows.
- Novel rows are deterministically assigned by source_article_id to one fold;
  the matching base fold model and matching cross-fitted meta-model score it.
  This avoids training on one-fold OOF features and evaluating on a five-fold
  averaged feature vector.
- All thresholds and the SVM-veto rule are selected using OOF training labels
  only. Novel labels are used only once for final reporting.

This is an experimental stacking protocol, not a claim of an untouched test
set: the original model development already used the existing validation set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def normalize_sentence(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def stable_fold(value: Any, n_splits: int) -> int:
    raw = str(value).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], "big") % n_splits



def assign_source_groups_balanced(values: pd.Series, n_splits: int) -> pd.Series:
    """Deterministically assign whole source groups to balanced folds.

    Uses only group sizes and stable hashes, never labels. Largest groups are
    assigned first to the currently least-loaded fold (LPT scheduling).
    """
    sizes = values.astype(str).value_counts(dropna=False).to_dict()
    ordered = sorted(
        sizes.items(),
        key=lambda kv: (-int(kv[1]), hashlib.sha256(str(kv[0]).encode("utf-8")).hexdigest()),
    )
    loads = [0] * n_splits
    mapping: dict[str, int] = {}
    for source_id, count in ordered:
        fold = min(range(n_splits), key=lambda f: (loads[f], f))
        mapping[str(source_id)] = fold
        loads[fold] += int(count)
    return values.astype(str).map(mapping).astype(int)

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns {missing}; available={list(df.columns)}")


def assert_same(left: pd.DataFrame, right: pd.DataFrame, key: list[str], cols: list[str], label: str) -> None:
    merged = left[key + cols].merge(
        right[key + cols], on=key, how="outer", suffixes=("_l", "_r"),
        indicator=True, validate="one_to_one",
    )
    bad = int((merged["_merge"] != "both").sum())
    if bad:
        raise ValueError(f"{label}: {bad} keys missing on one side")
    both = merged.loc[merged["_merge"] == "both"]
    for col in cols:
        a = both[f"{col}_l"]
        b = both[f"{col}_r"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            ok = np.isclose(pd.to_numeric(a), pd.to_numeric(b), rtol=1e-6, atol=1e-8, equal_nan=True)
        else:
            ok = a.astype("string").fillna("<NA>").to_numpy() == b.astype("string").fillna("<NA>").to_numpy()
        if not bool(np.all(ok)):
            raise ValueError(f"{label}: column {col!r} has {int((~ok).sum())} mismatches")


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if np.unique(y).size == 2 else float("nan")


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if np.unique(y).size == 2 else float("nan")


def metrics_dict(y: np.ndarray, pred: np.ndarray, score: np.ndarray | None = None) -> dict[str, float | int]:
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    out: dict[str, float | int] = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_machine": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall_machine": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "f1_machine": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "precision_human": float(precision_score(y, pred, pos_label=0, zero_division=0)),
        "recall_human": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "f1_human": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    if score is not None:
        out["roc_auc"] = safe_auc(y, score)
        out["average_precision"] = safe_ap(y, score)
    return out


def best_threshold(y: np.ndarray, score: np.ndarray, candidates: np.ndarray, tie_target: float = 0.5) -> tuple[float, float, pd.DataFrame]:
    rows = []
    best_t = float(candidates[0])
    best_f1 = -1.0
    for t in candidates:
        pred = (score >= t).astype(int)
        mf1 = float(f1_score(y, pred, average="macro", zero_division=0))
        rows.append({"threshold": float(t), "macro_f1": mf1,
                     "accuracy": float(accuracy_score(y, pred)),
                     "precision_machine": float(precision_score(y, pred, zero_division=0)),
                     "recall_machine": float(recall_score(y, pred, zero_division=0))})
        # Deterministic tie break: prefer threshold closest to 0.5.
        if mf1 > best_f1 + 1e-12 or (abs(mf1 - best_f1) <= 1e-12 and abs(float(t) - tie_target) < abs(best_t - tie_target)):
            best_f1, best_t = mf1, float(t)
    return best_t, best_f1, pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof-dir", default="results/oof")
    ap.add_argument("--folds-dir", default="data/oof_folds")
    ap.add_argument("--output-dir", default="results/oof_fusion")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--c", type=float, default=1.0, help="LogisticRegression inverse regularization strength")
    ap.add_argument("--veto-grid-size", type=int, default=31)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    oof_dir = Path(args.oof_dir)
    folds_dir = Path(args.folds_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    svm_oof = read_csv(oof_dir / "svm_oof_predictions.csv")
    rob_oof = read_csv(oof_dir / "roberta_oof_predictions.csv")
    svm_novel = read_csv(oof_dir / "svm_novel_fold_predictions.csv")
    rob_novel = read_csv(oof_dir / "roberta_novel_fold_predictions.csv")
    novel_base = read_csv(folds_dir / "novel_validation.csv")

    require_columns(svm_oof, ["oof_row_id", "fold", "article_key", "sentence_index", "text", "label_id", "y_true", "source_article_id", "svm_decision_score"], "SVM OOF")
    require_columns(rob_oof, ["oof_row_id", "fold", "article_key", "sentence_index", "text", "label_id", "y_true", "source_article_id", "roberta_logit_margin"], "RoBERTa OOF")
    require_columns(svm_novel, ["novel_row_id", "fold", "article_key", "sentence_index", "text", "label_id", "y_true", "svm_decision_score"], "SVM Novel")
    require_columns(rob_novel, ["novel_row_id", "fold", "article_key", "sentence_index", "text", "label_id", "y_true", "roberta_logit_margin"], "RoBERTa Novel")
    require_columns(novel_base, ["novel_row_id", "article_key", "sentence_index", "text", "label_id"], "Novel base")

    # Verify base OOF prediction alignment and exactly-once row coverage.
    if svm_oof["oof_row_id"].duplicated().any() or rob_oof["oof_row_id"].duplicated().any():
        raise ValueError("Duplicate oof_row_id detected in base OOF predictions")
    expected_ids = set(range(len(svm_oof)))
    if set(svm_oof["oof_row_id"].astype(int)) != expected_ids or set(rob_oof["oof_row_id"].astype(int)) != expected_ids:
        raise ValueError("OOF row IDs do not cover 0..N-1 exactly")
    assert_same(svm_oof, rob_oof, ["oof_row_id"], ["fold", "article_key", "sentence_index", "text", "label_id", "y_true", "source_article_id"], "SVM/RoBERTa OOF alignment")

    # Verify Novel predictions are aligned by fold + novel_row_id.
    assert_same(svm_novel, rob_novel, ["fold", "novel_row_id"], ["article_key", "sentence_index", "text", "label_id", "y_true"], "SVM/RoBERTa Novel alignment")
    if len(novel_base) != novel_base["novel_row_id"].nunique():
        raise ValueError("Novel base has duplicate novel_row_id")

    oof = svm_oof.merge(
        rob_oof[["oof_row_id", "roberta_logit_margin", "roberta_p_machine"]],
        on="oof_row_id", validate="one_to_one",
    ).copy()
    oof["fold"] = oof["fold"].astype(int)
    oof["y_true"] = oof["y_true"].astype(int)
    oof["norm_text"] = oof["text"].map(normalize_sentence)
    if (oof["norm_text"] == "").any():
        raise ValueError("Empty normalized text in OOF rows")
    if set(oof["fold"].unique()) != set(range(args.n_splits)):
        raise ValueError("OOF fold values do not match --n-splits")

    # Per-fold scaling: fit on the held-out OOF score distribution for that base model,
    # then reuse exactly those parameters for the corresponding Novel fold predictions.
    score_stats: dict[str, dict[str, dict[str, float]]] = {"svm": {}, "roberta": {}}
    for fold in range(args.n_splits):
        part = oof[oof["fold"] == fold]
        for name, col in [("svm", "svm_decision_score"), ("roberta", "roberta_logit_margin")]:
            arr = pd.to_numeric(part[col], errors="coerce").to_numpy(float)
            if not np.isfinite(arr).all():
                raise ValueError(f"Non-finite {col} values in fold {fold}")
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=0))
            if std < 1e-8:
                raise ValueError(f"Near-zero score std for {name}, fold {fold}")
            score_stats[name][str(fold)] = {"mean": mean, "std": std}
            z_col = "svm_z" if name == "svm" else "roberta_z"
            oof.loc[oof["fold"] == fold, z_col] = (arr - mean) / std

    # Build Novel table with each fold's raw score and matching fold-specific z-score.
    novel_parts = []
    for fold in range(args.n_splits):
        s = svm_novel[svm_novel["fold"].astype(int) == fold].copy()
        r = rob_novel[rob_novel["fold"].astype(int) == fold].copy()
        part = s.merge(
            r[["novel_row_id", "fold", "roberta_logit_margin", "roberta_p_machine"]],
            on=["novel_row_id", "fold"], validate="one_to_one",
        )
        sstat = score_stats["svm"][str(fold)]
        rstat = score_stats["roberta"][str(fold)]
        part["svm_z"] = (part["svm_decision_score"].astype(float) - sstat["mean"]) / sstat["std"]
        part["roberta_z"] = (part["roberta_logit_margin"].astype(float) - rstat["mean"]) / rstat["std"]
        part["fold"] = fold
        novel_parts.append(part)
    novel_all_folds = pd.concat(novel_parts, ignore_index=True)

    # The Novel set is assigned to one fold per source ID. Use source_article_id
    # when available; otherwise fall back to article_id, then article_key.
    source_col = next((c for c in ["source_article_id", "article_id", "source_id"] if c in novel_base.columns), None)
    if source_col is None:
        source_col = "article_key"
        print("[WARN] Novel Validation has no source ID column; using article_key for fold assignment.")
    novel_assign = novel_base[["novel_row_id", "article_key", source_col]].copy()
    if novel_assign[source_col].isna().any():
        raise ValueError(f"Novel source key {source_col} contains missing values")
    novel_assign["assigned_fold"] = assign_source_groups_balanced(novel_assign[source_col], args.n_splits)
    

    # novel_validation.csv 是句子级数据，同一 article_key 可以出现多次。
    # 真正要求唯一的是 novel_row_id。
    if novel_assign["novel_row_id"].duplicated().any():
        raise ValueError("Duplicate novel_row_id detected in novel_validation.csv")

    # 检查同一文章是否对应多个 source ID。
    article_source_counts = (
        novel_assign.groupby("article_key")[source_col]
        .nunique(dropna=False)
    )

    if (article_source_counts > 1).any():
        bad_articles = article_source_counts[
            article_source_counts > 1
        ].index.tolist()[:10]

        raise ValueError(
            "同一 article_key 对应多个 source_article_id："
            f"{bad_articles}"
        )

    # 检查同一文章是否被分配到多个 Fold。
    article_fold_counts = (
        novel_assign.groupby("article_key")["assigned_fold"]
        .nunique()
    )

    if (article_fold_counts > 1).any():
        bad_articles = article_fold_counts[
            article_fold_counts > 1
        ].index.tolist()[:10]

        raise RuntimeError(
            "同一 article_key 被分配到多个 Fold："
            f"{bad_articles}"
        )


    source_fold_counts = novel_assign.groupby(source_col)["assigned_fold"].nunique()
    if (source_fold_counts > 1).any():
        raise RuntimeError("A Novel source ID was assigned to multiple folds")

    # Check that Novel source IDs are disjoint from the complete OOF training set.
    train_source = set(oof["source_article_id"].astype(str))
    novel_source = set(novel_assign[source_col].astype(str))
    source_overlap = train_source & novel_source
    if source_overlap:
        raise ValueError(
            f"Novel Validation shares {len(source_overlap)} source IDs with OOF training. "
            "Do not proceed until the original train/validation source split is investigated."
        )

    # Select exactly one matching fold's Novel predictions per row, so the meta
    # feature structure matches the one-fold OOF feature structure during training.
    novel_selected = novel_all_folds.merge(
        novel_assign[["novel_row_id", "assigned_fold"]], on="novel_row_id", validate="many_to_one"
    )
    novel_selected = novel_selected[novel_selected["fold"] == novel_selected["assigned_fold"]].copy()
    if len(novel_selected) != len(novel_base) or novel_selected["novel_row_id"].nunique() != len(novel_base):
        raise RuntimeError("Fold-matched Novel selection does not cover every Novel row exactly once")
    novel_selected = novel_selected.merge(
        novel_base[["novel_row_id", "article_key", "text", "label_id"]],
        on=["novel_row_id", "article_key", "text", "label_id"], validate="one_to_one",
    )
    novel_selected["y_true"] = novel_selected["label_id"].astype(int)
    novel_selected["norm_text"] = novel_selected["text"].map(normalize_sentence)

    y_oof = oof["y_true"].to_numpy(int)

    # Cross-fitted meta-models: for each held-out OOF fold k, train on all other
    # folds, then purge exact normalized text overlap with fold k from meta-train.
    meta_oof_prob = np.full(len(oof), np.nan, dtype=float)
    meta_models: dict[int, LogisticRegression] = {}
    meta_fold_rows: list[dict[str, Any]] = []
    for fold in range(args.n_splits):
        val_mask = oof["fold"].eq(fold).to_numpy()
        meta_val = oof.loc[val_mask]
        meta_train = oof.loc[~val_mask]
        heldout_texts = set(meta_val["norm_text"])
        purge_mask = meta_train["norm_text"].isin(heldout_texts)
        purged_count = int(purge_mask.sum())
        meta_train = meta_train.loc[~purge_mask]
        x_train = meta_train[["svm_z", "roberta_z"]].to_numpy(float)
        y_train = meta_train["y_true"].to_numpy(int)
        x_val = meta_val[["svm_z", "roberta_z"]].to_numpy(float)
        if np.unique(y_train).size != 2:
            raise ValueError(f"Meta fold {fold}: training labels do not contain both classes")
        model = LogisticRegression(C=args.c, max_iter=3000, solver="lbfgs", random_state=args.seed)
        model.fit(x_train, y_train)
        meta_oof_prob[val_mask] = model.predict_proba(x_val)[:, 1]
        meta_models[fold] = model
        meta_fold_rows.append({
            "fold": fold,
            "meta_train_before_purge": int((~val_mask).sum()),
            "meta_train_purged_exact_text": purged_count,
            "meta_train_after_purge": int(len(meta_train)),
            "meta_validation_rows": int(len(meta_val)),
            "meta_train_human": int((y_train == 0).sum()),
            "meta_train_machine": int((y_train == 1).sum()),
        })

    if not np.isfinite(meta_oof_prob).all():
        raise RuntimeError("Meta OOF predictions contain non-finite or missing values")

    # Thresholds selected on OOF training predictions only.
    threshold_candidates = np.linspace(0.01, 0.99, 197)
    svm_t, svm_oof_best, svm_thr = best_threshold(y_oof, oof["svm_z"].to_numpy(float), np.linspace(-4, 4, 401), tie_target=0.0)
    rob_t, rob_oof_best, rob_thr = best_threshold(y_oof, oof["roberta_z"].to_numpy(float), np.linspace(-4, 4, 401), tie_target=0.0)
    stack_t, stack_oof_best, stack_thr = best_threshold(y_oof, meta_oof_prob, threshold_candidates)

    # SVM veto: accept machine only if both the RoBERTa signal and SVM signal
    # exceed thresholds. Tune both thresholds using OOF scores only.
    grid_size = max(11, int(args.veto_grid_size))
    q = np.linspace(0.05, 0.95, grid_size)
    r_candidates = np.unique(np.quantile(oof["roberta_z"].to_numpy(float), q))
    s_candidates = np.unique(np.quantile(oof["svm_z"].to_numpy(float), q))
    best_veto = {"macro_f1": -1.0, "roberta_z_threshold": 0.0, "svm_z_threshold": 0.0}
    veto_rows: list[dict[str, float]] = []
    rz = oof["roberta_z"].to_numpy(float)
    sz = oof["svm_z"].to_numpy(float)
    for rt in r_candidates:
        rpos = rz >= rt
        for st in s_candidates:
            pred = (rpos & (sz >= st)).astype(int)
            mf1 = float(f1_score(y_oof, pred, average="macro", zero_division=0))
            veto_rows.append({"roberta_z_threshold": float(rt), "svm_z_threshold": float(st), "macro_f1": mf1})
            if mf1 > best_veto["macro_f1"] + 1e-12:
                best_veto = {"macro_f1": mf1, "roberta_z_threshold": float(rt), "svm_z_threshold": float(st)}
    veto_table = pd.DataFrame(veto_rows).sort_values("macro_f1", ascending=False)

    # Score Novel: each source article is assigned to one fold, then scored by
    # that fold's base models and the corresponding cross-fitted meta-model.
    
    # 按 novel_row_id 排序，统一 Novel 特征、预测和标签的顺序
    novel_selected = novel_selected.sort_values(
        "novel_row_id"
    ).reset_index(drop=True)

    # 检查每条 Novel 记录仅出现一次
    if novel_selected["novel_row_id"].duplicated().any():
        raise ValueError("Novel predictions contain duplicate novel_row_id")

    if len(novel_selected) != len(novel_base):
        raise ValueError(
            f"Novel row count mismatch: "
            f"{len(novel_selected)} != {len(novel_base)}"
        )

    # 关键修复：排序后重新提取真实标签
    y_novel = novel_selected["y_true"].astype(int).to_numpy()

    # 再次与原始 Novel 数据核对标签
    label_check = novel_selected[
        ["novel_row_id", "y_true"]
    ].merge(
        novel_base[["novel_row_id", "label_id"]],
        on="novel_row_id",
        how="left",
        validate="one_to_one"
    )

    if label_check["label_id"].isna().any():
        raise ValueError("Missing canonical Novel labels")

    if not np.array_equal(
        label_check["y_true"].to_numpy(int),
        label_check["label_id"].to_numpy(int)
    ):
        raise ValueError(
            "Novel prediction labels do not match canonical labels"
        )

    # 所有后续预测数组均基于当前排序
    novel_prob = np.full(
        len(novel_selected), np.nan, dtype=float
    )

    for fold, model in meta_models.items():
        mask = novel_selected["fold"].astype(int).eq(fold).to_numpy()
        if not mask.any():
            continue
        x = novel_selected.loc[mask, ["svm_z", "roberta_z"]].to_numpy(float)
        novel_prob[mask] = model.predict_proba(x)[:, 1]
    if not np.isfinite(novel_prob).all():
        raise RuntimeError("Fold-matched Novel stacker predictions are incomplete")

    novel_selected["stacker_p_machine"] = novel_prob
    novel_selected["stacker_pred"] = (novel_prob >= stack_t).astype(int)
    novel_selected["svm_pred_oof_threshold"] = (novel_selected["svm_z"].to_numpy(float) >= svm_t).astype(int)
    novel_selected["roberta_pred_oof_threshold"] = (novel_selected["roberta_z"].to_numpy(float) >= rob_t).astype(int)
    novel_selected["veto_pred"] = (
        (novel_selected["roberta_z"].to_numpy(float) >= best_veto["roberta_z_threshold"])
        & (novel_selected["svm_z"].to_numpy(float) >= best_veto["svm_z_threshold"])
    ).astype(int)

    # Cross-fitted OOF predictions and selected thresholds.
    oof["stacker_p_machine_oof"] = meta_oof_prob
    oof["stacker_pred_oof"] = (meta_oof_prob >= stack_t).astype(int)
    oof["svm_pred_oof_threshold"] = (oof["svm_z"].to_numpy(float) >= svm_t).astype(int)
    oof["roberta_pred_oof_threshold"] = (oof["roberta_z"].to_numpy(float) >= rob_t).astype(int)
    oof["veto_pred_oof"] = ((oof["roberta_z"].to_numpy(float) >= best_veto["roberta_z_threshold"]) & (oof["svm_z"].to_numpy(float) >= best_veto["svm_z_threshold"])).astype(int)

    # Metrics. OOF metrics are model-selection diagnostics; Novel is the external
    # evaluation for this stacking stage, but it is still the project's dev split.
    result_rows = []
    for name, pred, score in [
        ("svm_oof_threshold", oof["svm_pred_oof_threshold"].to_numpy(int), oof["svm_z"].to_numpy(float)),
        ("roberta_oof_threshold", oof["roberta_pred_oof_threshold"].to_numpy(int), oof["roberta_z"].to_numpy(float)),
        ("stacker_oof_crossfit", oof["stacker_pred_oof"].to_numpy(int), meta_oof_prob),
        ("veto_oof", oof["veto_pred_oof"].to_numpy(int), oof[["svm_z", "roberta_z"]].min(axis=1).to_numpy(float)),
    ]:
        row = {"split": "oof_crossfit_diagnostic", "model": name}
        row.update(metrics_dict(y_oof, pred, score))
        result_rows.append(row)
    for name, pred, score in [
        ("svm_fold_matched", novel_selected["svm_pred_oof_threshold"].to_numpy(int), novel_selected["svm_z"].to_numpy(float)),
        ("roberta_fold_matched", novel_selected["roberta_pred_oof_threshold"].to_numpy(int), novel_selected["roberta_z"].to_numpy(float)),
        ("stacker_fold_matched", novel_selected["stacker_pred"].to_numpy(int), novel_prob),
        ("svm_veto_fold_matched", novel_selected["veto_pred"].to_numpy(int), novel_selected[["svm_z", "roberta_z"]].min(axis=1).to_numpy(float)),
    ]:
        row = {"split": "novel_validation_fold_matched", "model": name}
        row.update(metrics_dict(y_novel, pred, score))
        result_rows.append(row)
    # Five-fold mean-score ranking diagnostics on Novel. These are deliberately
    # ranking-only: the stacker is trained on one-fold OOF features, so we do not
    # apply its threshold to a five-fold averaged feature representation.
    novel_ensemble = novel_all_folds.groupby("novel_row_id", as_index=False).agg(
        y_true=("y_true", "first"),
        svm_z_mean=("svm_z", "mean"),
        roberta_z_mean=("roberta_z", "mean"),
        svm_z_std=("svm_z", "std"),
        roberta_z_std=("roberta_z", "std"),
        fold_count=("fold", "nunique"),
    )
    for name, score_col in [("svm_5fold_mean_z_ranking_only", "svm_z_mean"), ("roberta_5fold_mean_z_ranking_only", "roberta_z_mean")]:
        score = novel_ensemble[score_col].to_numpy(float)
        result_rows.append({
            "split": "novel_5fold_ensemble_ranking_only",
            "model": name,
            "n": int(len(novel_ensemble)),
            "roc_auc": safe_auc(novel_ensemble["y_true"].to_numpy(int), score),
            "average_precision": safe_ap(novel_ensemble["y_true"].to_numpy(int), score),
        })
    metrics_df = pd.DataFrame(result_rows)

    # Save outputs.
    oof.to_csv(out_dir / "oof_meta_predictions.csv", index=False, encoding="utf-8-sig")
    novel_selected.to_csv(out_dir / "novel_fusion_predictions.csv", index=False, encoding="utf-8-sig")
    novel_assign.to_csv(out_dir / "novel_fold_assignments.csv", index=False, encoding="utf-8-sig")
    novel_assign.groupby("assigned_fold").agg(
        article_count=("article_key", "nunique"),
        source_group_count=(source_col, "nunique"),
        sentence_count=("novel_row_id", "size"),
    ).reset_index().to_csv(out_dir / "novel_fold_assignment_summary.csv", index=False, encoding="utf-8-sig")
    novel_ensemble.to_csv(out_dir / "novel_five_fold_score_diagnostics.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(out_dir / "metrics_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(meta_fold_rows).to_csv(out_dir / "meta_fold_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat([
        svm_thr.assign(model="svm"), rob_thr.assign(model="roberta"), stack_thr.assign(model="stacker")
    ], ignore_index=True).to_csv(out_dir / "threshold_search.csv", index=False, encoding="utf-8-sig")
    veto_table.to_csv(out_dir / "veto_grid_search.csv", index=False, encoding="utf-8-sig")
    (out_dir / "score_normalization.json").write_text(json.dumps(score_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {
        "n_splits": args.n_splits,
        "seed": args.seed,
        "logistic_C": args.c,
        "normalization": "per-fold z-score fitted on that fold's OOF validation score distribution; reused for matching Novel fold",
        "novel_assignment": f"deterministic largest-first source-group balancing; key column={source_col}; no labels used",
        "meta_cv": "hold out original OOF fold; train on other folds; purge exact normalized text overlap with held-out meta fold",
        "svm_threshold_z": svm_t,
        "roberta_threshold_z": rob_t,
        "stacker_threshold": stack_t,
        "veto_roberta_z_threshold": best_veto["roberta_z_threshold"],
        "veto_svm_z_threshold": best_veto["svm_z_threshold"],
        "veto_oof_macro_f1": best_veto["macro_f1"],
        "novel_source_id_overlap_with_oof_train": len(source_overlap),
        "warning": "Thresholds and rules are selected on OOF training labels only. Novel metrics are evaluation results for development validation, not final blind-test guarantees.",
    }
    (out_dir / "fusion_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"meta_models_by_heldout_fold": meta_models, "config": config, "score_stats": score_stats}, out_dir / "fusion_models.joblib", compress=3)

    summary = [
        "AIGTD OOF fusion training/evaluation",
        "=" * 72,
        f"OOF rows: {len(oof):,}; Novel rows: {len(novel_selected):,}",
        f"Novel source key: {source_col}; source IDs shared with OOF train: {len(source_overlap)}",
        "Novel fold assignment (source groups kept intact):",
        novel_assign.groupby("assigned_fold").agg(article_count=("article_key", "nunique"), source_group_count=(source_col, "nunique"), sentence_count=("novel_row_id", "size")).to_string(),
        f"Per-fold score normalization: {json.dumps(score_stats, ensure_ascii=False)}",
        "",
        f"OOF-selected SVM z threshold: {svm_t:.6f}",
        f"OOF-selected RoBERTa z threshold: {rob_t:.6f}",
        f"OOF-selected Stacker probability threshold: {stack_t:.6f}",
        f"OOF-selected veto: roberta_z>={best_veto['roberta_z_threshold']:.6f} AND svm_z>={best_veto['svm_z_threshold']:.6f}",
        "",
        "Metrics:",
        metrics_df.to_string(index=False),
        "",
        "Important interpretation:",
        "- Meta-models are cross-fitted by the original OOF fold; exact normalized text overlap is purged from meta-training.",
        "- Novel rows are assigned by source ID to one fold and scored by its corresponding base fold models and meta-model.",
        "- This fold-matched evaluation avoids feeding five-fold averaged features to a stacker trained on single-fold OOF features.",
        "- OOF metrics are training/model-selection diagnostics; Novel metrics are the held-out evaluation for this stacking stage.",
        "- Five-fold mean z-score ranking metrics are reported as ranking-only diagnostics; no classification threshold is applied to them.",
        "- Fold-matched Novel metrics use one deterministically assigned base fold per source ID, matching the one-fold OOF feature structure.",
        "- Duplicate normalized texts across different OOF validation folds remain in sentence-level evaluation; meta-training purge prevents exact-text overlap with each meta holdout fold.",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("\n[OOF Fusion] Completed.")
    print(f"OOF-selected thresholds: SVM z={svm_t:.4f}; RoBERTa z={rob_t:.4f}; stacker={stack_t:.4f}")
    print(f"SVM veto: roberta_z>={best_veto['roberta_z_threshold']:.4f} AND svm_z>={best_veto['svm_z_threshold']:.4f}")
    print("\nNovel Validation metrics:")
    print(metrics_df[metrics_df["split"] == "novel_validation_fold_matched"].to_string(index=False))
    print(f"\nOutputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
