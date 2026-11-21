#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze which Fold-4 rows are harmed by context and whether risk is predictable.

The analysis is paired at oof_row_id, uses only out-of-fold predictions, and
keeps article IDs out of aggregate reports. It does not retrain either base
model or claim that associations are causal.

Primary target for learnability:
  among rows classified correctly by Sentence-only, predict whether Context
  changes that correct decision into an error. Features are available at
  inference time: the two model scores and text/context length statistics.
Article-grouped cross-validation prevents sentences from one article appearing
in both train and validation partitions of the diagnostic risk model.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def load_pair(root: Path) -> pd.DataFrame:
    a = pd.read_csv(root / "sentence_matched_oof_predictions.csv", low_memory=False)
    b = pd.read_csv(root / "context_oof_predictions.csv", low_memory=False)
    required = {"oof_row_id", "fold", "y_true", "context_p_machine"}
    for name, df in (("sentence", a), ("context", b)):
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{name} prediction file missing columns: {missing}")
        if df.oof_row_id.isna().any() or df.oof_row_id.duplicated().any():
            raise ValueError(f"{name} oof_row_id must be non-null and unique")
        df["context_p_machine"] = pd.to_numeric(df.context_p_machine, errors="coerce")
        if not np.isfinite(df.context_p_machine.to_numpy(float)).all():
            raise ValueError(f"{name} contains non-finite probabilities")
        if not df.context_p_machine.between(0, 1).all():
            raise ValueError(f"{name} probabilities outside [0,1]")
    a = a.sort_values("oof_row_id").reset_index(drop=True)
    b = b.sort_values("oof_row_id").reset_index(drop=True)
    if not np.array_equal(a.oof_row_id, b.oof_row_id):
        raise ValueError("OOF row IDs do not match")
    for col in ("fold", "y_true"):
        if not np.array_equal(a[col], b[col]):
            raise ValueError(f"Paired files disagree on {col}")
    for col in ("article_key", "article_id", "source_article_id",
                "sentence_index", "text", "context_text", "label_id"):
        if col in a and col in b:
            if not a[col].fillna("<NA>").astype(str).equals(
                b[col].fillna("<NA>").astype(str)
            ):
                raise ValueError(f"Paired files disagree on metadata: {col}")
    out = a.copy().rename(columns={"context_p_machine": "sentence_p_machine"})
    out["context_p_machine"] = b.context_p_machine.to_numpy(float)
    out["y_true"] = out.y_true.astype(int)
    out["sentence_pred"] = (out.sentence_p_machine >= 0.5).astype(int)
    out["context_pred"] = (out.context_p_machine >= 0.5).astype(int)
    out["sentence_correct"] = out.sentence_pred.eq(out.y_true)
    out["context_correct"] = out.context_pred.eq(out.y_true)
    out["harmed"] = out.sentence_correct & ~out.context_correct
    out["helped"] = ~out.sentence_correct & out.context_correct
    out["both_correct"] = out.sentence_correct & out.context_correct
    out["both_wrong"] = ~out.sentence_correct & ~out.context_correct
    out["probability_delta"] = out.context_p_machine - out.sentence_p_machine
    out["sentence_confidence"] = np.maximum(
        out.sentence_p_machine, 1 - out.sentence_p_machine
    )
    out["context_confidence"] = np.maximum(
        out.context_p_machine, 1 - out.context_p_machine
    )
    out["confidence_delta"] = out.context_confidence - out.sentence_confidence
    # Paired proper-scoring-rule deltas: positive means Context is worse.
    # Clip only for log-loss, so probabilities at exactly 0/1 stay finite.
    y = out.y_true.to_numpy(dtype=float)
    ps = out.sentence_p_machine.to_numpy(dtype=float)
    pc = out.context_p_machine.to_numpy(dtype=float)
    out["brier_delta"] = (pc - y) ** 2 - (ps - y) ** 2
    eps = 1e-15
    ps_clip = np.clip(ps, eps, 1 - eps)
    pc_clip = np.clip(pc, eps, 1 - eps)
    sentence_loss = -(y * np.log(ps_clip) + (1 - y) * np.log1p(-ps_clip))
    context_loss = -(y * np.log(pc_clip) + (1 - y) * np.log1p(-pc_clip))
    out["logloss_delta"] = context_loss - sentence_loss
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    text = x["text"].fillna("").astype(str) if "text" in x else pd.Series("", index=x.index)
    context = x["context_text"].fillna("").astype(str) if "context_text" in x else pd.Series("", index=x.index)
    x["target_chars"] = text.str.len()
    x["target_words"] = text.str.split().str.len()
    prev, nxt = [], []
    for value in context:
        m = re.search(r"\[PREVIOUS\]\s*(.*?)\s*\[TARGET\]", value, flags=re.S)
        n = re.search(r"\[NEXT\]\s*(.*)$", value, flags=re.S)
        pv = m.group(1).strip() if m else ""
        nv = n.group(1).strip() if n else ""
        prev.append("" if pv == "[START]" else pv)
        nxt.append("" if nv == "[END]" else nv)
    x["previous_chars"] = pd.Series(prev, index=x.index).str.len()
    x["next_chars"] = pd.Series(nxt, index=x.index).str.len()
    x["context_chars"] = context.str.len()
    x["has_previous"] = (x.previous_chars > 0).astype(int)
    x["has_next"] = (x.next_chars > 0).astype(int)
    x["target_punctuation"] = text.str.count(r"[.,;:!?]")
    x["target_uppercase_ratio"] = [
        sum(ch.isupper() for ch in s) / max(sum(ch.isalpha() for ch in s), 1)
        for s in text
    ]
    x["target_digit_ratio"] = [
        sum(ch.isdigit() for ch in s) / max(len(s), 1) for s in text
    ]
    return x


def outcome_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby("outcome", sort=True):
        rows.append({
            "outcome": name, "n": len(g), "share": len(g) / len(df),
            "human_n": int((g.y_true == 0).sum()),
            "machine_n": int((g.y_true == 1).sum()),
            "mean_target_chars": float(g.target_chars.mean()),
            "median_target_chars": float(g.target_chars.median()),
            "mean_previous_chars": float(g.previous_chars.mean()),
            "mean_next_chars": float(g.next_chars.mean()),
            "mean_sentence_confidence": float(g.sentence_confidence.mean()),
            "mean_context_confidence": float(g.context_confidence.mean()),
            "mean_confidence_delta": float(g.confidence_delta.mean()),
            "mean_probability_delta": float(g.probability_delta.mean()),
            "mean_brier_delta": float(g.brier_delta.mean()),
            "mean_logloss_delta": float(g.logloss_delta.mean()),
            "fraction_brier_worsened": float((g.brier_delta > 0).mean()),
            "fraction_logloss_worsened": float((g.logloss_delta > 0).mean()),
        })
    return pd.DataFrame(rows)


def stratified_report(df: pd.DataFrame, column: str, min_n: int = 30) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(column, dropna=False):
        if len(g) < min_n:
            continue
        rows.append({
            "feature": column, "stratum": str(key), "n": len(g),
            "harm_n": int(g.harmed.sum()), "help_n": int(g.helped.sum()),
            "harm_rate": float(g.harmed.mean()),
            "help_rate": float(g.helped.mean()),
            "net_harm_rate": float((g.harmed.sum() - g.helped.sum()) / len(g)),
            "human_share": float((g.y_true == 0).mean()),
            "mean_target_chars": float(g.target_chars.mean()),
            "mean_previous_chars": float(g.previous_chars.mean()),
            "mean_next_chars": float(g.next_chars.mean()),
        })
    return pd.DataFrame(rows)


def article_concentration(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df:
        return pd.DataFrame()
    g = df.groupby(group_col, dropna=False).agg(
        n=("harmed", "size"), harmed=("harmed", "sum"),
        helped=("helped", "sum"), human=("y_true", lambda s: int((s == 0).sum()))
    ).reset_index(drop=True)
    g = g[g.n >= 3].copy()
    if g.empty:
        return pd.DataFrame()
    g["harm_rate"] = g.harmed / g.n
    g["net_harm"] = g.harmed - g.helped
    g = g.sort_values(["harmed", "net_harm", "n"], ascending=False)
    total_harm = max(int(df.harmed.sum()), 1)
    top = g.head(max(1, int(np.ceil(len(g) * 0.10))))
    return pd.DataFrame([{
        "grouping": group_col,
        "groups_with_at_least_3_rows": len(g),
        "groups_with_any_harm": int((g.harmed > 0).sum()),
        "groups_with_harm_exceeding_help": int((g.net_harm > 0).sum()),
        "top_10pct_groups": len(top),
        "harm_share_in_top_10pct_groups": float(top.harmed.sum() / total_harm),
        "max_harm_in_one_group": int(g.harmed.max()),
        "median_group_harm_rate": float(g.harm_rate.median()),
        "p90_group_harm_rate": float(g.harm_rate.quantile(0.90)),
    }])


def learnability_report(df: pd.DataFrame, out: Path, seed: int) -> dict:
    # A gate is only relevant where Sentence-only is correct: should it trust
    # Context, or retain the Sentence prediction? Predict the harmful flip.
    eligible = df.loc[df.sentence_correct].copy()
    if eligible.harmed.nunique() < 2:
        return {"status": "insufficient_target_variation", "n": len(eligible)}
    feature_cols = [
        "sentence_p_machine", "context_p_machine", "probability_delta",
        "sentence_confidence", "context_confidence", "confidence_delta",
        "target_chars", "target_words", "previous_chars", "next_chars",
        "context_chars", "has_previous", "has_next", "target_punctuation",
        "target_uppercase_ratio", "target_digit_ratio",
    ]
    if "domain" in eligible.columns:
        eligible["domain_code"] = eligible.domain.astype("category").cat.codes
        feature_cols.append("domain_code")
    X = eligible[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = eligible.harmed.astype(int).to_numpy()
    group_col = next((c for c in ("article_key", "source_article_id", "article_id")
                      if c in eligible.columns), None)
    if group_col is None:
        return {"status": "missing_article_group", "n": len(eligible)}
    groups = eligible[group_col].fillna("__MISSING__").astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    if n_splits < 2:
        return {"status": "too_few_article_groups", "n": len(eligible)}
    cv = GroupKFold(n_splits=n_splits)
    predictions = {
        "logistic": np.full(len(eligible), np.nan),
        "extra_trees": np.full(len(eligible), np.nan),
    }
    models = {
        "logistic": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5)
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=400, min_samples_leaf=8, max_features=0.8,
                class_weight="balanced", random_state=seed, n_jobs=-1
            )
        ),
    }
    for train_idx, test_idx in cv.split(X, y, groups):
        for name, model in models.items():
            model.fit(X.iloc[train_idx], y[train_idx])
            predictions[name][test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    prevalence = float(y.mean())
    rows = []
    for name, score in predictions.items():
        valid = np.isfinite(score)
        s, yy = score[valid], y[valid]
        rows.append({
            "model": name, "n": int(valid.sum()), "article_groups": n_groups,
            "harm_prevalence": prevalence,
            "average_precision": float(average_precision_score(yy, s)),
            "roc_auc": float(roc_auc_score(yy, s)) if len(np.unique(yy)) == 2 else float("nan"),
            "precision_top_10pct": float(yy[np.argsort(s)[-max(1, int(np.ceil(.1*len(yy)))):]].mean()),
            "precision_top_20pct": float(yy[np.argsort(s)[-max(1, int(np.ceil(.2*len(yy)))):]].mean()),
        })
    pd.DataFrame(rows).to_csv(out / "fold4_harm_risk_groupcv.csv", index=False)
    # Fit a descriptive final model only to summarize feature importance. Its
    # in-sample importance is not used as evidence of generalization.
    final = models["extra_trees"]
    final.fit(X, y)
    forest = final.named_steps["extratreesclassifier"]
    importance = pd.DataFrame({
        "feature": feature_cols, "importance": forest.feature_importances_
    }).sort_values("importance", ascending=False)
    importance.to_csv(out / "fold4_harm_feature_importance_descriptive.csv", index=False)
    return {
        "status": "ok", "eligible_sentence_correct_rows": len(eligible),
        "harm_count": int(y.sum()), "harm_prevalence": prevalence,
        "article_group_column": group_col, "article_groups": n_groups,
        "cross_validation": "GroupKFold by article; OOF risk scores",
        "warning": "Fold-4-only exploratory predictability; not an independent test or evidence of causal mechanism.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="results/context_oof")
    p.add_argument("--output-dir", default="results/context_harm_patterns")
    p.add_argument("--fold", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = add_features(load_pair(Path(args.input_dir)))
    df = df.loc[df.fold.astype(int).eq(args.fold)].copy()
    if df.empty:
        raise ValueError(f"No rows for fold {args.fold}")
    df["outcome"] = np.select(
        [df.both_correct, df.both_wrong, df.helped, df.harmed],
        ["both_correct", "both_wrong", "context_only_correct", "sentence_only_correct"],
        default="unknown",
    )
    df["true_class"] = df.y_true.map({0: "Human", 1: "Machine"})
    df["target_length_bucket"] = pd.cut(
        df.target_chars, bins=[-1, 70, 98, 130, np.inf],
        labels=["<=70", "71-98", "99-130", ">130"]
    ).astype(str)
    df["previous_context_bucket"] = pd.cut(
        df.previous_chars, bins=[-1, 0, 64, 128, 256, np.inf],
        labels=["0", "1-64", "65-128", "129-256", ">256"]
    ).astype(str)
    df["next_context_bucket"] = pd.cut(
        df.next_chars, bins=[-1, 0, 64, 128, 256, np.inf],
        labels=["0", "1-64", "65-128", "129-256", ">256"]
    ).astype(str)

    outcome_summary(df).to_csv(out / "fold4_outcome_feature_summary.csv", index=False)
    by_class = df.groupby("true_class").agg(
        n=("harmed", "size"), harmed=("harmed", "sum"), helped=("helped", "sum"),
        harm_rate=("harmed", "mean"), help_rate=("helped", "mean")
    ).reset_index()
    by_class.to_csv(out / "fold4_outcome_by_true_class.csv", index=False)

    reports = []
    for col in ("target_length_bucket", "previous_context_bucket",
                "next_context_bucket", "true_class"):
        report = stratified_report(df, col)
        if not report.empty:
            reports.append(report)
    if reports:
        pd.concat(reports, ignore_index=True).to_csv(
            out / "fold4_harm_stratified_patterns.csv", index=False
        )

    concentration = []
    for col in ("article_key", "source_article_id", "article_id"):
        report = article_concentration(df, col)
        if not report.empty:
            concentration.append(report)
    if concentration:
        pd.concat(concentration, ignore_index=True).to_csv(
            out / "fold4_article_harm_concentration.csv", index=False
        )

    learnability = learnability_report(df, out, args.seed)
    summary = {
        "fold": args.fold, "rows": len(df),
        "outcome_counts": {str(k): int(v) for k, v in df.outcome.value_counts().items()},
        "human_harmed": int(((df.y_true == 0) & df.harmed).sum()),
        "machine_harmed": int(((df.y_true == 1) & df.harmed).sum()),
        "human_helped": int(((df.y_true == 0) & df.helped).sum()),
        "machine_helped": int(((df.y_true == 1) & df.helped).sum()),
        "learnability": learnability,
        "privacy": {
            "row_level_predictions_exported": False,
            "sentence_text_exported": False,
            "article_identifiers_exported": False,
        },
        "interpretation_limits": [
            "Fold-4 subgroup associations are descriptive, not causal.",
            "Risk-model cross-validation groups by article, but uses only Fold 4.",
            "A successful risk model would motivate, not validate, a learned gate.",
        ],
    }
    (out / "fold4_harm_pattern_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Aggregate outputs: {out.resolve()}")


if __name__ == "__main__":
    main()
