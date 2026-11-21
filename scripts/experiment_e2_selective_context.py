#!/usr/bin/env python3
"""E2: no-retraining selective-context experiment on strictly paired OOF rows.

Only the Sentence-matched / Context models trained on the same strict
context-window purge are compared. Never mix these results with the distinct
OOF Stacking model's training protocol or its Novel Validation scores.

All rule parameters and the classification threshold are fixed in source.
For the diagnostic fold-selected rule, choose on four OOF folds, evaluate on
the fifth. This is *development* cross-fitting, not blind-test evaluation.
Outputs are aggregate-only; no raw sentences or article/source identifiers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)

try:
    from scripts.analyze_context_oof import load_pair
except ImportError:
    from analyze_context_oof import load_pair


THRESHOLD = 0.5
RULES = (
    "sentence_matched",
    "context",
    "mean_50",
    "context_75",
    "short_context_12",
    "short_context_24",
    "long_human_veto_20",
    "long_human_veto_32",
)
SELECTION_RULES = tuple(r for r in RULES if r != "sentence_matched")
# An exploratory Human-FPR ceiling relative to context on *training* folds.
# Not a claimed statistical error guarantee.
MAX_TRAIN_FPR_INCREASE = 0.002


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def load_strict_pair(folder: Path) -> pd.DataFrame:
    a, b = load_pair(
        folder / "sentence_matched_oof_predictions.csv",
        folder / "context_oof_predictions.csv", "oof_row_id")
    required = {"source_article_id", "article_key", "sentence_index", "text"}
    for name, frame in (("sentence", a), ("context", b)):
        missing = required - set(frame.columns)
        require(not missing, f"{name} missing paired metadata: {sorted(missing)}")
        require(not frame[list(required)].isna().any().any(),
                f"{name} has missing article/source/text/position metadata")
    for col in (*sorted(required), "article_id", "label_id", "context_text"):
        if col in a.columns and col in b.columns:
            require(a[col].astype(str).equals(b[col].astype(str)),
                    f"Paired OOF files disagree on metadata: {col}")
    require(set(a["fold"].astype(int)) == set(range(5)),
            "Expected exactly 5 existing OOF folds")
    group_fold_count = a.groupby(a["source_article_id"].astype(str))["fold"].nunique()
    require((group_fold_count == 1).all(),
            "Source article ID crosses original OOF evaluation folds")
    require(a["article_key"].nunique() <= a["source_article_id"].nunique() * 100,
            "Unexpected article-to-source cardinality")
    words = a["text"].astype(str).str.split().str.len().to_numpy(int)
    require((words > 0).all(), "Empty sentence found in OOF inputs")
    out = pd.DataFrame({
        "fold": a["fold"].astype(int),
        "y_true": a["y_true"].astype(int),
        "source_article_id": a["source_article_id"].astype(str),
        "article_key": a["article_key"].astype(str),
        "target_words": words,
        "sentence_p_machine": a["context_p_machine"].to_numpy(float),
        "context_p_machine": b["context_p_machine"].to_numpy(float),
    })
    require(out["y_true"].isin([0, 1]).all(), "Invalid class labels")
    return out


def predict_rule(name: str, s: np.ndarray, c: np.ndarray,
                 target_words: np.ndarray) -> np.ndarray:
    """Pure inference-time function: no labels, correctness or source IDs."""
    s, c, target_words = np.asarray(s, dtype=float), np.asarray(c, dtype=float), np.asarray(target_words)
    require(s.ndim == c.ndim == target_words.ndim == 1 and s.shape == c.shape == target_words.shape,
            "Rule features must be equal-length 1-D vectors")
    require(np.isfinite(s).all() and np.isfinite(c).all(), "Nonfinite prediction scores")
    require(((s >= 0) & (s <= 1) & (c >= 0) & (c <= 1)).all(), "Invalid probability score")
    if name == "sentence_matched":
        p = s
    elif name == "context":
        p = c
    elif name == "mean_50":
        p = 0.5 * s + 0.5 * c
    elif name == "context_75":
        p = 0.25 * s + 0.75 * c
    elif name in ("short_context_12", "short_context_24"):
        limit = int(name.rsplit("_", 1)[1])
        p = np.where(target_words <= limit, c, s)
    elif name in ("long_human_veto_20", "long_human_veto_32"):
        limit = int(name.rsplit("_", 1)[1])
        # Only override Context=>Machine when Sentence=>Human on long targets.
        override = (target_words >= limit) & (c >= THRESHOLD) & (s < THRESHOLD)
        p = np.where(override, s, c)
    else:
        raise ValueError(f"Unknown fixed rule: {name}")
    return np.asarray(p, dtype=float)


def metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    require(y.size == probability.size and y.size > 0, "Metrics require nonempty aligned arrays")
    require(set(np.unique(y)).issubset({0, 1}), "Class labels must be 0/1")
    require(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(),
            "Invalid probability vector")
    p = (probability >= THRESHOLD).astype(int)
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tp = int(((y == 1) & (p == 1)).sum())
    return {
        "n": int(y.size), "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, p, average="micro", zero_division=0)),
        "f1_human": float(f1_score(y, p, pos_label=0, zero_division=0)),
        "f1_machine": float(f1_score(y, p, pos_label=1, zero_division=0)),
        "precision_machine": float(precision_score(y, p, pos_label=1, zero_division=0)),
        "recall_machine": float(recall_score(y, p, pos_label=1, zero_division=0)),
        "precision_human": float(precision_score(y, p, pos_label=0, zero_division=0)),
        "recall_human": float(recall_score(y, p, pos_label=0, zero_division=0)),
        "tn": tn, "fp_human_to_machine": fp, "fn_machine_to_human": fn, "tp": tp,
        "human_fpr": float(fp / (tn + fp)) if tn + fp else float("nan"),
        "machine_recall": float(tp / (fn + tp)) if fn + tp else float("nan"),
        "human_support": tn + fp, "machine_support": fn + tp,
    }


def choose_rule_on_other_folds(
    y_train: np.ndarray, predicted_scores: dict[str, np.ndarray],
    train_mask: np.ndarray, allowed: Sequence[str] = SELECTION_RULES,
) -> tuple[str, list[dict]]:
    """Select using training-fold labels only; prefer Context for exact ties."""
    require(train_mask.dtype == bool and train_mask.ndim == 1, "Invalid train mask")
    baseline = metrics(y_train[train_mask], predicted_scores["context"][train_mask])
    max_fpr = baseline["human_fpr"] + MAX_TRAIN_FPR_INCREASE
    trials = []
    for rule in allowed:
        m = metrics(y_train[train_mask], predicted_scores[rule][train_mask])
        eligible = bool(m["human_fpr"] <= max_fpr + 1e-12)
        trials.append({
            "rule": rule, "macro_f1": m["macro_f1"],
            "human_fpr": m["human_fpr"], "eligible": eligible,
        })
    good = [x for x in trials if x["eligible"]]
    require(bool(good), "No eligible rules, including the Context default")
    # Ties go to Context and then to the earlier pre-specified simpler rule.
    winner = sorted(good, key=lambda x: (-x["macro_f1"], list(allowed).index(x["rule"])))[0]
    return str(winner["rule"]), trials


def crossfit_selector(frame: pd.DataFrame,
                      scores: dict[str, np.ndarray]) -> tuple[np.ndarray, list[dict]]:
    """Five-fold rule selection; source groups do not cross held-out folds."""
    fold_values = frame["fold"].to_numpy(int)
    y = frame["y_true"].to_numpy(int)
    require(set(fold_values) == set(range(5)), "Must have all five folds")
    predictions = np.full(len(frame), np.nan, dtype=float)
    selected = []
    for fold in range(5):
        test = fold_values == fold
        train = ~test
        require(train.any() and test.any(), f"Empty split on fold {fold}")
        choice, trials = choose_rule_on_other_folds(y, scores, train)
        predictions[test] = scores[choice][test]
        context_train = metrics(y[train], scores["context"][train])
        chosen_train = metrics(y[train], scores[choice][train])
        selected.append({
            "heldout_fold": fold, "rule_selected_from_other_four_folds": choice,
            "train_rows": int(train.sum()), "heldout_rows": int(test.sum()),
            "train_macro_f1_context": context_train["macro_f1"],
            "train_macro_f1_selected": chosen_train["macro_f1"],
            "train_human_fpr_context": context_train["human_fpr"],
            "train_human_fpr_selected": chosen_train["human_fpr"],
            "eligible_rule_count": sum(row["eligible"] for row in trials),
        })
    require(np.isfinite(predictions).all(), "Cross-fit selector left unscored predictions")
    return predictions, selected


def transition_rows(frame: pd.DataFrame,
                    s: np.ndarray, c: np.ndarray) -> list[dict]:
    y = frame["y_true"].to_numpy(int)
    s_good = (s >= THRESHOLD).astype(int) == y
    c_good = (c >= THRESHOLD).astype(int) == y
    outcomes = {
        "both_correct": s_good & c_good,
        "both_wrong": ~s_good & ~c_good,
        "context_only_correct": ~s_good & c_good,
        "sentence_only_correct": s_good & ~c_good,
    }
    rows = []
    for label, scope in (("all", np.ones(len(y), dtype=bool)),
                         ("human", y == 0), ("machine", y == 1)):
        for outcome, mask in outcomes.items():
            rows.append({
                "true_class": label, "outcome": outcome,
                "rows": int((scope & mask).sum()),
                "scope_rows": int(scope.sum()),
                "share": float((scope & mask).sum() / max(1, scope.sum())),
            })
    return rows


def length_analysis(frame: pd.DataFrame,
                    scores: dict[str, np.ndarray]) -> list[dict]:
    bins = ((1, 8), (9, 16), (17, 32), (33, 64), (65, None))
    y = frame["y_true"].to_numpy(int)
    words = frame["target_words"].to_numpy(int)
    s = (scores["sentence_matched"] >= THRESHOLD).astype(int)
    c = (scores["context"] >= THRESHOLD).astype(int)
    rows = []
    for low, high in bins:
        length_mask = (words >= low) & ((words <= high) if high is not None else True)
        for cls_name, cls_mask in (
            ("all", np.ones(len(y), dtype=bool)),
            ("human", y == 0), ("machine", y == 1),
        ):
            mask = length_mask & cls_mask
            if not mask.any():
                continue
            sentence_correct = int((s[mask] == y[mask]).sum())
            context_correct = int((c[mask] == y[mask]).sum())
            row = {
                "words_bin": f"{low}-{high if high is not None else 'plus'}",
                "true_class": cls_name,
                "rows": int(mask.sum()),
                "sentence_correct": sentence_correct,
                "context_correct": context_correct,
                "context_net_correct_minus_sentence": context_correct - sentence_correct,
                "context_net_correct_share": (context_correct - sentence_correct) / int(mask.sum()),
                "context_helped": int(((s[mask] != y[mask]) & (c[mask] == y[mask])).sum()),
                "context_harmed": int(((s[mask] == y[mask]) & (c[mask] != y[mask])).sum()),
            }
            rows.append(row)
    return rows


def _by_source_confusions(y: np.ndarray, predictions: np.ndarray,
                          group_inverse: np.ndarray, n_groups: int) -> np.ndarray:
    p = (predictions >= THRESHOLD).astype(int)
    outcome = (2 * y + p).astype(int)  # [TN, FP, FN, TP]
    codes = 4 * group_inverse + outcome
    return np.bincount(codes, minlength=4*n_groups).reshape(n_groups, 4)


def _aggregates(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tn, fp, fn, tp = [counts[:, i].astype(float) for i in range(4)]
    f1_h = np.divide(2 * tn, 2*tn + fp + fn,
                     out=np.zeros_like(tn), where=(2*tn + fp + fn) > 0)
    f1_m = np.divide(2 * tp, 2*tp + fp + fn,
                     out=np.zeros_like(tp), where=(2*tp + fp + fn) > 0)
    fpr = np.divide(fp, tn + fp,
                    out=np.zeros_like(fp), where=(tn + fp) > 0)
    accuracy = np.divide(tn + tp, tn + fp + fn + tp,
                         out=np.zeros_like(tp), where=(tn + fp + fn + tp) > 0)
    return 0.5 * (f1_h + f1_m), fpr, accuracy


def clustered_bootstrap(frame: pd.DataFrame, target: np.ndarray,
                        baseline: np.ndarray, reps: int, seed: int) -> list[dict]:
    require(reps >= 10, "Bootstrap repetitions must be at least 10")
    y = frame["y_true"].to_numpy(int)
    source = frame["source_article_id"].astype(str).to_numpy()
    unique, inverse = np.unique(source, return_inverse=True)
    n_groups = len(unique)
    require(n_groups >= 2, "Need at least two distinct source groups")
    target_count = _by_source_confusions(y, target, inverse, n_groups)
    base_count = _by_source_confusions(y, baseline, inverse, n_groups)
    observed = np.array(_aggregates(target_count.sum(axis=0, keepdims=True)))[:, 0] - \
               np.array(_aggregates(base_count.sum(axis=0, keepdims=True)))[:, 0]
    rng = np.random.default_rng(seed)
    differences = np.empty((reps, 3), dtype=float)
    for offset in range(0, reps, 128):
        n = min(128, reps - offset)
        weights = rng.multinomial(n_groups, np.full(n_groups, 1.0/n_groups), size=n)
        target_sum = weights @ target_count
        base_sum = weights @ base_count
        differences[offset:offset+n, :] = (
            np.column_stack(_aggregates(target_sum)) -
            np.column_stack(_aggregates(base_sum))
        )
    return [
        {
            "comparison": "fold_selected_rule_minus_context",
            "cluster": "source_article_id", "clusters": n_groups,
            "reps": reps, "metric": name, "observed_difference": float(observed[i]),
            "bootstrap_mean_difference": float(differences[:, i].mean()),
            "ci95_low": float(np.quantile(differences[:, i], 0.025)),
            "ci95_high": float(np.quantile(differences[:, i], 0.975)),
        }
        for i, name in enumerate(("macro_f1", "human_fpr", "accuracy"))
    ]


def evaluate(frame: pd.DataFrame, bootstrap_reps: int, seed: int) -> dict:
    y = frame["y_true"].to_numpy(int)
    s = frame["sentence_p_machine"].to_numpy(float)
    c = frame["context_p_machine"].to_numpy(float)
    words = frame["target_words"].to_numpy(int)
    scores = {r: predict_rule(r, s, c, words) for r in RULES}
    selected_score, selected_rows = crossfit_selector(frame, scores)
    scores["fold_selected_rule"] = selected_score

    overall = [{"model": name, "protocol": "matched_paired_oof_threshold_0.5",
                **metrics(y, p)} for name, p in scores.items()]
    fold_rows = []
    for fold in range(5):
        mask = frame["fold"].to_numpy(int) == fold
        for name, p in scores.items():
            fold_rows.append({
                "model": name, "fold": fold,
                **metrics(y[mask], p[mask]),
            })
    by_length = length_analysis(frame, scores)
    transitions = transition_rows(frame, scores["sentence_matched"], scores["context"])
    bootstrap = clustered_bootstrap(frame, selected_score, c, bootstrap_reps, seed)

    score_dict = {x["model"]: x for x in overall}
    fold_df = pd.DataFrame(fold_rows)
    context_fold = fold_df.loc[fold_df["model"].eq("context")].set_index("fold")
    selected_fold = fold_df.loc[fold_df["model"].eq("fold_selected_rule")].set_index("fold")
    delta_fold = selected_fold["macro_f1"] - context_fold["macro_f1"]
    delta_fpr = score_dict["fold_selected_rule"]["human_fpr"] - score_dict["context"]["human_fpr"]
    delta_f1 = score_dict["fold_selected_rule"]["macro_f1"] - score_dict["context"]["macro_f1"]
    f1_ci = next(item for item in bootstrap if item["metric"] == "macro_f1")

    # A conservative *exploratory* go/no-go guideline. This is not a
    # pre-registered scientific confirmatory test or a deployment guarantee.
    go = bool(
        delta_f1 >= 0.002
        and delta_fpr <= 0.002
        and int((delta_fold >= -1e-12).sum()) >= 3
        and f1_ci["ci95_low"] > 0
    )
    decision = {
        "stage": "E2", "status": "COMPLETED",
        "evaluation_protocol": "five_fold_paired_development_oof_not_independent_test",
        "n_rows": int(len(frame)),
        "source_groups": int(frame["source_article_id"].nunique()),
        "article_records": int(frame["article_key"].nunique()),
        "folds": 5, "probability_threshold": THRESHOLD,
        "selected_rules": {
            str(row["heldout_fold"]): row["rule_selected_from_other_four_folds"]
            for row in selected_rows
        },
        "sentence_macro_f1": score_dict["sentence_matched"]["macro_f1"],
        "context_macro_f1": score_dict["context"]["macro_f1"],
        "fold_selected_macro_f1": score_dict["fold_selected_rule"]["macro_f1"],
        "fold_selected_minus_context_macro_f1": delta_f1,
        "fold_selected_minus_context_human_fpr": delta_fpr,
        "nonnegative_fold_macro_f1_differences": int((delta_fold >= -1e-12).sum()),
        "source_bootstrap_macro_f1_delta_ci95": [f1_ci["ci95_low"], f1_ci["ci95_high"]],
        "exploratory_gate": "consider_E3" if go else "stop_E3_keep_frozen_stacking",
        "gate_criteria": {
            "macro_f1_improvement_at_least": 0.002,
            "human_fpr_increase_no_more_than": 0.002,
            "nonnegative_fold_count_at_least": 3,
            "source_bootstrap_macro_f1_delta_lower_bound_positive": True,
        },
        "warnings": [
            "Previous OOF and Novel analyses are development data, not blind test.",
            "Fold-selected rules use only other-fold labels, but upstream base models are not nested retrained; interpret as exploratory cross-fitting.",
            "Fixed-rule pooled metrics are descriptive; choosing the best by overall OOF score is post hoc.",
            "Do not directly compare these strictly window-purged OOF scores against Stacking trained with different purging and base-model protocols.",
            "Do not use labels to select inference-time samples; all rule features are observable at prediction time.",
            "No new model checkpoint, test prediction or official submission artifact is produced.",
        ],
    }
    return {
        "overall": pd.DataFrame(overall),
        "by_fold": fold_df,
        "by_length": pd.DataFrame(by_length),
        "transitions": pd.DataFrame(transitions),
        "fold_selection": pd.DataFrame(selected_rows),
        "bootstrap": pd.DataFrame(bootstrap),
        "decision": decision,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("results/context_oof"))
    p.add_argument("--output-dir", type=Path, default=Path("results/e2_selective_context"))
    p.add_argument("--bootstrap-reps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    frame = load_strict_pair(args.input_dir)
    result = evaluate(frame, bootstrap_reps=args.bootstrap_reps, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in ("overall", "by_fold", "by_length", "transitions", "fold_selection", "bootstrap"):
        result[key].to_csv(args.output_dir / f"{key}.csv", index=False, encoding="utf-8")
    (args.output_dir / "decision.json").write_text(
        json.dumps(result["decision"], ensure_ascii=False, indent=2), encoding="utf-8")
    print("E2 no-retraining analysis; 52,864 paired OOF rows expected by existing reference.")
    print(result["overall"][["model", "n", "macro_f1", "accuracy", "fp_human_to_machine", "human_fpr"]]
          .to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))
    print(f"Aggregate-only results: {args.output_dir}")


if __name__ == "__main__":
    main()
