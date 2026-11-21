#!/usr/bin/env python3
"""E2: evaluate frozen >=20-word Human veto on Novel without tuning it.

This is an exploratory follow-up on an already-reused DEVELOPMENT validation
set, not an independent/blind test. Evaluate two *separate* aggregation
protocols: 5-fold mean probabilities (historical Context Novel convention),
and one label-free source-group-assigned fold per target (OOF-like sensitivity).
Never pool all five Novel prediction rows as independent samples.
No new training, checkpoint inference or row-level output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.experiment_e2_selective_context import (
        THRESHOLD, clustered_bootstrap, metrics, predict_rule, require,
    )
    from scripts.train_oof_fusion import assign_source_groups_balanced
except ImportError:
    from experiment_e2_selective_context import THRESHOLD, clustered_bootstrap, metrics, predict_rule, require
    from train_oof_fusion import assign_source_groups_balanced


FIXED_RULES = ("sentence_matched", "context", "mean_50", "long_human_veto_20")
REQUIRED = {"novel_row_id", "fold", "y_true", "context_p_machine", "text",
            "article_key", "sentence_index"}


def load_novel_pair(root: Path, n_splits: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    paths = (
        root / "sentence_matched_novel_fold_predictions.csv",
        root / "context_novel_fold_predictions.csv",
    )
    a, b = (pd.read_csv(path, encoding="utf-8-sig", low_memory=False) for path in paths)
    source_col = next(
        (col for col in ("source_article_id", "article_id") if col in a and col in b),
        None,
    )
    require(source_col is not None, "Novel requires a shared source_article_id or article_id")
    must_have = REQUIRED | {source_col}
    for name, frame in (("sentence", a), ("context", b)):
        missing = must_have - set(frame.columns)
        require(not missing, f"{name} Novel missing columns: {sorted(missing)}")
        require(not frame[list(must_have)].isna().any().any(),
                f"{name} Novel has missing key/source/label/text/score")
        require(not frame.duplicated(["novel_row_id", "fold"]).any(),
                f"{name} Novel has duplicate (novel_row_id,fold)")
        require(set(frame["fold"].astype(int)) == set(range(n_splits)),
                f"{name} Novel must cover all {n_splits} trained folds")
        require(frame.groupby("novel_row_id")["fold"].nunique().eq(n_splits).all(),
                f"{name} Novel does not have exactly {n_splits} fold scores per row")
        require(frame.groupby("novel_row_id").size().eq(n_splits).all(),
                f"{name} Novel duplicate or missing scores")
        require(frame.groupby("novel_row_id")["y_true"].nunique().eq(1).all(),
                f"{name} Novel labels vary across model folds")
        require(frame.groupby("novel_row_id")[source_col].nunique().eq(1).all(),
                f"{name} Novel source ID varies across model folds")
        p = pd.to_numeric(frame["context_p_machine"], errors="coerce").to_numpy(float)
        require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(),
                f"{name} Novel has invalid p_machine scores")
        require(set(frame["y_true"].astype(int).unique()).issubset({0, 1}),
                f"{name} Novel has invalid class labels")

    # Each row has one prediction from each of five models. Match by compound
    # (novel_row_id, fold), never by sentence text or by file ordering.
    a = a.sort_values(["novel_row_id", "fold"], kind="stable").reset_index(drop=True)
    b = b.sort_values(["novel_row_id", "fold"], kind="stable").reset_index(drop=True)
    for col in ("novel_row_id", "fold", "y_true", "article_key",
                "sentence_index", "text", source_col):
        require(a[col].astype(str).equals(b[col].astype(str)),
                f"Novel Sentence/Context mismatch: {col}")
    for name, frame in (("sentence", a), ("context", b)):
        for col in ("article_key", "sentence_index", "text", source_col):
            require(frame.groupby("novel_row_id")[col].nunique().eq(1).all(),
                    f"{name} Novel {col} changes across model folds")
        require(frame.groupby("article_key")[source_col].nunique().eq(1).all(),
                f"{name} Novel article record spans source IDs")
    a["context_p_machine"] = a["context_p_machine"].astype(float)
    b["context_p_machine"] = b["context_p_machine"].astype(float)
    return a, b, source_col


def make_unique_protocols(
    a: pd.DataFrame, b: pd.DataFrame, source_col: str,
    n_splits: int = 5,
) -> dict[str, pd.DataFrame]:
    # Source IDs can repeat across different article records. Derive the fold
    # assignment from exactly one row per unique target (not 5x prediction
    # rows), otherwise a model-fold row is mistaken for another example.
    base = a.drop_duplicates("novel_row_id", keep="first").copy()
    base["source_article_id"] = base[source_col].astype(str)
    base = base.sort_values("novel_row_id", kind="stable").reset_index(drop=True)
    require(base["novel_row_id"].is_unique, "Non-unique Novel target rows")
    require(len(base) * n_splits == len(a) == len(b),
            "Incorrect unique Novel target count")

    def pack(frame: pd.DataFrame, s: np.ndarray, c: np.ndarray) -> pd.DataFrame:
        output = frame[["novel_row_id", "y_true", "source_article_id", "article_key", "text"]].copy()
        output["target_words"] = output["text"].astype(str).str.split().str.len().astype(int)
        output["sentence_p_machine"] = s
        output["context_p_machine"] = c
        require((output["target_words"] > 0).all(), "Empty target sentence")
        require(np.isfinite(s).all() and np.isfinite(c).all(), "Missing Novel scores")
        return output.drop(columns=["text"])

    mean_a = a.groupby("novel_row_id", sort=True)["context_p_machine"].mean()
    mean_b = b.groupby("novel_row_id", sort=True)["context_p_machine"].mean()
    ids = base["novel_row_id"]
    mean5 = pack(base, mean_a.reindex(ids).to_numpy(float),
                 mean_b.reindex(ids).to_numpy(float))

    base["assigned_fold"] = assign_source_groups_balanced(
        base["source_article_id"], n_splits)
    require(base.groupby("source_article_id")["assigned_fold"].nunique().eq(1).all(),
            "One source ID assigned to multiple folds")
    # Each target contributes one score to both models, from the SAME fold.
    assigned = base[["novel_row_id", "assigned_fold"]]
    selected_a = a.merge(assigned, on="novel_row_id", validate="many_to_one")
    selected_b = b.merge(assigned, on="novel_row_id", validate="many_to_one")
    selected_a = selected_a.loc[selected_a["fold"].eq(selected_a["assigned_fold"])]
    selected_b = selected_b.loc[selected_b["fold"].eq(selected_b["assigned_fold"])]
    selected_a = selected_a.set_index("novel_row_id").reindex(ids)
    selected_b = selected_b.set_index("novel_row_id").reindex(ids)
    require(not selected_a["context_p_machine"].isna().any() and
            not selected_b["context_p_machine"].isna().any(),
            "Source-fold selection dropped Novel targets")
    matched = pack(base, selected_a["context_p_machine"].to_numpy(float),
                   selected_b["context_p_machine"].to_numpy(float))
    matched["assigned_fold"] = base["assigned_fold"].to_numpy(int)
    return {"novel_five_fold_mean": mean5, "novel_source_fold_matched": matched}


def evaluate_fixed(protocol_name: str, frame: pd.DataFrame,
                   bootstrap_reps: int, seed: int) -> dict:
    y = frame["y_true"].to_numpy(int)
    s = frame["sentence_p_machine"].to_numpy(float)
    c = frame["context_p_machine"].to_numpy(float)
    words = frame["target_words"].to_numpy(int)
    rules = {r: predict_rule(r, s, c, words) for r in FIXED_RULES}
    rows = [{
        "protocol": protocol_name, "model": name,
        **metrics(y, p),
    } for name, p in rules.items()]
    by_cls = []
    for label, mask in (("human", y == 0), ("machine", y == 1)):
        for model in ("context", "long_human_veto_20"):
            m = metrics(y[mask], rules[model][mask])
            by_cls.append({
                "protocol": protocol_name, "true_class": label, "model": model,
                "rows": int(mask.sum()), "correct": m["tn"] + m["tp"],
                "wrong": m["fp_human_to_machine"] + m["fn_machine_to_human"],
            })
    # Each Novel target appears once in the cluster bootstrap.
    boot = clustered_bootstrap(frame, rules["long_human_veto_20"],
                               rules["context"], bootstrap_reps, seed)
    for entry in boot:
        entry["comparison"] = "fixed_veto20_minus_context"
        entry["protocol"] = protocol_name
    scores = {x["model"]: x for x in rows}
    context, veto = scores["context"], scores["long_human_veto_20"]
    f1_ci = next(r for r in boot if r["metric"] == "macro_f1")
    decision = {
        "protocol": protocol_name,
        "n": len(frame), "source_groups": frame["source_article_id"].nunique(),
        "context_macro_f1": context["macro_f1"],
        "fixed_veto20_macro_f1": veto["macro_f1"],
        "delta_macro_f1": veto["macro_f1"] - context["macro_f1"],
        "delta_human_fpr": veto["human_fpr"] - context["human_fpr"],
        "delta_machine_recall": veto["machine_recall"] - context["machine_recall"],
        "source_bootstrap_delta_macro_f1_ci95": [f1_ci["ci95_low"], f1_ci["ci95_high"]],
        "fixed_rule_unchanged_from_e2": True,
        "interpretation": "exploratory_reused_Novel_not_independent_test",
    }
    return {"metrics": rows, "class_errors": by_cls, "bootstrap": boot,
            "decision": decision}


def compare_mean_reference(
    mean5: dict, reference: Path, tolerance: float = 1e-9,
) -> None:
    require(reference.is_file(), f"Missing historical Novel baseline summary: {reference}")
    history = pd.read_csv(reference)
    for model in ("sentence_matched", "context"):
        match = history[history["model"].eq(model)]
        require(len(match) == 1, f"Historical baseline '{model}' must be unique")
        current = next(x for x in mean5["metrics"] if x["model"] == model)
        for metric in ("accuracy", "macro_f1"):
            require(abs(float(match.iloc[0][metric]) - current[metric]) <= tolerance,
                    f"Five-fold mean {model} {metric} differs from established baseline")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("results/context_oof"))
    p.add_argument("--reference", type=Path,
                   default=Path("results/context_oof_analysis/novel_metrics.csv"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/e2_novel_fixed_rule"))
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--expected-targets", type=int, default=11682)
    args = p.parse_args()
    require(args.bootstrap_reps >= 10, "bootstrap reps must be >=10")
    a, b, source_col = load_novel_pair(args.input_dir)
    protocols = make_unique_protocols(a, b, source_col)
    require(all(len(frame) == args.expected_targets for frame in protocols.values()),
            f"Novel unique target count differs from {args.expected_targets}")
    result = {name: evaluate_fixed(name, frame, args.bootstrap_reps, args.seed)
              for name, frame in protocols.items()}
    compare_mean_reference(result["novel_five_fold_mean"], args.reference)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("metrics", "class_errors", "bootstrap"):
        pd.DataFrame([row for v in result.values() for row in v[name]]).to_csv(
            args.output_dir / f"{name}.csv", index=False)
    summary = {
        "status": "PASS",
        "evaluation": "reused_development_Novel_validation_not_blind_test",
        "novel_targets": args.expected_targets,
        "paired_fold_prediction_rows_per_model": len(a),
        "n_splits": 5,
        "rule": "long_human_veto_20",
        "rule_threshold": THRESHOLD,
        "reference_five_fold_mean_baselines_match": True,
        "protocol_results": [v["decision"] for v in result.values()],
        "warnings": [
            "The fixed 20-word rule came from earlier OOF analysis; Novel has also been used in earlier model development.",
            "Five-fold probability mean and source-fold matching are DIFFERENT prediction protocols; never merge or select based on whichever scores better.",
            "All Novel rows have labels and may not reflect the unlabeled test distribution.",
            "The E2 rule uses strict-window-purged Sentence/Context models; it is NOT the frozen Stacking inference model.",
            "No label or source ID may be used as a selective inference rule feature.",
            "Source-cluster bootstrap assumes pre-trained predictors and does not account for previous exploration or retraining variance.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("E2 fixed 20-word Human-veto replication on Novel (not blind test).")
    print(pd.DataFrame([row for v in result.values() for row in v["metrics"]])[
        ["protocol", "model", "n", "macro_f1", "accuracy",
         "fp_human_to_machine", "fn_machine_to_human", "human_fpr", "machine_recall"]
    ].to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Aggregate-only local output: {args.output_dir}")


if __name__ == "__main__":
    main()
