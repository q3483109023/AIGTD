#!/usr/bin/env python3
"""Fail-closed consistency check for aggregate-only corrected Stacking results.

No raw text, sentence-level predictions or article identifiers are exported.
This does NOT establish independent generalization: Novel Validation is a
development set. Run analyze_stacking_errors.py --aggregate-only first.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


MODEL_KEYS = {
    "stacker": "stacker_fold_matched",
    "roberta": "roberta_fold_matched",
    "svm": "svm_fold_matched",
    "veto": "svm_veto_fold_matched",
}
PAIRS = {"roberta": "stacker_vs_roberta", "svm": "stacker_vs_svm", "veto": "stacker_vs_veto"}
METRICS = ("accuracy", "macro_f1", "roc_auc", "average_precision")


def load_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing aggregate file: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        return list(reader)


def number(row: dict, key: str, name: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name}: invalid {key}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name}: non-finite {key}")
    return value


def close(actual: float, expected: float, name: str, tolerance: float = 1e-7) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{name}: {actual} != {expected}")


def verify(analysis_dir: Path, reference: Path, expected_rows: int = 11682) -> dict:
    observed = load_csv(
        analysis_dir / "metrics_all_models.csv",
        {"split", "model", "n", "accuracy", "macro_f1", "tn", "fp", "fn", "tp",
         "roc_auc", "average_precision"},
    )
    ref = load_csv(
        reference, {"split", "model", "n", "accuracy", "macro_f1", "tn", "fp",
                    "fn", "tp", "roc_auc", "average_precision"},
    )
    reference_rows = {
        row["model"]: row for row in ref
        if row["split"] == "novel_validation_fold_matched"
    }
    observed_rows = {row["model"]: row for row in observed}
    if len(observed_rows) != 4 or set(observed_rows) != set(MODEL_KEYS):
        raise ValueError("corrected metrics must contain exactly stacker/roberta/svm/veto")
    for model, reference_name in MODEL_KEYS.items():
        row = observed_rows[model]
        name = f"metrics/{model}"
        if row["split"] != "novel_validation_fold_matched":
            raise ValueError(f"{name}: protocol mismatch")
        if reference_name not in reference_rows:
            raise ValueError(f"reference missing {reference_name}")
        expected = reference_rows[reference_name]
        close(number(row, "n", name), expected_rows, f"{name}/n")
        counts = [number(row, key, name) for key in ("tn", "fp", "fn", "tp")]
        close(sum(counts), expected_rows, f"{name}/confusion sum")
        tn, fp, fn, tp = counts
        close(number(row, "accuracy", name), (tn + tp) / expected_rows, f"{name}/accuracy")
        human_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
        machine_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        close(number(row, "macro_f1", name), (human_f1 + machine_f1) / 2, f"{name}/macro_f1")
        for metric in ("n", "tn", "fp", "fn", "tp", "accuracy", "macro_f1"):
            close(number(row, metric, name), number(expected, metric, reference_name),
                  f"{name}/reference/{metric}")
        # Veto has no probability ranking in this protocol. Other AUC/AP values
        # MUST match the standardised-score reference, not raw pooled scores.
        if model != "veto":
            for metric in ("roc_auc", "average_precision"):
                close(number(row, metric, name), number(expected, metric, reference_name),
                      f"{name}/reference/{metric}", tolerance=5e-7)

    groups = load_csv(analysis_dir / "stacker_vs_roberta_groups.csv",
                      {"comparison_group", "n", "human", "machine"})
    by_group = {row["comparison_group"]: row for row in groups}
    group_names = {"both_correct", "both_wrong", "stacker_only_correct", "roberta_only_correct"}
    if len(groups) != 4 or set(by_group) != group_names:
        raise ValueError("expected four non-duplicated correctness groups")
    totals = {}
    for name, row in by_group.items():
        n = number(row, "n", name)
        close(n, number(row, "human", name) + number(row, "machine", name), f"{name}/class sum")
        totals[name] = n
    close(sum(totals.values()), expected_rows, "correctness groups/total")
    correct_stacker = number(observed_rows["stacker"], "tn", "stacker") + number(observed_rows["stacker"], "tp", "stacker")
    correct_roberta = number(observed_rows["roberta"], "tn", "roberta") + number(observed_rows["roberta"], "tp", "roberta")
    close(totals["both_correct"] + totals["stacker_only_correct"], correct_stacker, "groups/stacker")
    close(totals["both_correct"] + totals["roberta_only_correct"], correct_roberta, "groups/roberta")

    mcnemar = load_csv(analysis_dir / "paired_mcnemar_sentence_level.csv",
                       {"comparison", "stacker_only_correct", "other_only_correct",
                        "discordant_total", "exact_mcnemar_binomial_p"})
    by_pair = {row["comparison"]: row for row in mcnemar}
    if len(mcnemar) != 3 or set(by_pair) != set(PAIRS.values()):
        raise ValueError("expected three Stacking-vs-base McNemar comparisons")
    for other, pair in PAIRS.items():
        row = by_pair[pair]
        b, c = number(row, "stacker_only_correct", pair), number(row, "other_only_correct", pair)
        close(b + c, number(row, "discordant_total", pair), f"{pair}/discordance")
        other_correct = (number(observed_rows[other], "tn", other)
                         + number(observed_rows[other], "tp", other))
        close(b - c, correct_stacker - other_correct, f"{pair}/accuracy delta")
        p = number(row, "exact_mcnemar_binomial_p", pair)
        if not 0 <= p <= 1:
            raise ValueError(f"{pair}: invalid p-value")
    close(number(by_pair["stacker_vs_roberta"], "stacker_only_correct", "mcnemar"),
          totals["stacker_only_correct"], "McNemar/groups Stacking-only")
    close(number(by_pair["stacker_vs_roberta"], "other_only_correct", "mcnemar"),
          totals["roberta_only_correct"], "McNemar/groups RoBERTa-only")

    bootstrap = load_csv(analysis_dir / "cluster_bootstrap_differences.csv",
                         {"cluster_unit", "comparison", "metric", "clusters", "reps",
                          "ci95_low", "ci95_high", "status"})
    expected_pairs = {(cluster, f"stacker_minus_{other}", metric)
                      for cluster in ("article_id", "article_key")
                      for other in PAIRS for metric in ("accuracy", "macro_f1")}
    seen = set()
    for row in bootstrap:
        key = (row["cluster_unit"], row["comparison"], row["metric"])
        if key in seen or key not in expected_pairs:
            raise ValueError(f"unexpected/duplicate bootstrap row {key}")
        seen.add(key)
        if row["status"] != "ok" or number(row, "clusters", "bootstrap") < 20 or number(row, "reps", "bootstrap") <= 0:
            raise ValueError(f"invalid bootstrap run {key}")
        low, high = number(row, "ci95_low", "bootstrap"), number(row, "ci95_high", "bootstrap")
        if low > high or low < -1 or high > 1:
            raise ValueError(f"invalid bootstrap CI {key}")
    if seen != expected_pairs:
        raise ValueError("missing bootstrap comparison rows")

    return {
        "status": "PASS", "protocol": "novel_validation_fold_matched_development",
        "rows": expected_rows, "models": 4, "mcnemar_pairs": 3,
        "bootstrap_comparisons": len(bootstrap),
        "stacker_only_correct_vs_roberta": int(totals["stacker_only_correct"]),
        "roberta_only_correct": int(totals["roberta_only_correct"]),
        "interpretation": "Aggregate consistency only; not independent-test validation.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=Path("results/stacking_error_analysis_corrected"))
    parser.add_argument("--reference", type=Path, default=Path("results/oof_fusion/metrics_comparison.csv"))
    parser.add_argument("--expected-rows", type=int, default=11682)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()
    try:
        if args.expected_rows <= 0:
            raise ValueError("expected_rows must be positive")
        report = verify(args.analysis_dir, args.reference, args.expected_rows)
    except ValueError as exc:
        parser.exit(1, f"FAIL: {exc}\n")
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
