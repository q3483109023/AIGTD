"""Synthetic regression tests for fixed E2 Novel replication, no model training."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.validate_e2_fixed_rule_novel import (
    compare_mean_reference, evaluate_fixed, load_novel_pair, make_unique_protocols,
)


def synthetic_novel(n=20):
    rows_a, rows_b = [], []
    for i in range(n):
        label = int(i % 3 != 0)
        text = " ".join(["hello"] * (25 if i % 2 == 0 else 7))
        group = f"source_{i // 2}"
        for fold in range(5):
            # Moderate fold differences so the five-fold mean is not the same
            # operation as selecting a single source-matched fold.
            ps = (0.35 if label == 0 else 0.7) + 0.03 * fold
            pc = (0.75 if label == 0 else 0.8) - 0.02 * fold
            base = {
                "novel_row_id": i, "fold": fold, "y_true": label,
                "text": text, "article_key": f"record_{i}",
                "article_id": group, "source_article_id": group,
                "sentence_index": 0,
                "context_text": "[PREVIOUS] [START] [TARGET] " + text + " [NEXT] [END]",
            }
            rows_a.append({**base, "context_p_machine": ps})
            rows_b.append({**base, "context_p_machine": pc})
    return pd.DataFrame(rows_a), pd.DataFrame(rows_b)


def save_pair(path: Path, a: pd.DataFrame, b: pd.DataFrame):
    a.to_csv(path / "sentence_matched_novel_fold_predictions.csv", index=False)
    b.to_csv(path / "context_novel_fold_predictions.csv", index=False)


class NovelFixedRuleTests(unittest.TestCase):
    def test_load_pair_rejects_label_mismatch(self):
        a, b = synthetic_novel()
        b.loc[0, "y_true"] = 1 - int(b.loc[0, "y_true"])
        with tempfile.TemporaryDirectory() as p:
            save_pair(Path(p), a, b)
            with self.assertRaisesRegex(ValueError, "labels vary|mismatch"):
                load_novel_pair(Path(p))

    def test_load_pair_rejects_duplicate_fold_score(self):
        a, b = synthetic_novel()
        b = pd.concat([b, b.iloc[[0]]], ignore_index=True)
        with tempfile.TemporaryDirectory() as p:
            save_pair(Path(p), a, b)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_novel_pair(Path(p))

    def test_load_pair_rejects_text_mismatch(self):
        a, b = synthetic_novel()
        b.loc[0, "text"] = "CHANGED"
        with tempfile.TemporaryDirectory() as p:
            save_pair(Path(p), a, b)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_novel_pair(Path(p))

    def test_unique_target_coverage_and_source_grouping(self):
        a, b = synthetic_novel()
        with tempfile.TemporaryDirectory() as p:
            save_pair(Path(p), a.sample(frac=1, random_state=2),
                      b.sample(frac=1, random_state=3))
            aa, bb, source = load_novel_pair(Path(p))
            protocols = make_unique_protocols(aa, bb, source)
            self.assertEqual(set(protocols), {"novel_five_fold_mean", "novel_source_fold_matched"})
            for frame in protocols.values():
                self.assertEqual(len(frame), 20)
                self.assertTrue(frame["novel_row_id"].is_unique)
            matched = protocols["novel_source_fold_matched"]
            self.assertTrue(matched.groupby("source_article_id")["assigned_fold"].nunique().eq(1).all())
            for _, row in matched.iterrows():
                source_score = aa.loc[
                    aa["novel_row_id"].eq(row["novel_row_id"]) &
                    aa["fold"].eq(row["assigned_fold"]),
                    "context_p_machine",
                ].iloc[0]
                self.assertAlmostEqual(float(row["sentence_p_machine"]), float(source_score))

    def test_fixed_20_word_rule_applied_without_retuning(self):
        a, b = synthetic_novel()
        with tempfile.TemporaryDirectory() as p:
            save_pair(Path(p), a, b)
            aa, bb, source = load_novel_pair(Path(p))
            proto = make_unique_protocols(aa, bb, source)["novel_five_fold_mean"]
            output = evaluate_fixed("novel_five_fold_mean", proto, bootstrap_reps=20, seed=3)
            self.assertEqual(output["decision"]["fixed_rule_unchanged_from_e2"], True)
            baseline = next(r for r in output["metrics"] if r["model"] == "context")
            veto = next(r for r in output["metrics"] if r["model"] == "long_human_veto_20")
            # On synthetic long Human targets, Context wrongly calls Machine;
            # Sentence correctly calls Human; this intervention must happen.
            self.assertGreater(baseline["fp_human_to_machine"],
                               veto["fp_human_to_machine"])
            self.assertEqual(len(output["bootstrap"]), 3)

    def test_historical_reference_is_checked(self):
        a, b = synthetic_novel()
        with tempfile.TemporaryDirectory() as p:
            root = Path(p)
            save_pair(root, a, b)
            aa, bb, source = load_novel_pair(root)
            proto = make_unique_protocols(aa, bb, source)["novel_five_fold_mean"]
            result = evaluate_fixed("novel_five_fold_mean", proto, bootstrap_reps=10, seed=1)
            good = pd.DataFrame(result["metrics"])
            good = good.loc[good["model"].isin(["sentence_matched", "context"]),
                            ["model", "accuracy", "macro_f1"]]
            good.to_csv(root / "reference.csv", index=False)
            compare_mean_reference(result, root / "reference.csv")
            good.loc[good["model"].eq("context"), "macro_f1"] += 0.03
            good.to_csv(root / "reference.csv", index=False)
            with self.assertRaisesRegex(ValueError, "differs"):
                compare_mean_reference(result, root / "reference.csv")


if __name__ == "__main__":
    unittest.main()
