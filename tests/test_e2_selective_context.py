"""Synthetic E2 tests: paired integrity, fixed-label-free rules and source grouping."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiment_e2_selective_context import (
    RULES, choose_rule_on_other_folds, clustered_bootstrap, crossfit_selector,
    evaluate, load_strict_pair, metrics, predict_rule,
)


def make_pair(n: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 5 original folds and exactly one source group per fold's micro-groups.
    k = np.arange(n)
    folds = k % 5
    labels = (k % 3 != 0).astype(int)
    sentence_p = np.where(labels == 1, 0.8, 0.2).astype(float)
    context_p = sentence_p.copy()
    # Deterministic disagreement examples; no rule features use the labels.
    context_p[k % 7 == 0] = 1 - context_p[k % 7 == 0]
    sentence_p[k % 11 == 0] = 1 - sentence_p[k % 11 == 0]
    base = pd.DataFrame({
        "oof_row_id": k, "fold": folds, "y_true": labels,
        "source_article_id": [f"source_{i % 5}_{i // 10}" for i in k],
        "article_key": [f"record_{i}" for i in k],
        "sentence_index": np.zeros(n, dtype=int),
        "text": ["short text" if i % 2 == 0 else "a much longer example sentence for testing" for i in k],
        "context_text": ["[PREVIOUS] [START] [TARGET] target [NEXT] [END]"] * n,
        "label_id": labels,
    })
    # Each source group can only contain members with the same fold.
    a = base.copy()
    b = base.copy()
    a["context_p_machine"] = sentence_p
    b["context_p_machine"] = context_p
    a["context_logit_margin"] = np.log(sentence_p / (1 - sentence_p))
    b["context_logit_margin"] = np.log(context_p / (1 - context_p))
    return a, b


class E2Tests(unittest.TestCase):
    def test_fixed_rules_do_not_use_labels(self):
        s = np.array([0.1, 0.8, 0.4, 0.2])
        c = np.array([0.9, 0.2, 0.9, 0.8])
        words = np.array([5, 15, 24, 32])
        self.assertEqual(set(RULES), set((
            "sentence_matched", "context", "mean_50", "context_75",
            "short_context_12", "short_context_24",
            "long_human_veto_20", "long_human_veto_32",
        )))
        np.testing.assert_allclose(predict_rule("mean_50", s, c, words), (s + c) / 2)
        np.testing.assert_allclose(predict_rule("short_context_12", s, c, words),
                                   [0.9, 0.8, 0.4, 0.2])
        np.testing.assert_allclose(predict_rule("long_human_veto_20", s, c, words),
                                   [0.9, 0.2, 0.4, 0.2])

    def test_metrics_human_fpr(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.6, 0.4, 0.7, 0.2])
        m = metrics(y, p)
        self.assertEqual((m["tn"], m["fp_human_to_machine"], m["fn_machine_to_human"], m["tp"]),
                         (1, 1, 1, 1))
        self.assertAlmostEqual(m["human_fpr"], 0.5)
        self.assertAlmostEqual(m["macro_f1"], 0.5)

    def test_paired_source_and_text_integrity(self):
        a, b = make_pair()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a.to_csv(root / "sentence_matched_oof_predictions.csv", index=False)
            b.to_csv(root / "context_oof_predictions.csv", index=False)
            frame = load_strict_pair(root)
            self.assertEqual(len(frame), len(a))
            self.assertEqual(frame["source_article_id"].nunique(), 25)
            b.loc[0, "text"] = "CORRUPTED"
            b.to_csv(root / "context_oof_predictions.csv", index=False)
            with self.assertRaisesRegex(ValueError, "metadata: text"):
                load_strict_pair(root)

    def test_source_group_cross_fold_rejected(self):
        a, b = make_pair()
        a.loc[0, "source_article_id"] = "source_1_0"
        b.loc[0, "source_article_id"] = "source_1_0"
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a.to_csv(root / "sentence_matched_oof_predictions.csv", index=False)
            b.to_csv(root / "context_oof_predictions.csv", index=False)
            with self.assertRaisesRegex(ValueError, "crosses original"):
                load_strict_pair(root)

    def test_fifth_fold_rule_selection_does_not_use_its_labels(self):
        frame = pd.DataFrame({
            "fold": np.tile(np.arange(5), 10),
            "y_true": np.tile(np.array([0, 1, 1, 0, 1]), 10),
        })
        s = np.tile(np.array([0.2, 0.8, 0.4, 0.2, 0.9]), 10)
        c = np.tile(np.array([0.6, 0.7, 0.8, 0.2, 0.7]), 10)
        length = np.full(len(frame), 14)
        scores = {rule: predict_rule(rule, s, c, length) for rule in RULES}
        _, chosen = crossfit_selector(frame, scores)
        flip = frame.copy()
        flip.loc[flip["fold"] == 2, "y_true"] = 1 - flip.loc[flip["fold"] == 2, "y_true"]
        _, changed = crossfit_selector(flip, scores)
        self.assertEqual(chosen[2]["rule_selected_from_other_four_folds"],
                         changed[2]["rule_selected_from_other_four_folds"])

    def test_bootstrap_identical_predictions_zero_delta(self):
        frame = pd.DataFrame({
            "y_true": [0, 1, 0, 1, 0, 1],
            "source_article_id": ["a", "a", "b", "b", "c", "c"],
        })
        score = np.array([0.2, 0.8, 0.9, 0.3, 0.1, 0.7])
        rows = clustered_bootstrap(frame, score, score.copy(), reps=25, seed=42)
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertAlmostEqual(r["observed_difference"], 0.0)
            self.assertAlmostEqual(r["ci95_low"], 0.0)
            self.assertAlmostEqual(r["ci95_high"], 0.0)

    def test_e2_aggregate_only_output_contract(self):
        a, b = make_pair()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a.to_csv(root / "sentence_matched_oof_predictions.csv", index=False)
            b.to_csv(root / "context_oof_predictions.csv", index=False)
            frame = load_strict_pair(root)
            result = evaluate(frame, bootstrap_reps=20, seed=2026)
            self.assertEqual(result["decision"]["status"], "COMPLETED")
            self.assertEqual(result["decision"]["n_rows"], 50)
            self.assertEqual(result["decision"]["folds"], 5)
            self.assertEqual(len(result["fold_selection"]), 5)
            self.assertEqual(len(result["overall"]), len(RULES) + 1)
            # No public artifact exposes row text/source IDs/individual predictions.
            for key in ("overall", "by_fold", "by_length", "transitions",
                        "fold_selection", "bootstrap"):
                df = result[key]
                self.assertTrue(set(df.columns).isdisjoint(
                    {"text", "article_id", "article_key", "source_article_id",
                     "oof_row_id", "sentence_index", "context_text",
                     "sentence_p_machine", "context_p_machine", "y_true"}))


if __name__ == "__main__":
    unittest.main()
