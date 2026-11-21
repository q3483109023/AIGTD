import unittest

import pandas as pd

from scripts.build_baseline_comparison import (
    build_comparison,
    build_context_fold_diagnostics,
)


class BaselineComparisonTests(unittest.TestCase):
    def test_protocols_remain_separate_and_ranking_only_is_unthresholded(self):
        fusion = pd.DataFrame(
            [
                {
                    "split": "novel_validation_fold_matched",
                    "model": "stacker_fold_matched",
                    "n": 100,
                    "accuracy": 0.91,
                    "macro_f1": 0.90,
                    "f1_human": 0.88,
                    "f1_machine": 0.92,
                    "precision_machine": 0.93,
                    "recall_machine": 0.91,
                    "roc_auc": 0.95,
                    "average_precision": 0.96,
                },
                {
                    "split": "novel_5fold_ensemble_ranking_only",
                    "model": "roberta_5fold_mean_z_ranking_only",
                    "n": 100,
                    "roc_auc": 0.97,
                    "average_precision": 0.98,
                },
            ]
        )
        context_overall = pd.DataFrame(
            [
                {
                    "model": "context",
                    "split": "pooled_oof_fixed_threshold_0.5",
                    "n": 500,
                    "accuracy": 0.90,
                    "macro_f1": 0.89,
                    "f1_human": 0.87,
                    "f1_machine": 0.91,
                    "roc_auc": 0.94,
                    "average_precision": 0.95,
                },
                {
                    "model": "context",
                    "split": "pooled_oof_fold_standardized_margin",
                    "n": 500,
                    "roc_auc": 0.96,
                    "average_precision": 0.97,
                },
            ]
        )
        context_novel = pd.DataFrame(
            [
                {
                    "model": "context",
                    "split": "novel_validation_mean_probability_over_5_fold_models",
                    "n": 100,
                    "accuracy": 0.92,
                    "macro_f1": 0.91,
                    "f1_human": 0.89,
                    "f1_machine": 0.93,
                    "roc_auc": 0.98,
                    "average_precision": 0.99,
                }
            ]
        )
        svm = pd.DataFrame(
            [
                {
                    "experiment": "svm_tfidf",
                    "threshold": 0.52,
                    "novel_validation_count": 100,
                    "novel_accuracy": 0.88,
                    "novel_macro_f1": 0.87,
                }
            ]
        )
        bert = pd.DataFrame(
            [
                {
                    "experiment": "bert",
                    "threshold": 0.80,
                    "novel_validation_count": 100,
                    "novel_accuracy": 0.90,
                    "novel_macro_f1": 0.89,
                }
            ]
        )

        result = build_comparison(fusion, context_overall, context_novel, svm, bert)
        stacker = result.loc[result["model"].eq("Stacking")].iloc[0]
        ranking = result.loc[
            result["protocol_id"].eq("novel_5fold_ranking_only")
        ].iloc[0]
        context_raw = result.loc[
            result["protocol_id"].eq("context_pooled_oof_fixed_05")
        ].iloc[0]
        context_standardized = result.loc[
            result["protocol_id"].eq("context_pooled_oof_standardized_ranking")
        ].iloc[0]
        single_split = result.loc[
            result["protocol_id"].eq("single_model_novel_threshold_selected")
        ]

        self.assertEqual(stacker["protocol_id"], "novel_fold_matched")
        self.assertEqual(ranking["model"], "RoBERTa")
        self.assertTrue(pd.isna(ranking["macro_f1"]))
        self.assertEqual(context_raw["threshold_protocol"], "Fixed probability threshold 0.5")
        self.assertTrue(pd.isna(context_standardized["macro_f1"]))
        self.assertEqual(len(single_split), 2)
        self.assertTrue(single_split["threshold_protocol"].str.contains("selected on Novel").all())

    def test_context_fold_table_keeps_each_fold_including_fold4(self):
        source = pd.DataFrame(
            [
                {
                    "model": model,
                    "fold": fold,
                    "n": 10,
                    "accuracy": 0.8,
                    "macro_f1": 0.7,
                    "f1_human": 0.6,
                    "f1_machine": 0.8,
                    "roc_auc": 0.9 if fold < 4 else 0.6,
                    "average_precision": 0.9 if fold < 4 else 0.5,
                }
                for model in ("sentence_matched", "context")
                for fold in range(5)
            ]
        )
        result = build_context_fold_diagnostics(source)

        self.assertEqual(len(result), 10)
        fold4_context = result.loc[
            result["fold"].eq(4) & result["model"].eq("Context")
        ].iloc[0]
        self.assertEqual(fold4_context["roc_auc"], 0.6)
        self.assertEqual(fold4_context["average_precision"], 0.5)


if __name__ == "__main__":
    unittest.main()
