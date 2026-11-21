"""Synthetic/privacy-safe regression checks for the Stage-1 freeze tools."""
from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.check_stage1_artifacts import has_weights
from scripts.validate_test_predictions import validate
from scripts.verify_stacking_aggregates import verify


ROOT = Path(__file__).resolve().parents[1]


class Stage1PredictionValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        # Source IDs deliberately repeat; records must NOT be regrouped by id.
        self.raw = [
            {"id": "same-id", "domain": "news", "sent": ["First.", "Second."]},
            {"id": "same-id", "domain": "news", "sent": ["Third."]},
        ]
        self.pred = [
            {**self.raw[0], "predicted_labels": ["human", "machine"]},
            {**self.raw[1], "predicted_labels": ["machine"]},
        ]
        self.raw_path = self.folder / "raw.json"
        self.json_path = self.folder / "pred.json"
        self.csv_path = self.folder / "pred.csv"
        self._write()

    def _write(self):
        self.raw_path.write_text(json.dumps(self.raw), encoding="utf-8")
        self.json_path.write_text(json.dumps(self.pred), encoding="utf-8")
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["article_key", "sentence_index", "text",
                                            "predicted_label", "predicted_id", "p_machine"])
            w.writeheader()
            for key, i, sentence, label, prob in [
                ("record_00000000", 0, "First.", "human", 0.2),
                ("record_00000000", 1, "Second.", "machine", 0.8),
                ("record_00000001", 0, "Third.", "machine", 0.9),
            ]:
                w.writerow({"article_key": key, "sentence_index": i,
                            "text": sentence, "predicted_label": label,
                            "predicted_id": int(label == "machine"), "p_machine": prob})

    def test_duplicate_source_ids_and_order_are_preserved(self):
        report = validate(self.raw_path, self.json_path, self.csv_path, 2, 3, 0.5)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["labels"], {"human": 1, "machine": 2})

    def test_article_order_change_fails(self):
        self.pred.reverse()
        self._write()
        with self.assertRaisesRegex(ValueError, "sentence order/content"):
            validate(self.raw_path, self.json_path, self.csv_path, 2, 3)

    def test_missing_or_incorrect_label_fails(self):
        self.pred[0]["predicted_labels"][1] = "AI"
        self._write()
        with self.assertRaisesRegex(ValueError, "invalid label"):
            validate(self.raw_path, self.json_path, self.csv_path, 2, 3)

    def test_csv_mismatch_fails(self):
        self.pred[1]["predicted_labels"][0] = "human"
        self._write()
        with self.assertRaisesRegex(ValueError, "CSV label"):
            validate(self.raw_path, self.json_path, self.csv_path, 2, 3)

    def test_count_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "sentence count"):
            validate(self.raw_path, self.json_path, self.csv_path, 2, 4)


class Stage1AggregateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        original = ROOT / "results/stacking_error_analysis"
        self.reference = ROOT / "results/oof_fusion/metrics_comparison.csv"
        for name in ("metrics_all_models.csv", "stacker_vs_roberta_groups.csv",
                     "paired_mcnemar_sentence_level.csv", "cluster_bootstrap_differences.csv"):
            shutil.copyfile(original / name, self.folder / name)
        # The historical tracked snapshot has raw-fold AUC/AP for base models.
        # Replacing only those with fold-standardised reference values emulates
        # the corrected aggregate-only run. Real runs must regenerate, not patch.
        with self.reference.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        matched = {r["model"]: r for r in rows
                   if r["split"] == "novel_validation_fold_matched"}
        path = self.folder / "metrics_all_models.csv"
        with path.open(encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            metrics = list(r)
            columns = r.fieldnames
        for entry in metrics:
            ref_key = {"roberta": "roberta_fold_matched",
                       "svm": "svm_fold_matched"}.get(entry["model"])
            if ref_key:
                for metric in ("roc_auc", "average_precision"):
                    entry[metric] = matched[ref_key][metric]
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows(metrics)

    def test_corrected_consistent_aggregates_pass(self):
        result = verify(self.folder, self.reference)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["stacker_only_correct_vs_roberta"], 266)
        self.assertEqual(result["roberta_only_correct"], 184)

    def test_wrong_pooled_auc_fails(self):
        path = self.folder / "metrics_all_models.csv"
        text = path.read_text(encoding="utf-8")
        text = text.replace("0.9581776173", "0.000001", 1)
        # Use fieldwise rewrite so the test does not rely on float formatting.
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            metrics = list(reader)
            fields = reader.fieldnames
        for row in metrics:
            if row["model"] == "roberta":
                row["roc_auc"] = "0.12345"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(metrics)
        with self.assertRaisesRegex(ValueError, "roc_auc"):
            verify(self.folder, self.reference)


class Stage1InventoryTests(unittest.TestCase):
    def test_empty_checkpoint_directory_is_not_a_weight_bundle(self):
        import scripts.check_stage1_artifacts as inventory
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(inventory, "ROOT", Path(folder)):
                (Path(folder) / "model").mkdir()
                self.assertFalse(has_weights("model"))
                (Path(folder) / "model" / "model.safetensors").write_bytes(b"placeholder")
                self.assertTrue(has_weights("model"))


if __name__ == "__main__":
    unittest.main()
