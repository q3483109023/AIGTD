"""Synthetic safety/regression tests for the existing OOF stacking bundle inference."""
from __future__ import annotations

import tempfile
import unittest
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.predict_oof_stacking import infer_test, replay_novel, score_fold
from scripts.validate_test_predictions import validate
from scripts.train_oof_fusion import assign_source_groups_balanced


class FakeMeta:
    def predict_proba(self, x):
        x = np.asarray(x)
        z = 0.1 + 0.7 * x[:, 0] - 0.4 * x[:, 1]
        p = 1. / (1. + np.exp(-z))
        return np.column_stack([1 - p, p])


class StackInferenceSyntheticTests(unittest.TestCase):
    def setUp(self):
        self.stats = {
            "svm": {str(k): {"mean": 1.0, "std": 2.0} for k in range(5)},
            "roberta": {str(k): {"mean": -1.0, "std": 4.0} for k in range(5)},
        }
        self.meta = {k: FakeMeta() for k in range(5)}
        self.config = {"stacker_threshold": 0.585}

    def test_score_order_and_fold_scaling(self):
        s, r, p = score_fold(self.meta[1], self.stats, 1, np.array([3., 1.]),
                             np.array([3., -1.]))
        np.testing.assert_allclose(s, [1., 0.])
        np.testing.assert_allclose(r, [1., 0.])
        np.testing.assert_allclose(p, FakeMeta().predict_proba(
            np.array([[1., 1.], [0., 0.]]))[:, 1])

    def test_balanced_source_group_folds_are_order_robust(self):
        source = pd.Series(["one", "two", "one", "three", "one", "three", "four"])
        assigned = assign_source_groups_balanced(source, 5)
        self.assertTrue((pd.DataFrame({"s": source, "f": assigned})
                         .groupby("s")["f"].nunique() == 1).all())
        rev = source.iloc[::-1].reset_index(drop=True)
        rev_assigned = assign_source_groups_balanced(rev, 5)
        a = pd.DataFrame({"s": source, "f": assigned}).drop_duplicates("s").set_index("s")["f"]
        b = pd.DataFrame({"s": rev, "f": rev_assigned}).drop_duplicates("s").set_index("s")["f"]
        self.assertTrue(a.sort_index().equals(b.sort_index()))

    def make_replay(self, folder: Path) -> Path:
        rows = []
        for fold in range(5):
            s = np.array([0.0, 2.0])
            r = np.array([1.0, -3.0])
            sz, rz, p = score_fold(self.meta[fold], self.stats, fold, s, r)
            for i in range(2):
                rows.append({
                    "novel_row_id": fold * 2 + i,
                    "fold": fold,
                    "assigned_fold": fold,
                    "svm_decision_score": s[i],
                    "roberta_logit_margin": r[i],
                    "svm_z": sz[i],
                    "roberta_z": rz[i],
                    "stacker_p_machine": p[i],
                    "stacker_pred": int(p[i] >= self.config["stacker_threshold"]),
                })
        path = folder / "novel.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_development_replay_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_replay(Path(folder))
            report = replay_novel(path, self.config, self.stats, self.meta)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["rows"], 10)

    def test_development_replay_detects_wrong_feature_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_replay(Path(folder))
            table = pd.read_csv(path)
            table.loc[0, "svm_z"] += 0.3
            table.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "replay mismatch"):
                replay_novel(path, self.config, self.stats, self.meta)

    def test_mocked_end_to_end_preserves_article_order(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            raw = folder / "raw.json"
            raw.write_text(json.dumps([
                {"id": "repeat", "domain": "news", "sent": ["Alpha.", "Beta."]},
                {"id": "repeat", "domain": "news", "sent": ["Gamma."]},
            ]), encoding="utf-8")
            svm_dir, rob_dir, output = folder / "svm", folder / "roberta", folder / "out"
            svm_dir.mkdir()
            rob_dir.mkdir()
            for k in range(5):
                (svm_dir / f"svm_fold_{k}.joblib").write_bytes(b"fake")
                ckpt = rob_dir / f"roberta_fold_{k}"
                ckpt.mkdir()
                for name in ("config.json", "tokenizer.json", "model.safetensors"):
                    (ckpt / name).write_bytes(b"fake")
                (ckpt / "oof_metadata.json").write_text(
                    json.dumps({"fold": k, "max_length": 256}), encoding="utf-8")
            class FakePipeline:
                def decision_function(self, texts):
                    return np.array([1.0] * len(texts))
            fake_bundle = lambda path: {
                "fold": int(Path(path).stem.rsplit("_", 1)[-1]),
                "pipeline": FakePipeline(),
            }
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            with patch.dict(sys.modules, {"torch": fake_torch}), \
                 patch("scripts.predict_oof_stacking.joblib.load", side_effect=fake_bundle), \
                 patch("scripts.predict_oof_stacking.infer_roberta_margins",
                       side_effect=lambda texts, *_: np.array([1.0] * len(texts))):
                report = infer_test(raw, output, svm_dir, rob_dir,
                                    self.config, self.stats, self.meta,
                                    batch_size=8, device_arg="auto",
                                    expected_articles=2, expected_sentences=3)
            self.assertEqual(report["status"], "COMPLETED_AWAITING_SCHEMA_VALIDATION")
            self.assertEqual(report["sentences"], 3)
            validation = validate(raw, output / "test_predictions.json",
                                  output / "test_predictions.csv", 2, 3,
                                  threshold=self.config["stacker_threshold"])
            self.assertEqual(validation["status"], "PASS")
            self.assertTrue(validation["threshold_checked"])

    def test_development_replay_detects_missing_fold(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_replay(Path(folder))
            frame = pd.read_csv(path)
            frame = frame.loc[frame["fold"] != 4]
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "no replay samples for fold 4"):
                replay_novel(path, self.config, self.stats, self.meta)


if __name__ == "__main__":
    unittest.main()
