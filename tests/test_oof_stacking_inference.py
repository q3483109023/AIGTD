"""Synthetic safety/regression tests for the existing OOF stacking bundle inference."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.predict_oof_stacking import replay_novel, score_fold
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
