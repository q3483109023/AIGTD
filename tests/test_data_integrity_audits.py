from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.analyze_conflict_article_distribution import analyze as analyze_conflicts
from scripts.audit_context_oof_independently import (
    build_windows,
    main as run_context_audit,
    union_components,
)


class ConflictDistributionTests(unittest.TestCase):
    def test_conflict_categories_and_safe_aggregation(self):
        train = pd.DataFrame([
            {"article_key": "tr-a", "article_id": 1, "text": "Shared X", "label_id": 0},
            {"article_key": "tr-b", "article_id": 2, "text": "Shared Y", "label_id": 0},
            {"article_key": "tr-c", "article_id": 3, "text": "Shared Y", "label_id": 1},
            {"article_key": "tr-d", "article_id": 4, "text": "Shared Z", "label_id": 1},
        ])
        validation = pd.DataFrame([
            {"article_key": "va-a", "article_id": 5, "text": "shared x", "label_id": 1},
            {"article_key": "va-b", "article_id": 6, "text": "shared y", "label_id": 0},
            {"article_key": "va-c", "article_id": 7, "text": "shared z", "label_id": 0},
            {"article_key": "va-d", "article_id": 8, "text": "shared z", "label_id": 1},
        ])

        summary, details = analyze_conflicts(train, validation)

        self.assertEqual(summary["cross_split_conflicting_text_groups"], 3)
        self.assertEqual(summary["group_label_pattern_counts"], {
            "both_splits_consistent_but_disagree": 1,
            "train_internal_conflict_only": 1,
            "validation_internal_conflict_only": 1,
        })
        self.assertEqual(summary["distinct_article_keys_involved"], 8)
        self.assertEqual(summary["distinct_article_ids_involved"], 8)
        self.assertNotIn("Shared X", json.dumps(summary))
        self.assertNotIn("tr-a", json.dumps(summary))
        self.assertEqual(set(details), {"article_key", "article_id"})
        for detail in details.values():
            self.assertNotIn("text", detail.columns)
            self.assertNotIn("_norm_text", detail.columns)


class ContextAuditTests(unittest.TestCase):
    @staticmethod
    def _article_rows(article_key: str, article_id: int, prefix: str, fold: int | None):
        rows = []
        for sentence_index, label_id in enumerate((0, 1)):
            row = {
                "article_key": article_key,
                "article_id": article_id,
                "sentence_index": sentence_index,
                "text": f"{prefix} sentence {sentence_index}",
                "label_id": label_id,
            }
            if fold is not None:
                row["fold"] = fold
                row["source_article_id"] = str(article_id)
            rows.append(row)
        return rows

    def test_independent_audit_passes_exact_materialized_folds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = pd.DataFrame(
                self._article_rows("a0", 10, "train-a0", 0)
                + self._article_rows("a1", 11, "train-a1", 0)
                + self._article_rows("b0", 20, "train-b0", 1)
                + self._article_rows("b1", 21, "train-b1", 1)
            )
            validation = pd.DataFrame(
                self._article_rows("v0", 100, "validation-v0", None)
                + self._article_rows("v1", 101, "validation-v1", None)
            )
            manifest = train[[
                "article_key", "article_id", "sentence_index", "fold"
            ]].copy()
            manifest = manifest.rename(columns={"article_id": "source_article_id"})
            manifest["row_id"] = range(len(manifest))
            manifest["source_article_id"] = manifest["source_article_id"].astype(str)

            folds_dir = root / "context_oof_folds"
            folds_dir.mkdir()
            train_ctx = train.copy()
            train_ctx["oof_row_id"] = range(len(train_ctx))
            train_ctx["source_article_id"] = manifest["source_article_id"].to_numpy()
            train_ctx["fold"] = manifest["fold"].to_numpy()
            train_ctx = build_windows(train_ctx, 1, "test train")
            val_ctx = build_windows(validation, 1, "test validation")
            novel = val_ctx.loc[
                ~val_ctx["_norm_target"].isin(set(train_ctx["_norm_target"]))
            ].copy()
            novel["novel_row_id"] = range(len(novel))
            novel.drop(columns=["_norm_target", "_window_components"]).to_csv(
                folds_dir / "novel_validation.csv", index=False
            )

            for fold in (0, 1):
                fold_val = train_ctx.loc[train_ctx["fold"].eq(fold)].copy()
                candidate = train_ctx.loc[~train_ctx["fold"].eq(fold)].copy()
                protected = union_components(fold_val) | union_components(novel)
                keep = ~candidate["_window_components"].map(
                    lambda parts: any(part in protected for part in parts)
                )
                fold_train = candidate.loc[keep].copy()
                folder = folds_dir / f"fold_{fold}"
                folder.mkdir()
                drop_cols = ["_norm_target", "_window_components"]
                fold_train.drop(columns=drop_cols).to_csv(folder / "train.csv", index=False)
                fold_val.drop(columns=drop_cols).to_csv(folder / "validation.csv", index=False)

            train_path = root / "train.csv"
            val_path = root / "validation.csv"
            manifest_path = root / "manifest.csv"
            train.drop(columns=["fold", "source_article_id"]).to_csv(train_path, index=False)
            validation.to_csv(val_path, index=False)
            manifest.to_csv(manifest_path, index=False)
            output_dir = root / "audit_output"

            argv = [
                "audit_context_oof_independently.py",
                "--train", str(train_path),
                "--validation", str(val_path),
                "--row-manifest", str(manifest_path),
                "--folds-dir", str(folds_dir),
                "--output-dir", str(output_dir),
                "--n-splits", "2",
                "--expected-novel-rows", str(len(novel)),
            ]
            with patch("sys.argv", argv):
                run_context_audit()

            report_path = output_dir / "context_oof_independent_integrity_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["overall_pass"])
            self.assertTrue(report["oof_validation_rows_cover_each_train_row_once"])
            self.assertTrue(report["novel_export_mapping_check"]["pass"])
            self.assertTrue(all(item["pass"] for item in report["fold_reports"]))
            self.assertNotIn("train-a0 sentence", report_path.read_text(encoding="utf-8"))

            # Tamper with one context string: the independent export comparison
            # must fail rather than silently accepting an incomplete/mismatched CSV.
            fold0_train = pd.read_csv(folds_dir / "fold_0" / "train.csv")
            fold0_train.loc[0, "context_text"] = "[PREVIOUS] altered [TARGET] wrong [NEXT] altered"
            fold0_train.to_csv(folds_dir / "fold_0" / "train.csv", index=False)
            with patch("sys.argv", argv):
                with self.assertRaises(SystemExit):
                    run_context_audit()


if __name__ == "__main__":
    unittest.main()
