#!/usr/bin/env python3
"""Read-only inventory for the AIGTD Stage-1 baseline and inference handoff.

Reports presence/absence and Git state only. Never prints raw data, article IDs,
prediction rows or model contents; never deletes/retrains any artifact.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def git_value(*args: str) -> str | None:
    process = subprocess.run(["git", *args], cwd=ROOT, text=True,
                             capture_output=True, check=False)
    return process.stdout.strip() if process.returncode == 0 else None


def present(path: str) -> bool:
    return (ROOT / path).is_file()


def has_weights(directory: str) -> bool:
    path = ROOT / directory
    return (path.is_dir() and
            any(path.glob(pattern) for pattern in
                ("*.safetensors", "pytorch_model.bin", "*.bin")))


def build_inventory() -> dict:
    required = {
        "config": present("configs/experiment.yaml"),
        "inference_entry": present("scripts/predict.py"),
        "raw_test_local": present("data/raw/AIGTD_test_without_labels.json"),
        "fusion_config": present("results/oof_fusion/fusion_config.json"),
        "fusion_meta_models": present("results/oof_fusion/fusion_models.joblib"),
        "fusion_score_normalization": present("results/oof_fusion/score_normalization.json"),
        "novel_fusion_predictions_local": present("results/oof_fusion/novel_fusion_predictions.csv"),
        "corrected_stacking_metrics_local": present("results/stacking_error_analysis_corrected/metrics_all_models.csv"),
        "corrected_stacking_bootstrap_local": present("results/stacking_error_analysis_corrected/cluster_bootstrap_differences.csv"),
        "baseline_aggregate_reference": present("results/analysis_review/baseline_comparison_reference.csv"),
        "svm_inference_bundle": present("models/best_model/svm.joblib"),
        "roberta_single_inference_weights": has_weights("models/best_model/bert"),
        "roberta_oof_fold_bundles": all(
            has_weights(f"models/oof/roberta/roberta_fold_{i}") for i in range(5)
        ),
        "svm_oof_fold_bundles": all(
            present(f"models/oof/svm/svm_fold_{i}.joblib") for i in range(5)
        ),
        "context_oof_fold_bundles": all(
            has_weights(f"models/oof/context/context_fold_{i}") for i in range(5)
        ),
    }
    status = git_value("status", "--porcelain")
    return {
        "git_branch": git_value("branch", "--show-current"),
        "git_head": git_value("rev-parse", "--short", "HEAD"),
        "git_worktree_has_changes": None if status is None else bool(status),
        "artifact_presence": required,
        "all_checked_artifacts_present": all(required.values()),
        "stacking_end_to_end_test_inference_implemented": False,
        "stacking_inference_note": "Current scripts/predict.py does not implement the OOF Stacking model; do not select it for test inference yet.",
        "note": "GitHub ignores large weights and local data. Presence on GitHub does not prove absence on the user's server.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-out", type=Path, default=Path("results/stage1_local_audit/inventory.json"))
    args = parser.parse_args()
    report = build_inventory()
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
