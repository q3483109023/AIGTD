#!/usr/bin/env python3
"""Unlabeled AIGTD test inference with the existing fold-matched OOF Stacking bundle.

No retraining, threshold selection or ensemble feature averaging. This mirrors
the fold-matched Novel Validation feature construction: one base model pair,
its OOF-fold z-score normalization and its matching cross-fitted meta-model
per source group. Runs locally only; generated sentence/ID outputs must stay
private. Use --replay-novel first to verify the serialized meta pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from scripts.common import parse_test_json
    from scripts.train_oof_fusion import assign_source_groups_balanced
except ImportError:
    from common import parse_test_json
    from train_oof_fusion import assign_source_groups_balanced


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_artifacts(fusion_dir: Path) -> tuple[dict, dict, dict]:
    with (fusion_dir / "fusion_config.json").open(encoding="utf-8") as f:
        config = json.load(f)
    with (fusion_dir / "score_normalization.json").open(encoding="utf-8") as f:
        stats = json.load(f)
    bundle = joblib.load(fusion_dir / "fusion_models.joblib")
    require(isinstance(bundle, dict) and "meta_models_by_heldout_fold" in bundle,
            "fusion_models.joblib is not an OOF Stacking bundle")
    require(int(config["n_splits"]) == 5, "This frozen Stacking bundle expects 5 folds")
    require(isinstance(bundle.get("config"), dict), "Fusion bundle missing saved config")
    require(math.isclose(float(config["stacker_threshold"]),
                         float(bundle["config"]["stacker_threshold"]),
                         abs_tol=1e-12), "threshold mismatch between fusion config and bundle")
    require(bundle.get("score_stats") == stats,
            "fold standardization differs between saved joblib and JSON")
    models = {int(k): m for k, m in bundle["meta_models_by_heldout_fold"].items()}
    require(set(models) == set(range(5)), "missing or unexpected meta fold model")
    for fold, model in models.items():
        require(list(model.classes_) == [0, 1], f"meta fold {fold} class order is not [0,1]")
        require(int(model.n_features_in_) == 2, f"meta fold {fold} feature count != 2")
        for name in ("svm", "roberta"):
            entry = stats[name][str(fold)]
            mean, std = float(entry["mean"]), float(entry["std"])
            require(math.isfinite(mean) and math.isfinite(std) and std > 1e-8,
                    f"invalid {name} fold {fold} normalization")
    threshold = float(config["stacker_threshold"])
    require(0 < threshold < 1, "stacker threshold out of bounds")
    return config, stats, models


def score_fold(model, stats: dict, fold: int, svm_margin: np.ndarray,
               roberta_margin: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    svm_margin = np.asarray(svm_margin, dtype=float)
    roberta_margin = np.asarray(roberta_margin, dtype=float)
    require(svm_margin.shape == roberta_margin.shape and svm_margin.ndim == 1,
            "incompatible base-model score vectors")
    require(np.isfinite(svm_margin).all() and np.isfinite(roberta_margin).all(),
            "nonfinite raw SVM/RoBERTa margins")
    s = stats["svm"][str(fold)]
    r = stats["roberta"][str(fold)]
    svm_z = (svm_margin - float(s["mean"])) / float(s["std"])
    roberta_z = (roberta_margin - float(r["mean"])) / float(r["std"])
    x = np.column_stack([svm_z, roberta_z])
    require(np.isfinite(x).all(), "nonfinite standardized meta features")
    p = model.predict_proba(x)[:, 1]
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(),
            "nonfinite/out-of-range meta predictions")
    return svm_z, roberta_z, p


def replay_novel(path: Path, config: dict, stats: dict, models: dict,
                 atol: float = 5e-7) -> dict:
    """Recompute saved fold-matched Novel Stacking results from raw margins.

    This tests meta-model/normalizer/feature-order reproducibility, not the
    SVM/RoBERTa checkpoint forward passes. No row-level data are printed.
    """
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"novel_row_id", "fold", "assigned_fold", "svm_decision_score",
                "roberta_logit_margin", "svm_z", "roberta_z",
                "stacker_p_machine", "stacker_pred"}
    require(required.issubset(frame.columns),
            f"replay missing columns: {sorted(required - set(frame.columns))}")
    require(frame["novel_row_id"].is_unique and not frame.empty,
            "replay row IDs must be nonempty and unique")
    require((frame["fold"].astype(int) == frame["assigned_fold"].astype(int)).all(),
            "replay fold does not equal assigned fold")
    predicted = np.full(len(frame), np.nan, dtype=float)
    max_s_z, max_r_z = 0.0, 0.0
    for fold in range(5):
        mask = frame["fold"].astype(int).eq(fold).to_numpy()
        require(mask.any(), f"no replay samples for fold {fold}")
        s_z, r_z, p = score_fold(models[fold], stats, fold,
                                frame.loc[mask, "svm_decision_score"].to_numpy(float),
                                frame.loc[mask, "roberta_logit_margin"].to_numpy(float))
        predicted[mask] = p
        max_s_z = max(max_s_z, float(np.max(np.abs(s_z - frame.loc[mask, "svm_z"].to_numpy(float)))))
        max_r_z = max(max_r_z, float(np.max(np.abs(r_z - frame.loc[mask, "roberta_z"].to_numpy(float)))))
    max_p = float(np.max(np.abs(predicted - frame["stacker_p_machine"].to_numpy(float))))
    labels_match = np.array_equal(
        (predicted >= float(config["stacker_threshold"])).astype(int),
        frame["stacker_pred"].to_numpy(int),
    )
    require(np.isfinite(predicted).all(), "replay contained unset predictions")
    require(max(max_s_z, max_r_z, max_p) <= atol and labels_match,
            "replay mismatch: feature order / z-score / meta checkpoint / threshold")
    return {
        "status": "PASS", "evaluation": "development_replay_not_blind_test",
        "rows": int(len(frame)), "folds": 5,
        "max_svm_z_abs_delta": max_s_z,
        "max_roberta_z_abs_delta": max_r_z,
        "max_stacking_probability_abs_delta": max_p,
        "threshold_labels_match": bool(labels_match),
        "note": "This verifies saved meta predictions, not neural forward pass or new-test generalization.",
    }


def infer_roberta_margins(texts: list[str], checkpoint: Path,
                          batch_size: int, device: str, max_length: int) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True)
    require(int(model.config.num_labels) == 2, "RoBERTa checkpoint must have two logits")
    model.to(device).eval()
    result: list[np.ndarray] = []
    def collate(batch: list[str]):
        return tokenizer(batch, padding=True, truncation=True,
                         max_length=max_length, return_tensors="pt")
    loader = DataLoader(texts, batch_size=batch_size, shuffle=False, collate_fn=collate)
    with torch.inference_mode():
        for batch in loader:
            logits = model(**{key: val.to(device) for key, val in batch.items()}).logits
            result.append((logits[:, 1] - logits[:, 0]).float().cpu().numpy())
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    values = np.concatenate(result) if result else np.empty(0, dtype=float)
    require(values.size == len(texts) and np.isfinite(values).all(),
            "incomplete/nonfinite RoBERTa fold scores")
    return values


def infer_test(raw_test: Path, output_dir: Path, svm_dir: Path, roberta_dir: Path,
               config: dict, stats: dict, models: dict, batch_size: int,
               device_arg: str, expected_articles: int | None,
               expected_sentences: int | None) -> dict:
    import torch

    frame, records = parse_test_json(raw_test)
    require(not frame.empty, "no test sentences")
    require(batch_size > 0, "--batch-size must be positive")
    if expected_articles is not None:
        require(len(records) == expected_articles, "test article-record count differs from expected")
    if expected_sentences is not None:
        require(len(frame) == expected_sentences, "test sentence count differs from expected")
    require(frame["article_key"].nunique() == len(records), "article record key mismatch")
    require(frame[["article_key", "sentence_index"]].apply(tuple, axis=1).is_unique,
            "duplicate article-record + sentence position")
    require(frame["article_id"].notna().all(), "missing source ID")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device_arg == "auto" else device_arg
    require(not device.startswith("cuda") or torch.cuda.is_available(),
            "CUDA was requested but unavailable")
    # Exactly the source-group LPT assignment rule used for Novel Validation.
    # It depends on the full target set: dry-run assignments WILL differ from
    # those generated for a larger test set, but neither uses labels.
    assigned = assign_source_groups_balanced(frame["article_id"], 5).to_numpy(int)
    require(set(assigned).issubset(set(range(5))), "invalid assigned fold")
    require((pd.DataFrame({"source": frame["article_id"].astype(str),
                            "fold": assigned}).groupby("source")["fold"].nunique() == 1).all(),
            "source group assigned to multiple folds")
    output = np.full(len(frame), np.nan, dtype=float)

    # Preflight ALL weights before performing expensive inference.
    for fold in range(5):
        svm_path = svm_dir / f"svm_fold_{fold}.joblib"
        rob_path = roberta_dir / f"roberta_fold_{fold}"
        require(svm_path.is_file(), f"missing SVM fold model {fold}")
        require((rob_path / "config.json").is_file(), f"missing RoBERTa fold config {fold}")
        require((rob_path / "tokenizer.json").is_file(), f"missing RoBERTa fold tokenizer {fold}")
        require(any((rob_path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")),
                f"missing RoBERTa fold weights {fold}")
    counts = {}
    for fold in range(5):
        indices = np.flatnonzero(assigned == fold)
        counts[str(fold)] = int(len(indices))
        if len(indices) == 0:
            continue
        texts = frame.iloc[indices]["text"].astype(str).tolist()
        bundle = joblib.load(svm_dir / f"svm_fold_{fold}.joblib")
        require(int(bundle.get("fold", -1)) == fold, f"SVM bundle fold mismatch {fold}")
        svm_margin = np.asarray(bundle["pipeline"].decision_function(texts), dtype=float)
        del bundle
        rob_path = roberta_dir / f"roberta_fold_{fold}"
        metadata_file = rob_path / "oof_metadata.json"
        require(metadata_file.is_file(), f"missing RoBERTa OOF metadata fold {fold}")
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        require(int(metadata["fold"]) == fold, f"RoBERTa fold metadata mismatch {fold}")
        require(int(metadata["max_length"]) == 256,
                f"unexpected RoBERTa max_length fold {fold}; review frozen training protocol")
        roberta_margin = infer_roberta_margins(texts, rob_path, batch_size,
                                               device, int(metadata["max_length"]))
        _, _, p = score_fold(models[fold], stats, fold, svm_margin, roberta_margin)
        output[indices] = p
        print(f"Fold {fold}: {len(indices)} sentences scored; source groups kept together")

    require(np.isfinite(output).all(), "test inference left unscored sentences")
    threshold = float(config["stacker_threshold"])
    labels = np.where(output >= threshold, "machine", "human")
    result = frame.copy().reset_index(drop=True)
    result["assigned_fold"] = assigned
    result["p_machine"] = output
    result["predicted_id"] = (output >= threshold).astype(int)
    result["predicted_label"] = labels

    # Validate all group offsets before any output write. Source IDs may repeat.
    articles = []
    for record_idx, record in enumerate(records):
        article_key = f"record_{record_idx:08d}"
        section = result.loc[result["article_key"].eq(article_key)].sort_values(
            "sentence_index", kind="stable")
        original = record["num_sentences"]
        require(len(section) == original, f"article record {record_idx} has mismatched sentence count")
        require(section["sentence_index"].astype(int).tolist() == list(range(original)),
                f"article record {record_idx} has noncontiguous sentence order")
        articles.append({
            "id": str(record["article_id"]),
            "domain": str(record["domain"]),
            "sent": section["text"].astype(str).tolist(),
            "predicted_labels": section["predicted_label"].astype(str).tolist(),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    (output_dir / "test_predictions.json").write_text(
        json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "status": "COMPLETED_AWAITING_SCHEMA_VALIDATION",
        "protocol": "oof_stacking_fold_matched_test_inference",
        "articles": len(records), "sentences": len(frame),
        "human_predictions": int((labels == "human").sum()),
        "machine_predictions": int((labels == "machine").sum()),
        "stacker_threshold": threshold,
        "fold_sentence_counts": counts,
        "device": device,
        "note": "Unlabeled test: no Accuracy/F1. Generic JSON schema pending teacher submission format.",
    }
    (output_dir / "inference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-dir", type=Path, default=Path("results/oof_fusion"))
    parser.add_argument("--replay-novel", type=Path, default=None,
                        help="Verify metadata, saved z scores and fold-matched meta predictions without running base models")
    parser.add_argument("--test", type=Path, default=None,
                        help="Unlabeled test JSON; required unless --replay-novel")
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage1_local_audit/stacking_test"))
    parser.add_argument("--svm-dir", type=Path, default=Path("models/oof/svm"))
    parser.add_argument("--roberta-dir", type=Path, default=Path("models/oof/roberta"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--expected-articles", type=int, default=None)
    parser.add_argument("--expected-sentences", type=int, default=None)
    args = parser.parse_args()
    require((args.replay_novel is None) != (args.test is None),
            "Specify exactly one of --replay-novel or --test")
    config, stats, models = load_artifacts(args.fusion_dir)
    if args.replay_novel is not None:
        summary = replay_novel(args.replay_novel, config, stats, models)
    else:
        summary = infer_test(args.test, args.output_dir, args.svm_dir, args.roberta_dir,
                             config, stats, models, args.batch_size,
                             args.device, args.expected_articles, args.expected_sentences)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
