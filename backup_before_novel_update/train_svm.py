"""Train a TF-IDF word/character n-gram LinearSVC baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from scipy.special import expit

try:
    from scripts.common import cfg_path, load_config, check_group_leakage
    from scripts.evaluate import evaluate_validation
except ImportError:
    from common import cfg_path, load_config, check_group_leakage
    from evaluate import evaluate_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Train word/character TF-IDF + LinearSVC.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    splits_dir = cfg_path(cfg, "splits_dir", "data/splits")
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "validation.csv")
    check_group_leakage(train_df, val_df)

    svm_cfg = cfg.get("svm", {})
    word_ngram = tuple(svm_cfg.get("word_ngram_range", [1, 2]))
    char_ngram = tuple(svm_cfg.get("char_ngram_range", [2, 5]))
    min_df = int(svm_cfg.get("min_df", 2))
    max_features = svm_cfg.get("max_features", 250000)
    if max_features is not None:
        max_features = int(max_features)

    features = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=word_ngram, min_df=min_df,
            max_features=max_features, sublinear_tf=True, strip_accents="unicode",
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char", ngram_range=char_ngram, min_df=min_df,
            max_features=max_features, sublinear_tf=True, strip_accents="unicode",
        )),
    ])
    # LinearSVC scales better than kernel SVC for a large sparse n-gram matrix.
    # Its decision margin is mapped through sigmoid for threshold search; this is a score,
    # not a calibrated probability (the validation-set threshold is selected empirically).
    classifier = LinearSVC(
        C=float(svm_cfg.get("C", 2.0)),
        max_iter=int(svm_cfg.get("max_iter", 3000)),
        class_weight="balanced",
        random_state=int(cfg.get("project", {}).get("random_seed", 42)),
    )
    pipeline = Pipeline([("features", features), ("classifier", classifier)])
    print(f"训练句子数：{len(train_df):,}；验证句子数：{len(val_df):,}")
    pipeline.fit(train_df["text"].astype(str), train_df["label_id"].astype(int))
    probabilities = expit(pipeline.decision_function(val_df["text"].astype(str)))

    threshold_cfg = cfg.get("threshold", {})
    results_dir = cfg_path(cfg, "results_dir", "results")
    threshold, metrics, _ = evaluate_validation(
        val_df, probabilities, "svm_tfidf", results_dir,
        threshold_min=float(threshold_cfg.get("min", 0.30)),
        threshold_max=float(threshold_cfg.get("max", 0.80)),
        threshold_step=float(threshold_cfg.get("step", 0.01)),
    )
    model_dir = cfg_path(cfg, "best_model_dir", "models/best_model")
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "threshold": threshold,
        "score_transform": "sigmoid(decision_function); relative score, not calibrated probability",
        "model_type": "svm",
        "label_to_id": {"human": 0, "machine": 1},
    }
    model_path = model_dir / "svm.joblib"
    joblib.dump(bundle, model_path, compress=3)
    print(f"模型已保存：{model_path}")
    print(f"验证集最优阈值：{threshold:.4f}")
    print(pd.Series(metrics).round(4).to_string())


if __name__ == "__main__":
    main()
