"""Train TF-IDF word/character n-gram LinearSVC and select threshold on Novel Validation."""
from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

try:
    from scripts.common import cfg_path, check_group_leakage, load_config, mark_novel_validation
    from scripts.evaluate import evaluate_validation
except ImportError:  # Support: python scripts/train_svm.py
    from common import cfg_path, check_group_leakage, load_config, mark_novel_validation
    from evaluate import evaluate_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Train word/character TF-IDF + LinearSVC.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits_dir = cfg_path(cfg, "splits_dir", "data/splits")
    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "validation.csv"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("缺少 train.csv 或 validation.csv，请先运行 python scripts/preprocess.py。")
    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    val_df = pd.read_csv(val_path, encoding="utf-8-sig")
    check_group_leakage(train_df, val_df)
    val_df = mark_novel_validation(val_df, train_df)

    novel_mask = val_df["is_novel_validation"].astype(bool).to_numpy()
    novel_y = val_df.loc[novel_mask, "label_id"].astype(int)
    if len(novel_y) == 0 or novel_y.nunique() < 2:
        raise ValueError("Novel Validation 为空或只包含一个类别，无法选择 Macro-F1 最优阈值。")

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
    classifier = LinearSVC(
        C=float(svm_cfg.get("C", 2.0)),
        max_iter=int(svm_cfg.get("max_iter", 3000)),
        class_weight="balanced",
        random_state=int(cfg.get("project", {}).get("random_seed", 42)),
    )
    pipeline = Pipeline([("features", features), ("classifier", classifier)])

    print(f"训练句子数：{len(train_df):,}")
    print(f"完整验证句子数：{len(val_df):,}")
    print(
        f"Novel Validation：{int(novel_mask.sum()):,} "
        f"({novel_mask.mean():.2%})；训练中已出现：{int((~novel_mask).sum()):,}"
    )
    print("训练 TF-IDF + LinearSVC……")
    pipeline.fit(train_df["text"].astype(str), train_df["label_id"].astype(int))

    # LinearSVC has no calibrated predict_proba. Sigmoid is used only as a
    # monotonic score transform; the threshold is selected empirically on Novel Validation.
    decision_scores = pipeline.decision_function(val_df["text"].astype(str))
    score_values = expit(decision_scores)

    threshold_cfg = cfg.get("threshold", {})
    results_dir = cfg_path(cfg, "results_dir", "results")
    threshold, metrics, _ = evaluate_validation(
        frame=val_df,
        probabilities=score_values,
        experiment="svm_tfidf",
        results_dir=results_dir,
        threshold_min=float(threshold_cfg.get("min", 0.30)),
        threshold_max=float(threshold_cfg.get("max", 0.80)),
        threshold_step=float(threshold_cfg.get("step", 0.01)),
        optimize_threshold=True,
        train_df=train_df,
    )

    model_dir = cfg_path(cfg, "best_model_dir", "models/best_model")
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "threshold": threshold,
        "threshold_selection_metric": "novel_macro_f1",
        "novel_validation_count": int(novel_mask.sum()),
        "full_validation_count": int(len(val_df)),
        "score_transform": "sigmoid(decision_function); relative score, not calibrated probability",
        "model_type": "svm",
        "label_to_id": {"human": 0, "machine": 1},
        "novel_text_normalization": "Unicode NFKC + casefold + whitespace normalization; punctuation retained",
    }
    model_path = model_dir / "svm.joblib"
    joblib.dump(bundle, model_path, compress=3)
    (model_dir / "svm_metadata.json").write_text(
        json.dumps({key: value for key, value in bundle.items() if key != "pipeline"},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n模型已保存：{model_path}")
    print(f"Novel Validation 最优阈值：{threshold:.4f}")
    print("以下未加前缀的五项指标代表 Novel Validation：")
    print(pd.Series({key: metrics[key] for key in ("accuracy", "precision", "recall", "macro_f1", "micro_f1")}).round(4))
    print("Full Validation Macro-F1（同一阈值）：{:.4f}".format(metrics["full_macro_f1"]))


if __name__ == "__main__":
    main()
