"""Train Transformer semantic embeddings + statistical features fusion model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from features.semantic import extract_transformer_embeddings
    from features.statistical import extract_statistical_features
    from scripts.common import cfg_path, check_group_leakage, load_config, mark_novel_validation
    from scripts.evaluate import evaluate_validation
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from features.semantic import extract_transformer_embeddings
    from features.statistical import extract_statistical_features
    from scripts.common import cfg_path, check_group_leakage, load_config, mark_novel_validation
    from scripts.evaluate import evaluate_validation


def make_feature_dicts(df: pd.DataFrame, include_domain: bool = False) -> list[dict]:
    stats = extract_statistical_features(df["text"].astype(str).tolist())
    domains = (
        df["domain"].fillna("unknown").astype(str).tolist()
        if "domain" in df.columns else ["unknown"] * len(df)
    )
    records = []
    for row, domain in zip(stats.to_dict(orient="records"), domains):
        item = {key: float(value) for key, value in row.items()}
        if include_domain:
            item[f"domain={domain}"] = 1.0
        records.append(item)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Train semantic + statistical feature fusion.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fusion_cfg = cfg.get("fusion", {})
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

    embedding_model = fusion_cfg.get(
        "embedding_model_name", cfg.get("transformer", {}).get("model_name", "roberta-base")
    )
    max_length = int(fusion_cfg.get("max_length", 256))
    batch_size = int(fusion_cfg.get("batch_size", 32))
    device = cfg.get("transformer", {}).get("device", "auto")

    print("一次性提取训练集与验证集 Transformer 语义向量……")
    all_texts = train_df["text"].astype(str).tolist() + val_df["text"].astype(str).tolist()
    all_embeddings = extract_transformer_embeddings(
        all_texts, embedding_model, max_length, batch_size, device
    )
    train_embeddings = all_embeddings[:len(train_df)]
    val_embeddings = all_embeddings[len(train_df):]
    del all_embeddings

    dict_vectorizer = DictVectorizer(sparse=True)
    include_domain = bool(fusion_cfg.get("include_domain_feature", False))
    train_stats = dict_vectorizer.fit_transform(make_feature_dicts(train_df, include_domain))
    val_stats = dict_vectorizer.transform(make_feature_dicts(val_df, include_domain))
    x_train = hstack([csr_matrix(train_embeddings), train_stats], format="csr")
    x_val = hstack([csr_matrix(val_embeddings), val_stats], format="csr")
    scaler = StandardScaler(with_mean=False)
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    classifier_name = str(fusion_cfg.get("classifier", "logreg")).lower()
    seed = int(cfg.get("project", {}).get("random_seed", 42))
    if classifier_name == "logreg":
        classifier = LogisticRegression(
            C=float(fusion_cfg.get("C", 1.0)),
            max_iter=int(fusion_cfg.get("max_iter", 3000)),
            class_weight="balanced", solver="liblinear", random_state=seed,
        )
    elif classifier_name == "svm":
        classifier = SVC(
            C=float(fusion_cfg.get("C", 1.0)), kernel="linear", probability=True,
            class_weight="balanced", random_state=seed,
        )
    elif classifier_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("fusion.classifier=lightgbm 需要安装：pip install lightgbm") from exc
        classifier = LGBMClassifier(
            n_estimators=400, learning_rate=0.04, num_leaves=31,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
    else:
        raise ValueError("fusion.classifier 只允许 logreg、svm 或 lightgbm。")

    print(f"训练句子数：{len(train_df):,}")
    print(f"完整验证句子数：{len(val_df):,}")
    print(f"Novel Validation：{int(novel_mask.sum()):,} ({novel_mask.mean():.2%})")
    classifier.fit(x_train, train_df["label_id"].astype(int))
    if not hasattr(classifier, "predict_proba"):
        raise RuntimeError("当前融合分类器不支持 predict_proba，无法进行阈值优化。")
    probabilities = classifier.predict_proba(x_val)[:, 1]

    threshold_cfg = cfg.get("threshold", {})
    results_dir = cfg_path(cfg, "results_dir", "results")
    threshold, metrics, _ = evaluate_validation(
        frame=val_df,
        probabilities=probabilities,
        experiment="fusion",
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
        "dict_vectorizer": dict_vectorizer,
        "scaler": scaler,
        "classifier": classifier,
        "threshold": threshold,
        "threshold_selection_metric": "novel_macro_f1",
        "novel_validation_count": int(novel_mask.sum()),
        "full_validation_count": int(len(val_df)),
        "model_type": "fusion",
        "embedding_model_name": embedding_model,
        "max_length": max_length,
        "batch_size": batch_size,
        "classifier_name": classifier_name,
        "include_domain_feature": include_domain,
        "label_to_id": {"human": 0, "machine": 1},
    }
    model_path = model_dir / "fusion.joblib"
    joblib.dump(bundle, model_path, compress=3)
    metadata = {key: value for key, value in bundle.items() if key not in {"dict_vectorizer", "scaler", "classifier"}}
    (model_dir / "fusion_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n模型已保存：{model_path}")
    print(f"Novel Validation 最优阈值：{threshold:.4f}")
    print("以下未加前缀的五项指标代表 Novel Validation：")
    print(pd.Series({key: metrics[key] for key in ("accuracy", "precision", "recall", "macro_f1", "micro_f1")}).round(4))
    print("Full Validation Macro-F1（同一阈值）：{:.4f}".format(metrics["full_macro_f1"]))


if __name__ == "__main__":
    main()
