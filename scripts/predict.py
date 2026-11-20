"""Run sentence-level predictions on the unlabeled AIGTD test JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit
import pandas as pd
from scipy.sparse import csr_matrix, hstack

try:
    from scripts.common import build_context_text, cfg_path, load_config, parse_test_json, resolve_path
except ImportError:
    from common import build_context_text, cfg_path, load_config, parse_test_json, resolve_path


def make_feature_dicts(df: pd.DataFrame, include_domain: bool = False) -> list[dict]:
    try:
        from features.statistical import extract_statistical_features
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from features.statistical import extract_statistical_features
    stats = extract_statistical_features(df["text"].astype(str).tolist())
    domains = df["domain"].fillna("unknown").astype(str).tolist() if "domain" in df else ["unknown"] * len(df)
    records = []
    for values, domain in zip(stats.to_dict(orient="records"), domains):
        record = {k: float(v) for k, v in values.items()}
        if include_domain:
            record[f"domain={domain}"] = 1.0
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict human/machine labels for every test sentence.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model-type", choices=["svm", "bert", "context", "fusion"], default=None)
    parser.add_argument("--model-path", default=None, help="Optional model path override")
    parser.add_argument("--test", default=None, help="Optional override for raw test JSON path")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_type = args.model_type or cfg.get("prediction", {}).get("model_type", "bert")
    best_model_dir = cfg_path(cfg, "best_model_dir", "models/best_model")
    results_dir = resolve_path(args.output_dir) if args.output_dir else cfg_path(cfg, "results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    processed_test = cfg_path(cfg, "processed_dir", "data/processed") / "test_sentences.csv"
    if processed_test.exists() and args.test is None:
        test_df = pd.read_csv(processed_test)
    else:
        test_path = resolve_path(args.test) if args.test else cfg_path(cfg, "test_json", "data/raw/AIGTD_test_without_labels.json")
        test_df, _ = parse_test_json(test_path)
        processed_test.parent.mkdir(parents=True, exist_ok=True)
        test_df.to_csv(processed_test, index=False, encoding="utf-8-sig")
    if test_df.empty:
        raise ValueError("测试数据没有任何句子，无法预测。")

    if model_type == "svm":
        model_path = resolve_path(args.model_path) if args.model_path else best_model_dir / "svm.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"找不到 SVM 模型：{model_path}。先运行 scripts/train_svm.py。")
        bundle = joblib.load(model_path)
        pipeline = bundle["pipeline"]
        probabilities = expit(pipeline.decision_function(test_df.text.astype(str)))
        threshold = float(bundle.get("threshold", 0.5))

    elif model_type in ("bert", "context"):
        model_dir = resolve_path(args.model_path) if args.model_path else best_model_dir / model_type
        metadata_path = model_dir / "aigtd_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"找不到 Transformer 模型元数据：{metadata_path}。先运行相应训练脚本。")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model_type") != model_type:
            raise ValueError(f"模型目录类型为 {metadata.get('model_type')}，但指定了 {model_type}。")
        if model_type == "context":
            texts = build_context_text(test_df, int(metadata.get("context_window", 1)))
        else:
            texts = test_df.text.astype(str).tolist()
        try:
            from scripts.transformer_utils import predict_transformer_probabilities
        except ImportError:
            from transformer_utils import predict_transformer_probabilities
        device = cfg.get("transformer", {}).get("device", "auto")
        probabilities = predict_transformer_probabilities(
            texts, model_dir, batch_size=int(cfg.get("transformer", {}).get("eval_batch_size", 32)), device=device
        )
        threshold = float(metadata.get("threshold", 0.5))

    elif model_type == "fusion":
        model_path = resolve_path(args.model_path) if args.model_path else best_model_dir / "fusion.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"找不到融合模型：{model_path}。先运行 scripts/train_fusion.py。")
        bundle = joblib.load(model_path)
        try:
            from features.semantic import extract_transformer_embeddings
            from features.statistical import extract_statistical_features
        except ImportError:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from features.semantic import extract_transformer_embeddings
            from features.statistical import extract_statistical_features
        print("提取测试集语义向量……")
        device = cfg.get("transformer", {}).get("device", "auto")
        embeddings = extract_transformer_embeddings(
            test_df.text.astype(str).tolist(),
            bundle["embedding_model_name"],
            int(bundle["max_length"]),
            int(bundle["batch_size"]),
            device,
        )
        statistical = bundle["dict_vectorizer"].transform(
            make_feature_dicts(test_df, bool(bundle.get("include_domain_feature", False)))
        )
        features = hstack([csr_matrix(embeddings), statistical], format="csr")
        features = bundle["scaler"].transform(features)
        probabilities = bundle["classifier"].predict_proba(features)[:, 1]
        threshold = float(bundle.get("threshold", 0.5))
    else:
        raise ValueError(f"不支持的模型类型：{model_type}")

    if len(probabilities) != len(test_df):
        raise RuntimeError(f"预测数量 {len(probabilities)} 与测试句子数 {len(test_df)} 不一致。")
    predicted_ids = (np.asarray(probabilities) >= threshold).astype(int)
    output_df = test_df.copy().reset_index(drop=True)
    output_df["p_machine"] = np.asarray(probabilities, dtype=float)
    output_df["predicted_id"] = predicted_ids
    output_df["predicted_label"] = np.where(predicted_ids == 1, "machine", "human")
    output_df.to_csv(results_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    # Generic JSON output. The official competition submission schema was not present in the provided material;
    # this preserves article IDs and sentence ordering so it can be converted easily if another schema is specified.
    articles = []
    output_group_col = "article_key" if "article_key" in output_df.columns else "article_id"
    for _, group in output_df.groupby(output_group_col, sort=False):
        group = group.sort_values("sentence_index", kind="stable")
        articles.append({
            "id": str(group.article_id.iloc[0]),
            "domain": str(group.domain.iloc[0]) if "domain" in group else "unknown",
            "sent": group.text.astype(str).tolist(),
            "predicted_labels": group.predicted_label.astype(str).tolist(),
        })
    (results_dir / "test_predictions.json").write_text(
        json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("测试集预测完成")
    print(f"模型：{model_type}；阈值：{threshold:.4f}")
    print(f"文章记录数：{output_df[output_group_col].nunique():,}；句子数：{len(output_df):,}")
    print(f"Human 预测：{int((predicted_ids == 0).sum()):,}；Machine 预测：{int((predicted_ids == 1).sum()):,}")
    print(f"输出：{results_dir / 'test_predictions.csv'}")
    print(f"输出：{results_dir / 'test_predictions.json'}")


if __name__ == "__main__":
    main()
