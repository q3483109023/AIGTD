#!/usr/bin/env python3
"""Prepare strict source-grouped OOF data for context-window RoBERTa.

A candidate training window is purged if any normalized sentence component
(previous, target, or next) occurs in an outer validation window or a Novel
Validation target window. Windows are built on complete article records first.
"""
from __future__ import annotations
import argparse, json, re, unicodedata
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


def norm(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def add_windows(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    required = {"article_key", "sentence_index", "text"}
    if required - set(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if window < 1:
        raise ValueError("--context-window must be >= 1")
    out = frame.copy()
    out["_norm_target"] = out["text"].map(norm)
    if (out["_norm_target"] == "").any():
        raise ValueError("Empty normalized sentence found.")
    if out.duplicated(["article_key", "sentence_index"]).any():
        raise ValueError("Duplicate (article_key, sentence_index) rows.")
    lookup = {}
    for key, group in out.groupby("article_key", sort=False):
        group = group.sort_values("sentence_index", kind="stable")
        rows = list(group[["sentence_index", "text", "_norm_target"]].itertuples(index=False, name=None))
        by_index = {int(row[0]): row for row in rows}
        for idx, target, _ in rows:
            left = [by_index[j] for j in range(int(idx)-window, int(idx)) if j in by_index]
            right = [by_index[j] for j in range(int(idx)+1, int(idx)+window+1) if j in by_index]
            left_text = " ".join(str(x[1]) for x in left) if left else "[START]"
            right_text = " ".join(str(x[1]) for x in right) if right else "[END]"
            components = tuple(x[2] for x in left + [by_index[int(idx)]] + right if x[2])
            text = f"[PREVIOUS] {left_text} [TARGET] {target} [NEXT] {right_text}"
            lookup[(str(key), int(idx))] = (text, components)
    keys = [(str(r.article_key), int(r.sentence_index)) for r in out.itertuples()]
    out["context_text"] = [lookup[k][0] for k in keys]
    out["_window_components"] = [lookup[k][1] for k in keys]
    return out


def component_union(frame: pd.DataFrame) -> set[str]:
    result: set[str] = set()
    for values in frame["_window_components"]:
        result.update(values)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="data/splits/train.csv")
    p.add_argument("--validation", default="data/splits/validation.csv")
    p.add_argument("--row-manifest", default="results/oof_purged_group_audit/row_fold_manifest.csv")
    p.add_argument("--output-dir", default="data/context_oof_folds")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--context-window", type=int, default=1)
    a = p.parse_args()
    train_path, val_path, manifest_path = map(Path, (a.train, a.validation, a.row_manifest))
    for path in (train_path, val_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if a.n_splits < 2:
        raise ValueError("--n-splits must be >= 2")
    train = pd.read_csv(train_path, encoding="utf-8-sig").reset_index(drop=True)
    val = pd.read_csv(val_path, encoding="utf-8-sig").reset_index(drop=True)
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    required = {"row_id", "article_key", "source_article_id", "fold", "sentence_index"}
    if required - set(manifest.columns):
        raise ValueError(f"row manifest missing {sorted(required - set(manifest.columns))}")
    if len(manifest) != len(train):
        raise ValueError("Manifest/train row count mismatch.")
    manifest = manifest.sort_values("row_id", kind="stable").reset_index(drop=True)
    if not np.array_equal(manifest["row_id"].astype(int).to_numpy(), np.arange(len(train))):
        raise ValueError("Manifest row_id must cover 0..N-1 exactly.")
    for col in ("article_key", "sentence_index"):
        if not (manifest[col].astype(str).to_numpy() == train[col].astype(str).to_numpy()).all():
            raise ValueError(f"Manifest {col} order does not match train.csv.")
    source_col = "article_id" if "article_id" in train else "source_article_id"
    if source_col not in train:
        raise ValueError("train.csv requires article_id or source_article_id.")
    if not (manifest["source_article_id"].astype(str).to_numpy() == train[source_col].astype(str).to_numpy()).all():
        raise ValueError("Manifest source IDs do not match train.csv.")
    folds = sorted(manifest["fold"].astype(int).unique().tolist())
    if folds != list(range(a.n_splits)):
        raise ValueError(f"Expected folds 0..{a.n_splits-1}, got {folds}")

    train["oof_row_id"] = np.arange(len(train), dtype=int)
    train["source_article_id"] = manifest["source_article_id"].astype(str).to_numpy()
    train["fold"] = manifest["fold"].astype(int).to_numpy()
    train_ctx = add_windows(train, a.context_window)
    val_ctx = add_windows(val, a.context_window)
    train_norms = set(train_ctx["_norm_target"])
    val_ctx["is_novel_validation"] = val_ctx["_norm_target"].map(lambda x: bool(x) and x not in train_norms)
    novel = val_ctx.loc[val_ctx["is_novel_validation"]].copy()
    if novel.empty:
        raise ValueError("Novel Validation is empty.")
    novel_components = component_union(novel)

    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries, assigned = [], []
    for fold in folds:
        fold_val = train_ctx.loc[train_ctx["fold"].eq(fold)].copy()
        candidate = train_ctx.loc[~train_ctx["fold"].eq(fold)].copy()
        val_components = component_union(fold_val)
        protected = val_components | novel_components
        purge_mask = candidate["_window_components"].map(lambda xs: any(x in protected for x in xs))
        purged = candidate.loc[purge_mask]
        fold_train = candidate.loc[~purge_mask].copy()
        source_overlap = set(fold_train["source_article_id"].astype(str)) & set(fold_val["source_article_id"].astype(str))
        article_overlap = set(fold_train["article_key"].astype(str)) & set(fold_val["article_key"].astype(str))
        train_components = component_union(fold_train)
        val_overlap, novel_overlap = train_components & val_components, train_components & novel_components
        if source_overlap or article_overlap or val_overlap or novel_overlap:
            raise RuntimeError(f"Fold {fold} leakage: source={len(source_overlap)}, article={len(article_overlap)}, val_components={len(val_overlap)}, novel_components={len(novel_overlap)}")
        if fold_train["label_id"].nunique() != 2:
            raise ValueError(f"Fold {fold} has fewer than two classes after strict window purge.")
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        drop = ["_norm_target", "_window_components"]
        fold_train.drop(columns=drop).to_csv(fold_dir / "train.csv", index=False, encoding="utf-8-sig")
        fold_val.drop(columns=drop).to_csv(fold_dir / "validation.csv", index=False, encoding="utf-8-sig")
        assigned.extend(fold_val["oof_row_id"].astype(int).tolist())
        summaries.append({
            "fold": fold, "candidate_train_rows": len(candidate), "purged_train_rows": len(purged),
            "purged_train_pct": len(purged) / max(len(candidate), 1), "final_train_rows": len(fold_train),
            "final_train_human": int((fold_train["label_id"].astype(int) == 0).sum()),
            "final_train_machine": int((fold_train["label_id"].astype(int) == 1).sum()),
            "validation_rows": len(fold_val), "validation_component_count": len(val_components),
            "novel_component_count": len(novel_components), "post_purge_validation_component_overlap": len(val_overlap),
            "post_purge_novel_component_overlap": len(novel_overlap), "source_id_overlap": len(source_overlap),
            "article_key_overlap": len(article_overlap),
        })
        print(f"Fold {fold}: candidate={len(candidate):,}; purged={len(purged):,} ({len(purged)/max(len(candidate),1):.2%}); train={len(fold_train):,}; val={len(fold_val):,}; component overlaps=0")
    if len(assigned) != len(train) or len(set(assigned)) != len(train):
        raise RuntimeError("OOF validation rows do not cover every train row exactly once.")
    novel.drop(columns=["_norm_target", "_window_components"]).assign(
        novel_row_id=np.arange(len(novel), dtype=int)
    ).to_csv(out_dir / "novel_validation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(out_dir / "fold_purge_balance.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "n_splits": a.n_splits, "context_window": a.context_window, "train_rows": len(train),
        "validation_rows": len(val), "novel_validation_rows": len(novel),
        "fold_strategy": "source-ID grouped; purge a candidate training window if any normalized sentence component intersects outer-validation or Novel-Validation target windows",
        "normalization": "Unicode NFKC + casefold + collapsed whitespace; punctuation retained",
        "context_format": "[PREVIOUS] ... [TARGET] ... [NEXT] ...",
        "checks": {
            "every_train_row_in_exactly_one_oof_validation_fold": True,
            "source_ids_disjoint_per_fold": True, "article_keys_disjoint_per_fold": True,
            "training_window_components_disjoint_from_outer_validation_windows": True,
            "training_window_components_disjoint_from_novel_validation_windows": True,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nContext OOF preparation complete. Novel targets={len(novel):,}; output={out_dir.resolve()}")


if __name__ == "__main__":
    main()
