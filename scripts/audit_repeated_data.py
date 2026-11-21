#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit repeated sentences and article-ID relationships across train/validation.

This is a descriptive audit. It preserves conflicting-label duplicates and
never edits the source CSVs. Text normalization matches the OOF preparation
protocol: Unicode NFKC + casefold + collapsed whitespace; punctuation retained.
"""
from __future__ import annotations
import argparse, json, re, unicodedata, hashlib
from pathlib import Path
import pandas as pd


def norm(x):
    if pd.isna(x): return ""
    return re.sub(r"\s+"," ",unicodedata.normalize("NFKC",str(x)).casefold()).strip()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train",default="data/splits/train.csv")
    ap.add_argument("--validation",default="data/splits/validation.csv")
    ap.add_argument("--manifest",default="results/oof_purged_group_audit/row_fold_manifest.csv")
    ap.add_argument("--output-dir",default="results/duplicate_data_audit")
    a=ap.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(a.train,encoding="utf-8-sig",low_memory=False).copy()
    val=pd.read_csv(a.validation,encoding="utf-8-sig",low_memory=False).copy()
    for name,d in (("train",train),("validation",val)):
        for col in ("text","article_key","sentence_index"):
            if col not in d: raise ValueError(f"{name} missing required column {col}")
        d["split"]=name; d["norm_text"]=d.text.map(norm)
        d["text_sha256"]=d.norm_text.map(lambda x:hashlib.sha256(x.encode("utf-8")).hexdigest())
        if (d.norm_text=="").any(): raise ValueError(f"{name} contains empty normalized text")
    allrows=pd.concat([train,val],ignore_index=True,sort=False)
    id_cols=[c for c in ("article_key","article_id","source_article_id") if c in allrows.columns]
    label_col=next((c for c in ("label_id","label","target") if c in allrows.columns),None)
    # Duplicate normalized text groups, including cross-split and conflicting labels.
    group_rows=[]
    for txt,g in allrows.groupby("norm_text",sort=False):
        if len(g)<2: continue
        splits=sorted(g.split.unique().tolist())
        labels=sorted(g[label_col].dropna().astype(str).unique().tolist()) if label_col else []
        rec={"text_sha256":g.text_sha256.iloc[0],"normalized_text":txt[:1000],
             "rows":len(g),"train_rows":int((g.split=="train").sum()),
             "validation_rows":int((g.split=="validation").sum()),
             "distinct_articles":int(g.article_key.astype(str).nunique()),
             "splits":";".join(splits),"label_values":";".join(labels),
             "label_conflict":bool(len(labels)>1)}
        for col in id_cols:
            rec[f"distinct_{col}"]=int(g[col].dropna().astype(str).nunique())
        group_rows.append(rec)
    dup=pd.DataFrame(group_rows)
    if not dup.empty: dup.sort_values(["label_conflict","validation_rows","rows"],ascending=False,inplace=True)
    dup.to_csv(out/"normalized_duplicate_groups.csv",index=False)
    # Cross-split duplicate groups and counts by label consistency.
    cross=dup.loc[(dup.train_rows>0)&(dup.validation_rows>0)].copy() if not dup.empty else dup.copy()
    cross.to_csv(out/"cross_split_duplicate_groups.csv",index=False)
    # ID mapping cardinality and aliases: article_id/source_article_id per article_key.
    maps=[]
    for col in id_cols:
        for key,g in allrows.groupby("article_key",dropna=False):
            vals=g[col].dropna().astype(str).unique()
            maps.append({"article_key":str(key),"field":col,"distinct_values":len(vals),
                         "values":";".join(vals[:20]),"mapping_violation":len(vals)>1})
    pd.DataFrame(maps).to_csv(out/"article_key_id_mapping.csv",index=False)
    # Whether article_id and source_article_id are exact aliases row-by-row.
    alias={}
    if {"article_id","source_article_id"}.issubset(allrows.columns):
        x=allrows[["article_id","source_article_id"]].dropna().astype(str)
        alias={"both_non_null_rows":len(x),"rowwise_equal":int((x.article_id==x.source_article_id).sum()),
               "rowwise_different":int((x.article_id!=x.source_article_id).sum()),
               "article_id_to_source_id_cardinality":int(x.groupby("article_id").source_article_id.nunique().max()),
               "source_id_to_article_id_cardinality":int(x.groupby("source_article_id").article_id.nunique().max())}
    # Manifest row order, fold membership, and repeated normalized text across folds.
    fold_report={}
    manifest_path=Path(a.manifest)
    if manifest_path.exists():
        m=pd.read_csv(manifest_path,low_memory=False)
        if "row_id" not in m or "fold" not in m: raise ValueError("Manifest requires row_id and fold")
        if len(m)!=len(train): raise ValueError("Manifest/train row count mismatch")
        if not (m.row_id.astype(int).to_numpy()==range(len(train))).all(): raise ValueError("Manifest row_id must be 0..N-1 in train order")
        m["norm_text"]=train.norm_text.to_numpy()
        text_folds=m.groupby("norm_text").fold.nunique()
        repeated=text_folds[text_folds>1]
        fold_report={"manifest_rows":len(m),"folds":sorted(int(x) for x in m.fold.unique()),
                     "normalized_text_groups_spanning_multiple_oof_folds":int(len(repeated)),
                     "rows_in_those_groups":int(m.norm_text.isin(repeated.index).sum())}
        repeated.rename("distinct_folds").reset_index().to_csv(out/"train_duplicate_text_across_oof_folds.csv",index=False)
        # Same article_key/source ID should not be split across folds.
        for col in ("article_key","source_article_id"):
            if col in m:
                counts=m.groupby(col).fold.nunique()
                fold_report[f"{col}_spanning_multiple_folds_groups"]=int((counts>1).sum())
    summary={
        "train_rows":len(train),"validation_rows":len(val),
        "unique_normalized_text_train":int(train.norm_text.nunique()),
        "unique_normalized_text_validation":int(val.norm_text.nunique()),
        "duplicate_groups_all_splits":len(dup),
        "cross_split_duplicate_groups":len(cross),
        "cross_split_duplicate_rows":int(cross.rows.sum()) if len(cross) else 0,
        "conflicting_label_duplicate_groups":int(dup.label_conflict.sum()) if len(dup) and "label_conflict" in dup else 0,
        "id_columns":id_cols,"article_id_source_alias_audit":alias,
        "fold_manifest_audit":fold_report,
        "normalization":"Unicode NFKC + casefold + collapsed whitespace; punctuation retained",
        "warning":"Repeated text is not automatically leakage or an error; inspect label conflicts and article/source grouping before changing splits. No rows were deleted or relabeled."
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    print("Outputs:",out.resolve())
if __name__=="__main__": main()
