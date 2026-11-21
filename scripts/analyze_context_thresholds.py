#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Threshold sensitivity analysis for paired sentence/context OOF predictions.

Uses existing OOF scores only; no model fitting and no threshold is selected on
an outer fold for a claimed unbiased estimate. Per-fold best thresholds are
reported as diagnostics, not as deployable thresholds.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss)


def summarize(y, score, threshold):
    pred = (score >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "n": int(len(y)), "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "human_f1": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "machine_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "human_precision": float(precision_score(y, pred, pos_label=0, zero_division=0)),
        "human_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "machine_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "machine_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "tn": int(cm[0,0]), "fp_human_as_machine": int(cm[0,1]),
        "fn_machine_as_human": int(cm[1,0]), "tp": int(cm[1,1]),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "average_precision": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "brier": float(brier_score_loss(y, score)),
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default="results/context_oof")
    ap.add_argument("--output-dir", default="results/context_oof_threshold_analysis")
    ap.add_argument("--thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    a=ap.parse_args()
    root,out=Path(a.input_dir),Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    files={"sentence":root/"sentence_matched_oof_predictions.csv","context":root/"context_oof_predictions.csv"}
    frames={k:pd.read_csv(v,low_memory=False) for k,v in files.items()}
    key="oof_row_id"; required={key,"fold","y_true","context_p_machine"}
    for name,d in frames.items():
        miss=required-set(d.columns)
        if miss: raise ValueError(f"{name} missing columns: {sorted(miss)}")
        if d[key].isna().any() or d[key].duplicated().any(): raise ValueError(f"{name} row IDs not unique/non-null")
        d["context_p_machine"]=pd.to_numeric(d["context_p_machine"],errors="coerce")
        if not np.isfinite(d["context_p_machine"]).all() or not d["context_p_machine"].between(0,1).all():
            raise ValueError(f"{name} scores must be finite probabilities in [0,1]")
        d.sort_values(key,inplace=True); d.reset_index(drop=True,inplace=True)
    s,c=frames["sentence"],frames["context"]
    if not np.array_equal(s[key],c[key]): raise ValueError("Paired row IDs differ")
    for col in ("fold","y_true"):
        if not np.array_equal(s[col],c[col]): raise ValueError(f"Paired {col} differs")
    y=s.y_true.astype(int).to_numpy()
    if not set(np.unique(y)).issubset({0,1}): raise ValueError("y_true must be 0/1")
    thresholds=sorted(set(float(x) for x in a.thresholds.split(",")))
    if not thresholds or any(x<=0 or x>=1 for x in thresholds): raise ValueError("Thresholds must be in (0,1)")
    rows=[]; best=[]
    for model,d in frames.items():
        score=d.context_p_machine.to_numpy(float)
        for t in thresholds:
            rows.append({"model":model,"scope":"pooled_oof","fold":"ALL",**summarize(y,score,t)})
        for fold,g in d.groupby("fold",sort=True):
            gy=g.y_true.astype(int).to_numpy(); gs=g.context_p_machine.to_numpy(float)
            for t in thresholds:
                rows.append({"model":model,"scope":"fold","fold":int(fold),**summarize(gy,gs,t)})
            fold_grid=[summarize(gy,gs,t) for t in thresholds]
            optimum=max(fold_grid,key=lambda x:(x["macro_f1"],-abs(x["threshold"]-0.5)))
            best.append({"model":model,"fold":int(fold),"selection":"same_fold_diagnostic_only",**optimum})
    pd.DataFrame(rows).to_csv(out/"threshold_metrics.csv",index=False)
    pd.DataFrame(best).to_csv(out/"per_fold_best_threshold_diagnostic.csv",index=False)
    # Paired cluster bootstrap at fixed 0.5 and at a small set of common thresholds.
    bootstrap=[]
    cluster_col=next((x for x in ("article_key","article_id","source_article_id") if x in s.columns and s[x].notna().all()),None)
    if cluster_col:
        groups=s[cluster_col].astype(str).to_numpy(); uniq,inv=np.unique(groups,return_inverse=True)
        rng=np.random.default_rng(a.seed)
        for t in (0.35,0.40,0.45,0.50,0.55,0.60,0.65):
            diffs=[]
            ps=(s.context_p_machine.to_numpy(float)>=t).astype(int)
            pc=(c.context_p_machine.to_numpy(float)>=t).astype(int)
            for _ in range(a.bootstrap_reps):
                mult=rng.multinomial(len(uniq),np.full(len(uniq),1/len(uniq)))
                w=mult[inv]
                vals=[]
                for pred in (ps,pc):
                    tp=w[(y==1)&(pred==1)].sum(); fp=w[(y==0)&(pred==1)].sum()
                    fn=w[(y==1)&(pred==0)].sum(); tn=w[(y==0)&(pred==0)].sum()
                    f1p=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
                    f1n=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0
                    vals.append((f1p+f1n)/2)
                diffs.append(vals[1]-vals[0])
            bootstrap.append({"cluster":cluster_col,"clusters":len(uniq),"threshold":t,
                "delta_macro_f1_context_minus_sentence":float(np.mean(diffs)),
                "ci95_low":float(np.quantile(diffs,.025)),"ci95_high":float(np.quantile(diffs,.975)),
                "bootstrap_reps":len(diffs)})
    pd.DataFrame(bootstrap).to_csv(out/"paired_cluster_bootstrap_by_threshold.csv",index=False)
    report={"rows":len(s),"folds":sorted(int(x) for x in s.fold.unique()),
        "paired_ids":True,"thresholds":thresholds,"cluster_column_used":cluster_col,
        "warning":"Per-fold best thresholds are selected and evaluated on the same fold; diagnostic only, not unbiased performance estimates.",
        "interpretation":"Threshold curves change the decision rule only. ROC-AUC/AP are threshold-independent ranking metrics."}
    (out/"integrity_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))
    print(pd.DataFrame(best).to_string(index=False))
    print("Outputs:",out.resolve())
if __name__=="__main__": main()
