# Stage 1 — Baseline protocols and observed results (2026-11-22)

**Evidence:** aggregate files in this repository at PR #4 head (`2e64862`), including
`results/analysis_review/baseline_comparison_reference.csv`,
`results/oof_fusion/metrics_comparison.csv`,
`results/context_oof_analysis/metrics_by_fold.csv`, and historical Stacking aggregates.
The 18-row protocol-labelled reference CSV is the machine-readable baseline
snapshot. Run `scripts/build_baseline_comparison.py` for a new local snapshot.

**Interpretation:** all rows below are **development results**, not blind test
results. Never rank different protocol IDs as one leaderboard. Positive class
is Machine=1, negative class Human=0; FP is Human → Machine.

## A. Fold-matched Novel Validation (11,682 sentences)

One fold-specific prediction is chosen per source group. Fusion thresholds
were selected using OOF labels (not Novel labels), but Novel has been used
repeatedly in project development.

| Model | Accuracy | Macro-F1 | Human F1 | Machine F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| SVM | 0.8857 | 0.8800 | 0.8537 | 0.9062 | 717 | 618 |
| RoBERTa | 0.9147 | 0.9093 | 0.8874 | 0.9313 | 686 | 311 |
| Stacking | **0.9217** | **0.9174** | **0.8985** | **0.9362** | **562** | 353 |
| SVM Veto | 0.9191 | 0.9144 | 0.8943 | 0.9345 | 615 | 330 |

All rows use the same 4,613 Human / 7,069 Machine targets, but
**this is not a matched Context-vs-Sentence experiment**.
Compared with RoBERTa, Stacking yields +82 correct predictions, -124 Human FP,
and +42 Machine FN. Its Macro-F1 improvement is +0.0080.

**Historical archived paired statistics** (not a new corrected script run):
Stacking vs RoBERTa: 266 Stacking-only correct, 184 RoBERTa-only correct,
10,501 both correct, 731 both wrong. Sentence-level McNemar exact p=0.000129,
which ignores within-source dependence. The archived article-ID clustered
bootstrap reports ΔMacro-F1=+0.0081 with 95% percentile interval
[+0.0038,+0.0125], 2,000 replicates, 619 source clusters. The underlying
bootstrap resamples sources but cannot eliminate all repeated-text dependence.
**Re-run corrected aggregate-only analysis and verify it before citing its
numerical output as a newly confirmed result.** The tracked older metrics file
used *raw fold-local scores* for base-model AUC/AP; use the corrected script's
fold-standardized `roberta_z`/`svm_z` instead.

## B. Pooled five-fold OOF — historical SVM/RoBERTa/fusion protocol

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| SVM | 0.8882 | 0.8765 |
| RoBERTa | 0.9184 | 0.9080 |
| Stacking | 0.9250 | 0.9163 |
| SVM Veto | 0.9209 | 0.9113 |

These are **cross-fit training/selection diagnostics**. The meta-model is
cross-fitted by original OOF fold with normalized-text purge, but this is not a
fully nested outer validation of base training + meta-model selection.

## C. Strict window-purged, sentence-matched Context OOF

Fixed threshold 0.5; 52,864 pooled OOF sentences. The Sentence and Context
models use the **same strictly window-purged fold rows, training recipe and
seeds**. Do not compare these scores directly with Section A.

| Fold | Sentence Macro-F1 | Context Macro-F1 | Context − Sentence |
|---|---:|---:|---:|
| 0 | 0.8569 | 0.9089 | +0.0519 |
| 1 | 0.8873 | 0.9163 | +0.0290 |
| 2 | 0.8939 | 0.8916 | -0.0023 |
| 3 | 0.8718 | 0.9130 | +0.0412 |
| 4 | 0.8682 | 0.8667 | -0.0014 |
| Pooled OOF | 0.8760 | 0.8998 | +0.0238 |

Fold 4 Context AUC falls from 0.9664 to 0.9129; AP from 0.9824 to
0.9004. Human FP increases from 1,015 to 1,091, while Machine FN falls
from 143 to 62. The Fold-4 analysis is descriptive, not evidence of a
generalizable gate.

**Invalid analysis excluded:** the old Fold-4 harm-classifier was trained on
Sentence-correct-only cases, with both classifiers' predictions as features.
This makes its harm label nearly recoverable from their disagreement. Its
near-perfect AUC/AP **must not** be treated as learned gating evidence.

## D. Metric conventions and next verification

- Report Accuracy, Machine/Human and macro Precision/Recall, Macro-F1 and
  Micro-F1 to four decimal places. Micro-F1 equals Accuracy in this setting.
- Ranking metrics are score-scale-dependent; the historical pooled Context
  raw AUC/AP and fold-standardized margin AUC/AP are **different protocols**.
- Do not tune thresholds on the 500-article unlabeled test set or report its
  Accuracy/F1.
- Do not put sentence text, article IDs, OOF row files or model weights in
  new public commits.

Run from project root with private artifacts available locally:

```bash
python scripts/analyze_stacking_errors.py \
  --input results/oof_fusion/novel_fusion_predictions.csv \
  --output-dir results/stacking_error_analysis_corrected \
  --fusion-config results/oof_fusion/fusion_config.json \
  --bootstrap-reps 2000 --seed 42 --aggregate-only

python scripts/verify_stacking_aggregates.py \
  --analysis-dir results/stacking_error_analysis_corrected \
  --reference results/oof_fusion/metrics_comparison.csv \
  --summary-out results/stage1_local_audit/stacking_verification.json

python scripts/build_baseline_comparison.py \
  --results-dir results --output-dir results/analysis_review
```

The verifier checks class confusion counts, accuracy/F1 arithmetic, agreement
groups, McNemar counts, bootstrap structure, and **corrected fold-standardized
AUC/AP consistency** against the saved fusion reference. It does not rerun the
underlying bootstrap independently, establish nested CV, or confer a statistical
guarantee. If any check fails, do not freeze results.
