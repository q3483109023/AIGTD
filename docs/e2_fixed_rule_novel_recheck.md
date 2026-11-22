# E2 fixed 20-word Human-veto — Novel development recheck

**Evidence type:** exploratory replication on previously used **Novel
Validation**. This is not a new untouched external test, and it does not
alter the frozen Stage-1 Stacking test predictions.

## Lock the tested rule before reading Novel outcomes

The fixed rule is the one from E2's five-fold paired OOF experiment:

- Machine=1, Human=0; fixed classification threshold **0.5**.
- Use Context's Machine probability by default.
- If the target sentence contains **at least 20 whitespace-delimited
  words**, Context classifies it Machine and Sentence-matched classifies
  it Human, veto Context and predict Human.
- No new selection of length threshold, probability threshold,
  model architecture or feature weight. No label or ID enters the
  inference-time rule.

The models are the **paired Sentence-matched and Context models trained
with the same strict context-window-purged OOF protocol**. They are not
the OOF SVM/RoBERTa Stacking model.

## Two predeclared Novel prediction protocols, reported separately

Each Novel target has five precomputed Sentence-matched probabilities and
five Context probabilities, one from each trained fold. There are
**11,682 distinct targets x 5 fold outputs = 58,410 prediction rows
per model**. Treat each target exactly once in the main evaluation.

1. **Five-fold probability mean:** separately average five predicted
   probabilities from each model per `novel_row_id`, then apply the
   unchanged rule. This is consistent with the prior historical Context
   Novel evaluation. First reproduce exact Sentence/Context baseline
   Accuracy and Macro-F1 from
   `results/context_oof_analysis/novel_metrics.csv`.
2. **Source-fold matched:** assign each source article group (by its
   source ID, not by record key) deterministically to one of five folds
   using the existing label-free LPT fold-balancing rule, computed
   once per unique Novel target. Pair Sentence and Context scores from
   that same assigned fold, then apply the unchanged rule. This resembles
   original OOF's one-fold-per-target feature construction, but is a
   *different prediction protocol* from five-fold mean.

Pair using `(novel_row_id,fold)`. Assert matching labels, source IDs,
article record keys, sentence indices and raw target text. Reject
duplicate/missing folds, nonfinite scores, inconsistent labels, and
source groups split across assigned folds.

## First GitHub CI result (2026-11-22)

The [GitHub CI run #44927335934](https://github.com/q3483109023/AIGTD/actions/runs/44927335934)
passed **6/6 synthetic tests**, the 11,682-target real Novel
computation and the exact historical mean-five baseline check.
This run uses **1,000 source-ID cluster-bootstrap draws** (seed 2026).

| Novel protocol | Model | Macro-F1 | Accuracy | Human FP | Machine FN |
|---|---|---:|---:|---:|---:|
| Five-fold mean | Sentence-matched | 0.885089 | 0.894967 | 1,098 | 129 |
| Five-fold mean | Context | 0.906406 | 0.913970 | 935 | 70 |
| Five-fold mean | **Fixed Human-veto >=20** | **0.912574** | **0.919192** | **851** | **93** |
| Source-fold matched | Sentence-matched | 0.877963 | 0.888718 | 1,156 | 144 |
| Source-fold matched | Context | 0.895837 | 0.904640 | 1,027 | 87 |
| Source-fold matched | **Fixed Human-veto >=20** | **0.905625** | **0.912943** | **907** | **110** |

The fixed veto improves Macro-F1 against its **same-protocol Context
control** by **+0.006168** under five-fold mean and by **+0.009788**
under source-fold matching, with bootstrap percentile 95% delta CIs:

- Five-fold mean: **[+0.004013, +0.008730]**.
- Source-fold matched: **[+0.006895, +0.012892]**.

Both use 619 distinct source IDs and the same 11,682 Novel targets,
though their predictions and therefore metrics differ. In the
five-fold mean protocol Human FPR improves from **0.202688 to
0.184479**, while Machine Recall declines from **0.990098 to
0.986844**. In the source-fold-matched protocol Human FPR declines
from **0.222632 to 0.196618**, while Machine Recall declines from
**0.987693 to 0.984439**.

The rule **does not dominate** Context on every metric: the increased
Machine false negatives are part of the mechanism's cost.

## Local verification on the user's private server

From the existing project checkout, pull the branch, then run:

```bash
python -m unittest discover -s tests -p 'test_e2_novel_fixed_rule.py' -v

python -m scripts.validate_e2_fixed_rule_novel \
  --input-dir results/context_oof \
  --reference results/context_oof_analysis/novel_metrics.csv \
  --output-dir results/e2_novel_fixed_rule \
  --expected-targets 11682 \
  --bootstrap-reps 2000 \
  --seed 2026
```

The output directory is ignored by Git. It contains only aggregate
`metrics.csv`, `class_errors.csv`, `bootstrap.csv` and
`summary.json`. The output's `"status": "PASS"` means the
schema/alignment/reference and computation checks succeeded;
the resulting predictive gain is still *development evidence*.

## Conclusions and limitations

1. The same fixed rule improves both the five-fold mean and source-fold
   matched Novel comparisons against their corresponding Context controls.
   This adds evidence of **stability across a change of evaluated
   samples and aggregation protocol** but cannot prove generalization to
   unobserved independent data.
2. Novel has been repeatedly used in prior model selection and error
   analysis. The length cutoff originated from E2 exploration; this
   check is post-development replication, **not a confirmatory test**.
3. Source bootstrap is conditional on frozen base predictions and does
   not account for upstream model-training uncertainty or all previous
   researcher choices.
4. The course's frozen operational baseline remains five-fold
   **OOF Stacking, threshold 0.585**. No modification to its unlabeled
   500-article predictions is warranted by this separate experiment.
5. E2 supplies a clear limited innovation: disagreement-aware,
   length-conditioned selection of context, quantified alongside Human
   FP and Machine FN. Avoid claiming it establishes statistical
   false-positive guarantees or beats Stacking under a common split.

## Private-server 2,000-replicate acceptance record (2026-11-22)

The user independently ran the commands above on the private Ubuntu server.
All **six** regression tests passed, followed by the real 11,682-target Novel
evaluation returning `"status": "PASS"`, with five-fold-mean reference
metrics matching the saved historical report. Each model had **58,410**
fold prediction rows before target-level aggregation. There were **619**
unique source groups.

The observed deterministic test metrics reproduced the GitHub CI values:

| Protocol | Context Macro-F1 | Fixed-veto Macro-F1 | ΔMacro-F1 | 2,000 source bootstrap 95% percentile CI |
|---|---:|---:|---:|---|
| Five-fold probability mean | 0.906406 | 0.912574 | +0.006168 | **[+0.003962, +0.008624]** |
| Source-fold matched | 0.895837 | 0.905625 | +0.009788 | **[+0.006861, +0.012892]** |

The predicted labels, metrics, fixed **20-word** condition and
**0.5** classification threshold were unchanged. The user verified
`reference_five_fold_mean_baselines_match=true`. This is still
development evidence based on reused Novel Validation, not a fresh
independent test or teacher-upload-format validation.

**Stage decision:** close E2 as a successful *exploratory* course experiment.
Do not introduce additional hyperparameters, E3 training or test-label
claims merely to increase a development score. Retain the separate frozen
Stacking inference pipeline and use the above numbers only in the
correctly identified paired Context/Sentence experiment.
