# E2 — Selective Context Use: minimal no-retraining experiment

## Question and frozen baseline

**Question:** Under which sentence-length and model-disagreement conditions is
context useful or harmful for human-vs-machine sentence detection, and do
simple label-free selective rules help over a matched Context baseline?

The existing Stage-1 **five-fold OOF SVM+RoBERTa Logistic Stacking**
with threshold **0.585** is frozen and remains the accepted executable
baseline (the unlabeled test ran with 500 articles and 4,264 sentences).
This E2 experiment does not overwrite its checkpoints, inference pipeline
or unlabeled-test output. No hidden-test Accuracy/F1 can be inferred.

## Mandatory paired evaluation contract

Use **only** the previously trained Sentence-matched and Context models
from the exact same strict context-window purged training folds:

- Both inputs have 52,864 OOF rows, one row per `oof_row_id`.
- Pair by `oof_row_id`, not by sentence text. Require matching labels,
  original fold IDs, article/source IDs, sentence position and text.
- Enforce that each `source_article_id` belongs to exactly one original
  fold; source IDs can repeat across article records.
- Compare identical rows, fixed positive label Machine=1, fixed
  probability threshold **0.5** and identical metric definitions.
- Do not use older sentence-only RoBERTa (different purge) as the primary
  control; do not compare this OOF Macro-F1 directly with the Novel
  Stacking Macro-F1=0.9174 (different protocols).

Historical matched reference, before this experiment:

| Matched model | Pooled OOF Macro-F1 | Accuracy |
|---|---:|---:|
| Sentence-matched | 0.875989 | 0.894087 |
| Context | 0.899834 | 0.914384 |

Context is worse on some folds, including Fold 4:
Sentence 0.868178 vs Context 0.866750 Macro-F1.

## Fixed candidates: no new training, no label-derived input features

All use Machine-positive probability scores with **threshold 0.5**:

| Rule | Inference behavior |
|---|---|
| Sentence-matched | Always use target-only score |
| Context | Always use context-window score |
| mean_50 | Average the two probabilities 50/50 |
| context_75 | 75% context + 25% sentence probability |
| short_context_12 | Context if target word count <=12, else Sentence |
| short_context_24 | Context if <=24 words, else Sentence |
| long_human_veto_20 | For targets >=20 words, override Context-Machine only when Sentence predicts Human |
| long_human_veto_32 | Same override for targets >=32 words |

Word count uses whitespace splitting of original target text. No
ground-truth, correctness/error label, per-source ID feature, domain
identity or threshold tuning is allowed inside a rule.

These small, finite candidate settings were chosen for an inexpensive
exploratory comparison; they are **not** externally pre-registered.
Scores are not presumed perfectly calibrated. Report Human FPR
and Machine recall next to Macro-F1 to expose error tradeoffs.

## Descriptive fixed-rule comparison vs cross-fitted selection

The script reports every fixed candidate on all 52,864 OOF rows, and
also runs a five-fold **rule-selection diagnostic**:

1. For heldout fold `k`, use only the **four other folds'** OOF
   labels/scores to rank the seven rules (Sentence-only remains
   a baseline and is not in the selection pool).
2. A rule is eligible only when its four-fold Human FPR is no more than
   Context Human FPR + 0.002 (absolute rate units).
3. Choose eligible rule with largest four-fold Macro-F1; exact ties
   prefer Context and then the order in the fixed candidate list.
4. Apply that selected rule to fold `k` **without reading fold k labels**.
   Concatenate five heldout predictions and report *global* confusion,
   Macro-F1, Accuracy and Human FPR and **each fold**.
5. Report source-ID-cluster paired bootstrap 95% percentile CI for
   (selected − Context) Macro-F1, Human FPR and Accuracy.

This is meta-stage cross-fitting on existing OOF scores, **not**
a nested retraining of upstream base models, a fully independent test,
or a proof of deployable gate generalization. Earlier work looked
at these folds and Novel Validation. Selection and estimates remain
development/exploratory.

**Exploratory stop rule (not a course requirement):** Consider the
optional E3 learnable selector only if selected-rule Macro-F1 improves
over Context by >=0.002, Human FPR worsens by <=0.002, at least 3 of 5
folds have nonnegative Macro-F1 deltas, and the source-cluster
bootstrap Macro-F1 delta CI has positive lower bound. Otherwise
**stop E3 and keep the frozen Stacking baseline**. Do not tune the
criteria on E2 evaluation outputs.

## Reproducible command

From the repository root, with the existing Python environment active:

```bash
python -m unittest discover -s tests -p 'test_e2_selective_context.py' -v
python -m scripts.experiment_e2_selective_context \
  --input-dir results/context_oof \
  --output-dir results/e2_selective_context \
  --bootstrap-reps 2000 \
  --seed 2026
```

The local ignored directory contains:

- `overall.csv`: complete-sample metrics for each rule and selected rule.
- `by_fold.csv`: five-fold metrics for all candidates.
- `by_length.csv`: fixed word bins, Human/Machine supports and
  Context-helped vs Context-harmed transitions.
- `transitions.csv`: paired correction/error count by true class.
- `fold_selection.csv`: rule chosen per heldout fold and
  **other-four-fold** training-only metrics.
- `bootstrap.csv`: source-ID clustered paired difference estimates.
- `decision.json`: explicit stop/continue gate with limitations.

All written artifacts contain **aggregate counts and statistics only**.
The raw paired OOF files can contain text and IDs, and must stay local;
do not upload rows or checkpoints in PRs. The historical GitHub
repository already contains some prediction files—this experiment
must not add more raw diagnostic files.

## Interpretation and delivery

E2 is worthwhile as a course experiment even if *all* simple gates
fail: report why unconditional Context helped overall, how Human FP
and different length bands trade off, and why selective heuristics
were rejected. Do not report an E3 gate gain unless the prescribed
comparisons pass, and do not let this optional experiment delay
formal teacher-schema verification and course report completion.


## First completed run: aggregate evidence and limits

GitHub CI [E2 paired-OOF run #44926701731](https://github.com/q3483109023/AIGTD/actions/runs/44926701731)
executed the committed paired OOF artifacts on GitHub, using
1,000 source-cluster bootstrap replicates and seed 2026.
Seven synthetic tests passed before the actual experiment.

| Rule | OOF Macro-F1 | Accuracy | Human FP | Human FPR |
|---|---:|---:|---:|---:|
| Sentence-matched | 0.875989 | 0.894087 | 4,196 | 0.236648 |
| Context | 0.899834 | 0.914384 | 3,636 | 0.205065 |
| 50/50 mean | 0.897943 | 0.912871 | 3,711 | 0.209294 |
| 75% Context | 0.899977 | 0.914498 | 3,630 | 0.204726 |
| Context for <=12 words, else Sentence | 0.887548 | 0.903753 | 3,877 | 0.218657 |
| Context for <=24 words, else Sentence | 0.900072 | 0.914573 | 3,626 | 0.204501 |
| **Long Human-veto >=20 words** | **0.906659** | **0.919378** | **3,187** | **0.179742** |
| Long Human-veto >=32 words | 0.901488 | 0.915651 | 3,551 | 0.200271 |

Heldout rule selection on each of the five original folds selected
the same **long Human-veto >=20** rule; thus its fold-selected pooled
metrics match the fixed-rule descriptive metrics. It improved
Macro-F1 over Context by **0.006825** and reduced Human FPR
by **0.025323 absolute** (2.53 percentage points). In all five
folds, the Macro-F1 change was nonnegative. The clustered bootstrap
95% percentile interval for the Macro-F1 difference was
**[0.005292, 0.008303]** (2,473 source groups; 1,000 replicates).
This is conditional on the already-trained base model outputs and
is not a confidence guarantee on unseen data.

**Important trade-off:** the veto reduces Context's Human false
positives **3,636 -> 3,187** but increases Machine false negatives
approximately **890 -> 1,075**; Machine Recall decreases from
**0.974668 to 0.969402**. The simple condition favors lower Human
false-positive risk at some Machine recall cost. Compare this
trade-off openly in the final report.

**Evidence status:** promising *development OOF evidence only*.
The prior diagnostics of Fold 4 and other exploratory work make
this a hypothesis-generating experiment, not an untouched benchmark
or a proven generalizable performance gain. Source bootstrap
accounts for within-source clustering but not full upstream model
retraining, analysis-driven rule discovery, or distribution shift.

**Decision:** E2 passes the script's exploratory screen, so a
follow-on E3 is **permitted but optional**, not automatically required.
Given the simplicity and interpretability of this rule, the
conservative course-project path is to keep it as a meaningful
small-method experiment with error trade-off analysis, while
retaining the **frozen Stacking** deployment baseline. Deploying a
Context+Sentence selective rule on the unlabeled test would require
a separate, audited paired inference path using its own strict
OOF model protocol; do not mix it into Stacking at inference time.

## E2 follow-up: unchanged-rule Novel recheck

The fixed **>=20-word Human-veto** rule was also checked on the **11,682
previously used Novel Validation targets**, separately under (a) historical
five-fold mean-probability evaluation and (b) label-free source-fold
matching. It was **not retuned**. Against its same-protocol Context
control, the Macro-F1 improvement was respectively **+0.006168** and
**+0.009788**; Human FP decreased while Machine FN increased.

This supports a useful development-only E2 finding, **not** an untouched
independent test claim. See `docs/e2_fixed_rule_novel_recheck.md` for
the exact metrics, source-group bootstrap, verified CI and privacy-safe
server command. The frozen Stacking baseline and unlabeled test predictions
remain unchanged.
