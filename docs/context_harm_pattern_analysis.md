# Fold-4 Context Harm Pattern Analysis

Run from the repository root after the paired OOF prediction files exist:

```bash
python scripts/analyze_context_harm_patterns.py \
  --input-dir results/context_oof \
  --output-dir results/context_harm_patterns \
  --fold 4
```

The script performs four analyses:

1. Paired outcomes (both correct, both wrong, Context-only correct, Sentence-only correct), split by true class.
2. Harm/help rates by target length and previous/next context length.
3. Article-level concentration summaries. Article identifiers are used internally for grouping but are not exported.
4. An exploratory risk-prediction experiment among rows where Sentence-only is correct: predict whether adding Context flips the correct prediction to an error. Logistic regression and ExtraTrees are evaluated with GroupKFold, keeping each article entirely within one split. Features are limited to inference-available model scores and text/context length/style statistics.

## Interpretation

- The primary risk target is a *harmful Context flip*, not generic classification error.
- Grouped cross-validation tests whether harm risk is predictable beyond memorizing article identity.
- Compare average precision against the harm prevalence baseline. A useful risk model should improve AP and precision in the highest-risk deciles across article-held-out folds.
- This is a Fold-4 exploratory analysis. It does not establish causality or generalization to other folds / Novel Validation. A gate should not be trained or claimed effective from these results alone.
- Aggregate reports contain no sentence text, row IDs, or article identifiers. The script does not modify source predictions or model checkpoints.
