# CatalogIQ modeling findings

## Retailer transfer: October 10, 2026

Our original **96.95% accuracy demonstrates excellent performance on a familiar
retailer mix, but it does not establish that level of performance on entirely
new retailers**.

We trained a fresh MiniLM model on all **43,969 Amazon training records**, with
ProductCategory, **256 tokens**, seed 42, and three fixed epochs. Both models
were compared on identical validation records from Walmart, Walgreens, CVS,
Target, and Kroger. Amazon validation records provided a separate control.

| Validation records | Original mixed-retailer model | Amazon-only model |
| --- | ---: | ---: |
| Other retailers: 2,511 records, accuracy | 94.46% | 87.10% |
| Same other-retailer records, macro F1 | 89.92% | 77.96% |
| Amazon control: 9,441 records, accuracy | 97.61% | 97.61% |

The original 96.95% score covers the whole validation pool. **94.46% is the
appropriate comparison on the same other-retailer records.** Their accuracy
dropped by **7.37 percentage points** (95% paired group-bootstrap interval:
**5.57-9.44 points**). Macro F1 dropped by **11.96 points**.

Internal Analgesics, Digestive Health, and Allergy accounted for **80.5% of the
additional errors**. Excluding the 11 held-retailer records flagged by the
independent near-neighbor audit left essentially the same accuracy gap.

All held-retailer category paths were absent from Amazon training. However,
training volume, class mix, assortment, and wording also changed. The experiment
demonstrates weaker retailer transfer; it does not isolate category memorization
as its cause. Results use one training seed, and bootstrap intervals describe
validation-group sampling uncertainty.

The final test remained unopened and unscored. Existing checkpoints were
preserved, and the new model was saved separately.

### Saved evidence

- [Validation metrics](../../../data/processed/segment_retailer_holdout_category_20261010/validation_metrics.csv)
- [Per-class results](../../../data/processed/segment_retailer_holdout_category_20261010/per_class_comparison.csv)
- [Paired uncertainty intervals](../../../data/processed/segment_retailer_holdout_category_20261010/paired_differences.csv)
- [Experiment settings and input hashes](../../../data/processed/segment_retailer_holdout_category_20261010/protocol.json)
- [Model setup and earlier findings](catalogiq_splitting_and_transformer_setup.md)
