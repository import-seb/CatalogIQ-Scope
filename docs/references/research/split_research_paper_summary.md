# Research Notes: Train/Validation Split Methodology

These notes summarize ten papers relevant to preventing data leakage in CatalogIQ.
They cover group-aware splitting, near-duplicate detection, and product matching.
The summaries and references are retained from the original notes; this edit does
not independently verify the papers.

## Main implications for CatalogIQ

- Determine which identifiers reliably connect related product records before
  choosing a grouping rule.
- Assign each product group to a single split so related records stay together.
- Use text similarity to supplement identifiers and audit cross-split overlap.
  Validate similarity-based matches before treating them as reliable groups.
- Distinguish identical-product matching from broader product-family grouping.
  The product-matching papers do not establish a train/test splitting procedure.
- Report both row counts and unique product-group counts for each split.

The papers' split ratios are examples from their studies, not a finalized
CatalogIQ split strategy.

## Group-aware splitting

### 1. Patient-level grouping in dementia prediction

**Why it matters:** The researchers had multiple MRI records from the same patients. Random splitting could put one patient in both training and testing, causing data leakage.

**Method:** They grouped records by patient ID, then used:

- 80% training / 20% testing.
- Group-aware 5-fold cross-validation on the training data.
- Oversampling only within training.

**Application to CatalogIQ:** Our products are like their patients. Related products should stay in the same split. Unlike their dataset, however, we first need to determine which columns reliably identify related products.

**Reference (APA 7):**

Ghiasi, M. M., Falck, R. S., Liu-Ambrose, T., & Tam, R. C. (2026). Explainable machine learning for predicting longitudinal dementia status: Establishing a leakage-free benchmark. PLOS Digital Health, 5(5), e0001409. https://doi.org/10.1371/journal.pdig.0001409

### 2. Configuration-level grouping in steel prediction

**Why it matters:** The researchers had repeated measurements of the same steel configurations. Random splitting could put related measurements in both training and testing, causing data leakage.

**Method:**

- Grouped 1,255 records into 603 unique configurations based on material composition and processing conditions.
- Randomly assigned entire groups to 80% training and 20% testing.
- Used group-aware cross-validation and repeated resampling to evaluate reliability.

**Application to CatalogIQ:** We can similarly group products using combinations of identifying attributes rather than relying on a single column. Each group receives one split assignment.

**Reference (APA 7):**

Alqurashi, Y. (2026). Data-driven prediction of tensile strength in heat-treated steels using random forests for sustainable materials design. Sustainability, 18(2), 1087. https://doi.org/10.3390/su18021087

## Detecting and auditing near-duplicate leakage

### 3. Near-duplicate detection in CIFAR

**Why it matters:** Researchers discovered that 3.3%–10% of CIFAR test images had duplicates in training, potentially inflating model accuracy.

**Method:**

- Used CNN feature embeddings and nearest-neighbor similarity to identify duplicates.
- Manually reviewed exact duplicates, near-duplicates, and highly similar images.
- Replaced duplicated test images with new, nonduplicated images.
- Kept the original training set unchanged.

**Application to CatalogIQ:** We could use text similarity to identify related products when `SKU`, `JoiningKey`, or `ProductName` alone are insufficient. Their method supports checking for near-duplicate products across splits, not just exact matches.

**Reference (APA 7):**

Barz, B., & Denzler, J. (2020). Do we train on test data? Purging CIFAR of near-duplicates. Journal of Imaging, 6(6), Article 41. https://doi.org/10.3390/jimaging6060041

### 4. Text overlap auditing with SplitGuard

**Why it matters:** Traditional splitting can allow identical or nearly identical text records into both training and testing, inflating model performance.

**Method:**

- Developed SplitGuard, a framework for detecting leakage after splitting.
- Used MinHash and Locality-Sensitive Hashing (LSH) to identify near-duplicate records.
- Tested deterministic samples from 16 dataset variants across 11 dataset families.
- Measured overlap between training and evaluation datasets.

**Application to CatalogIQ:** We could use similar text-matching techniques on `ProductName` and `ProductDescription` to detect related products across our splits, even when their SKUs differ.

**Limitation:** This paper audits existing splits; it does not propose a new splitting algorithm. The analysis is based on the available abstract.

**Reference (APA 7):**

Karunanayaka, I. P. (2026). SplitGuard: A resource efficient framework for auditing train and eval overlap and near duplicate contamination in NLP datasets. TechRxiv. https://doi.org/10.36227/techrxiv.177205027.77074976/v1

### 5. Near-duplicate contamination in MRI benchmarks

**Why it matters:** Researchers wanted to measure how near-duplicate MRI images appearing across training and testing inflate model performance.

**Method:**

- Screened MRI images for near-duplicates.
- Compared screened data against deliberately contaminated training data.
- Kept validation and test sets identical.
- Repeated experiments using five random seeds.
- Found that contamination inflated accuracy by approximately 2.6 percentage points.

**Application to CatalogIQ:** We could identify near-duplicate products using `ProductName` and `ProductDescription`, then check for cross-split contamination. However, similarity detection alone may miss related products, so reliable identifiers remain important.

**Reference (APA 7):**

Huang, Y., Sun, Z., Chen, J., Yang, T., Xie, L., & Lin, X. (2026). Near-duplicate contamination inflates brain MRI benchmark performance: A size-matched, multi-seed MedNeXt-B study. Frontiers in Medicine, 13, Article 1955863. https://doi.org/10.3389/fmed.2026.1955863

## Evidence for entity-level splitting

### 6. Entity-level splitting in geoscience

**Why it matters:** Related geological samples were appearing in both training and testing, inflating model performance. Some F1 scores increased by over 20 percentage points due to leakage.

**Method:**

- Compared random row-level splitting (observation-splitting) with group-level splitting (entity-splitting).
- Grouped measurements belonging to the same geological entity.
- Assigned entire entities to either training or testing.
- Compared performance across simulated datasets and three real-world case studies.

**Application to CatalogIQ:** Very high. Our product records resemble their geological observations. We should group related products before splitting and report both the number of rows and the number of unique product groups.

**Reference (APA 7):**

Scharf, T., Daggitt, M. L., & Kirkland, C. L. (2026). Hierarchically structured data can undermine machine learning results in geoscience. Earth and Planetary Science Letters, 692, Article 120250. https://doi.org/10.1016/j.epsl.2026.120250

## Identifying and grouping product records

### 7. Multimodal product matching

**Why it matters:** Product names and descriptions can be incomplete or inconsistent across retailers, making it difficult to identify identical products.

**Method:**

- Used existing product identifiers (SKU, MPN) to establish initial product groups in the underlying WDC dataset.
- Applied heuristics, machine learning, and manual review to refine matches.
- Extended product matching using text embeddings and image features.
- Compared product pairs to predict whether they represented the same item.

**Application to CatalogIQ:** We could start with identifiers such as `SKU` and `UPC`, then use `ProductName` and `ProductDescription` similarity to identify additional matches.

However, the paper focuses on identical products, not necessarily broader product families. It also does not propose a group-aware train/test splitting procedure.

**Reference (APA 7):**

Wilke, M., & Rahm, E. (2021). Towards multi-modal entity resolution for product matching. In Proceedings of the 32nd GI-Workshop on Foundations of Databases. CEUR Workshop Proceedings, Vol. 3075. https://ceur-ws.org/Vol-3075/paper10.pdf

### 8. Matching product offers across retailers

**Why it matters:** Product listings from different retailers often describe the same product differently. Names can be inconsistent, identifiers incorrect, and descriptions incomplete.

**Method:**

- Cleaned and standardized manufacturer names.
- Extracted product codes from titles and descriptions.
- Compared products using multiple attributes and text similarity measures.
- Used machine learning to determine whether two listings represented the same product.
- Applied category-specific matching strategies.

**Application to CatalogIQ:** We could combine `ProductName`, `ProductDescription`, `UPC`, and other attributes to identify related records before splitting. Importantly, the researchers found that UPC alone was unreliable.

The paper addresses product matching, not train/test splitting.

**Reference (APA 7):**

Köpcke, H., Thor, A., Thomas, S., & Rahm, E. (2012). Tailoring entity resolution for matching product offers. In Proceedings of the 15th International Conference on Extending Database Technology (pp. 545–550). https://doi.org/10.1145/2247596.2247662

### 9. BERT-based product matching

**Why it matters:** Different retailers describe the same products differently, making exact matching difficult. Researchers wanted to identify identical products even when descriptions and attributes differed.

**Method:**

- Combined product titles, attribute values, and units into text representations.
- Used BERT to generate embeddings representing products.
- Applied cosine similarity and triplet-loss training to distinguish matching from nonmatching products.
- Restricted matching candidates to products within the same category.

**Application to CatalogIQ:** We could use embeddings from `ProductName` and other product attributes to identify related records before splitting. However, their approach identifies identical products rather than broader product families.

**Reference (APA 7):**

Tracz, J., Wójcik, P. I., Jasinska-Kobus, K., Belluzzo, R., Mroczkowski, R., & Gawlik, I. (2020). BERT-based similarity learning for product matching. In Proceedings of the Workshop on Natural Language Processing in E-Commerce (pp. 66–75). Association for Computational Linguistics. https://aclanthology.org/2020.ecomnlp-1.7/

### 10. Clustering product titles with Doc2Vec

**Why it matters:** Product titles vary across retailers, making it difficult to identify and group similar products.

**Method:**

- Used 34,000 product titles from Shopee.
- Trained Doc2Vec on 30,000 titles to convert text into numerical vectors.
- Applied agglomerative hierarchical clustering to group similar titles.
- Compared distance metrics and clustering thresholds.
- Achieved a best NMI score of 0.96 using Doc2Vec (DBOW).

**Application to CatalogIQ:** We could convert `ProductName` into embeddings and cluster similar products, assigning each cluster a group ID before splitting.

However, similarity does not guarantee that products are identical or belong to the same family. We would need to validate the clusters. Their 88/12 training/test split was also performed before clustering.

**Reference (APA 7):**

Pranoto, Y. M., Handayani, A. N., Herwanto, H. W., & Kristian, Y. (2024). Optimizing product matching in e-commerce with Doc2Vec: Leveraging hierarchical clustering parameters based on product titles. ECTI Transactions on Computer and Information Technology, 18(3), 396–405. https://doi.org/10.37936/ecti-cit.2024183.256164
