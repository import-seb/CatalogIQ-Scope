# CatalogIQ-Scope

CatalogIQ is a capstone project focused on building a reproducible decision-support system for product classification.

Using the labelled training data provided by the Product Owner (PO), we will train and compare classification models, then use the best-performing approach to classify products in the unlabelled target dataset.

The system will also:

* Compare multiple modeling approaches
* Validate classification relationships
* Show model confidence
* Provide supporting product evidence
* Flag uncertain classifications for human review

Model-generated classifications should be treated as **recommendations**, not final decisions, until they pass validation or are confirmed by a reviewer.

---

## Getting Started

### 1. Download the Provided Data

Download the ZIP files provided by the PO.

Copy both ZIP files into:

```text
data/provided/
```

The provided datasets should **not be committed to GitHub until we receive permission to redistribute them**.

The `data/provided/` directory has already been added to `.gitignore`.

---

## 2. Create the Conda Environment

From the project root, create a Conda environment:

```bash
conda create -n catalogiq python=3.12
```

Activate it:

```bash
conda activate catalogiq
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Project Principle

Our goal is not simply to generate a category for every product.

The prototype should make it clear:

> **What did the model classify this product as, how confident is it, and what evidence supports that recommendation?**

Low-confidence or questionable classifications should be surfaced for human review instead of being automatically accepted.
