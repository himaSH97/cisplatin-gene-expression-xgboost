# Predicting Cisplatin Response from Gene Expression Profiles

Chemotherapy resistance remains a major obstacle in cancer treatment. This project investigates whether transcriptomic data can predict sensitivity to cisplatin, one of the most commonly prescribed platinum-based chemotherapeutics. Using gradient boosting on RNA-seq profiles from 742 cancer cell lines, we achieve an **AUC of 0.826** in distinguishing sensitive from resistant lines.

---

## Background and Motivation

Real-world biomedical datasets rarely arrive analysis-ready. This project began with a practical challenge: integrating multiple messy data sources into a unified format suitable for machine learning, then building a predictive model on the result.

The raw data presented several characteristics common to high-dimensional biological datasets:

- **Wide feature space**: 41,000+ gene columns
- **Identifier mismatches**: Different datasets used different cell line ID schemes
- **Memory constraints**: Naive merges exceeded available RAM
- **Data quality issues**: Duplicate columns, metadata mixed with data, missing values

Beyond preprocessing, the modeling task posed its own challenges. With a 742 × 41,145 sample-to-feature ratio, overfitting is a significant concern, and standard cross-validation becomes computationally expensive. This project documents one approach to handling such data: streaming data integration, gradient boosting with regularization, and systematic hyperparameter search.

---

## Data Sources

We integrated three publicly available datasets from the [Cell Model Passports](https://cellmodelpassports.sanger.ac.uk/downloads) repository (Sanger Institute):


| Dataset                   | Description                                | Source               |
| ------------------------- | ------------------------------------------ | -------------------- |
| **RNA-seq expression**    | TPM-normalized transcript abundances       | Cell Model Passports |
| **GDSC2 IC50**            | Half-maximal inhibitory concentrations     | Cell Model Passports |
| **Cell line annotations** | Identifier mapping (SIDM → cell line name) | Cell Model Passports |


After restricting to cell lines present in both expression and pharmacological datasets, the final cohort comprised **742 samples** with **41,145 gene features**.

---

## Preprocessing Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Expression    │     │     IC50        │     │    Mapping      │
│   (RNA-seq)     │     │  (Drug resp.)   │     │  (Cell line ID) │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ clean_expression│     │   clean_ic50    │     │  clean_mapping  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    map_cell_lines       │
                    │  (SIDM → cell names)    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ filter_common_cell_lines│
                    │   (intersection only)   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  merge_datasets_stream  │
                    │   (to Parquet parts)    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Validation & QC       │
                    │    sanity_check()       │
                    └─────────────────────────┘
```

### Expression Data Processing

The raw expression matrix required several transformations:

1. Removal of metadata rows and unnamed columns
2. Transposition to sample × gene orientation
3. Resolution of duplicate gene identifiers via suffix annotation (`_dup1`, `_dup2`)
4. Mapping of internal identifiers to standardized cell line names

### Target Variable Construction

Drug sensitivity was measured as `LN_IC50`, the natural logarithm of the half-maximal inhibitory concentration. We dichotomized this continuous measure at the median (LN_IC50 = 3.07) to define:

- **Sensitive**: LN_IC50 < median (lower concentrations required for cytotoxicity)
- **Resistant**: LN_IC50 ≥ median

This threshold yields balanced classes and aligns with the pharmacological interpretation that lower IC50 values indicate greater drug efficacy.

### Data Integration

Merging high-dimensional expression data with drug response measurements presented computational challenges. We implemented a streaming approach that processes IC50 records in chunks of 2,000 rows and writes intermediate results to compressed Parquet files, enabling the pipeline to run within memory constraints.

---

## Modeling Approach

### Algorithm Selection

We selected XGBoost (Extreme Gradient Boosting) for its established performance on high-dimensional tabular data and native support for GPU acceleration. The latter consideration was practical given the feature space dimensionality.

### Baseline Model

Initial experiments used standard hyperparameters:

```python
XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    tree_method='hist',
    device='cuda',
    random_state=42
)
```

This configuration achieved 76.5% accuracy and 0.81 AUC on the held-out test set (20% of samples, stratified split).

### Hyperparameter Optimization

We conducted systematic hyperparameter search using Optuna with TPE sampling and median pruning. The search explored:


| Parameter          | Range                |
| ------------------ | -------------------- |
| `n_estimators`     | 50–200               |
| `max_depth`        | 3–8                  |
| `learning_rate`    | 0.05–0.3 (log scale) |
| `subsample`        | 0.7–1.0              |
| `colsample_bytree` | 0.7–1.0              |
| `min_child_weight` | 1–7                  |


Each configuration was evaluated using 3-fold stratified cross-validation. Twenty trials were completed, with early stopping to terminate unpromising runs.

Additionally, we evaluated eight manually specified configurations representing different regularization strategies:


| Configuration         | Accuracy   | AUC       |
| --------------------- | ---------- | --------- |
| **Balanced**          | **77.85%** | **0.826** |
| High Trees + Low LR   | 76.51%     | 0.817     |
| Shallow + Regularized | 75.17%     | 0.819     |
| Baseline              | 76.51%     | 0.809     |
| Deep Trees            | 73.83%     | 0.808     |


The optimal configuration balanced model capacity with regularization:

```python
{
    'n_estimators': 250,
    'max_depth': 7,
    'learning_rate': 0.08,
    'subsample': 0.85,
    'colsample_bytree': 0.85
}
```

---

## Results

### Classification Performance


| Metric        | Value  |
| ------------- | ------ |
| **AUC**       | 0.826  |
| **Accuracy**  | 77.85% |
| **Precision** | 0.77   |
| **Recall**    | 0.76   |


Performance was comparable across sensitivity and resistance classes, suggesting the model does not exhibit systematic bias toward either prediction.

#### ROC Curve

ROC Curve

#### Classification Threshold Analysis

Classification Threshold

### Feature Importance Analysis

Gradient boosting models provide intrinsic feature importance scores based on split frequency and information gain. The highest-ranked genes were:


| Gene         | Importance | Biological Context                                     |
| ------------ | ---------- | ------------------------------------------------------ |
| **SEPTIN6**  | 0.033      | Cytoskeletal GTPase involved in cytokinesis            |
| **PPIC**     | 0.020      | Peptidyl-prolyl isomerase; protein folding             |
| **PCDH1**    | 0.013      | Protocadherin; cell-cell adhesion                      |
| **ADAMTS15** | 0.009      | Metalloproteinase; ECM remodeling                      |
| **ARHGAP9**  | 0.008      | Rho GTPase-activating protein; cytoskeletal regulation |


The prominence of SEPTIN6 is noteworthy given that septins regulate cell division—the process disrupted by cisplatin-induced DNA damage. Several other top features relate to cytoskeletal organization and cell adhesion, pathways implicated in drug efflux and apoptosis resistance.

Feature Importance

---

## Data Quality Assurance

The pipeline incorporates validation checks at multiple stages:

- **Dimensional verification**: Final dataset should contain 742 samples × 41,148 features
- **Duplicate detection**: No repeated (cell line, drug) combinations
- **Type checking**: All gene columns must be numeric
- **Range validation**: IC50 values within expected bounds
- **Merge integrity**: Logging of unmatched records

Failed processing runs generate diagnostic files documenting the failure point.

---

## Limitations

Several technical and methodological constraints affect this work:

1. **Extreme dimensionality**: With 41,145 features and only 742 samples, the feature-to-sample ratio (~55:1) creates high overfitting risk. No dimensionality reduction (PCA, feature selection) was applied before training.
2. **Limited hyperparameter search**: Optuna ran only 20 trials due to computational cost. A more exhaustive search or Bayesian optimization with more iterations might yield better configurations.
3. **Single train/test split**: While stratified, a single 80/20 split provides less robust performance estimates than nested cross-validation or repeated holdout.
4. **No external validation set**: Test performance is on held-out data from the same source distribution. True generalization requires testing on entirely independent datasets.
5. **Memory-constrained architecture**: The streaming merge was necessary due to RAM limits, but prevented exploring join strategies or in-memory operations that might catch edge cases.
6. **Black-box model**: XGBoost provides feature importances but not confidence intervals, prediction uncertainty, or causal interpretations. SHAP values would improve explainability but were not implemented.

---

## Future Directions

- Incorporate SHAP values for more granular feature attribution
- Extend to multi-drug prediction with compound-specific features
- Explore deep learning architectures for potential performance gains

---

## Reproducibility

### Dependencies

```
pandas, numpy, xgboost, scikit-learn, optuna, pyarrow, tqdm, matplotlib, seaborn
```

### Execution

```bash
# Run preprocessing pipeline
python main.py

# Model training and evaluation
jupyter notebook xgboost_cisplatin.ipynb
```

---

## Repository Structure

```
drg-res/
├── data/
│   ├── raw/                     # Source files
│   └── processed/               # Processed Parquet output
├── src/data/
│   ├── load_data.py             # Data loading utilities
│   ├── preprocess.py            # Cleaning functions
│   ├── merge.py                 # Streaming merge implementation
│   └── validate.py              # Validation routines
├── xgboost_cisplatin.ipynb      # Analysis notebook
├── main.py                      # Pipeline entry point
├── README.md                    # Model documentation
└── DATA_CLEANING.md             # Preprocessing documentation
```

---

*May 2026*