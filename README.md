# Cisplatin Drug Response Prediction using XGBoost

## Overview

This project uses **XGBoost with GPU acceleration** and **Optuna hyperparameter optimization** to predict whether cisplatin (a chemotherapy drug) will be effective based on gene expression features. The model classifies cell lines as either "Useful" (sensitive to treatment) or "Not Useful" (resistant to treatment).

---

## Data Description

### Dataset
- **Source**: `data/processed/cisplatin_final.parquet`
- **Samples**: 742 cell lines
- **Features**: 41,145 gene expression values
- **Target**: `LN_IC50` (natural log of IC50 concentration)

### Key Columns
| Column | Description |
|--------|-------------|
| `CELL_LINE_NAME` | Cell line identifier |
| `LN_IC50` | Log-transformed IC50 value (drug sensitivity measure) |
| `DRUG_NAME` | Drug identifier (Cisplatin) |
| Gene columns (A1BG, A1CF, etc.) | Gene expression values |

---

## Data Manipulation Pipeline

### 1. Data Loading
```python
import pandas as pd
cisplatin_final = pd.read_parquet("data/processed/cisplatin_final.parquet")
```

### 2. Binary Target Creation
Classification threshold based on **median LC50**:
- `LC50 < median` → **Useful** (class 1, more sensitive)
- `LC50 >= median` → **Not Useful** (class 0, less sensitive)

```python
median_lc50 = cisplatin_final['LN_IC50'].median()  # ~3.07
cisplatin_final['useful'] = (cisplatin_final['LN_IC50'] < median_lc50).astype(int)
```

### 3. Feature Preparation
- Excludes metadata columns (CELL_LINE_NAME, DRUG_NAME, LN_IC50, etc.)
- Converts to `float32` for GPU efficiency
- Missing values filled with `0`

### 4. Train/Test Split
- **80/20 split** with stratification
- Training: 593 samples
- Test: 149 samples

---

## XGBoost Model Configuration

### Baseline Model
```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    tree_method='hist',      # Histogram-based algorithm
    device='cuda',           # GPU acceleration
    eval_metric='logloss'
)
```

---

## Hyperparameter Optimization

### Optuna Tuning

Using **Optuna** with **TPE Sampler** and **Median Pruner** for Bayesian hyperparameter optimization with 3-fold stratified cross-validation.

```python
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'tree_method': 'hist',
        'device': 'cuda',
        'early_stopping_rounds': 10
    }
    # 3-fold CV with pruning
    ...
    return mean_auc

study = optuna.create_study(
    direction='maximize', 
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1)
)
study.optimize(objective, n_trials=20)
```

#### Optuna Search Space
| Parameter | Range | Type |
|-----------|-------|------|
| `n_estimators` | 50 - 200 | Integer |
| `max_depth` | 3 - 8 | Integer |
| `learning_rate` | 0.05 - 0.3 | Log Float |
| `subsample` | 0.7 - 1.0 | Float |
| `colsample_bytree` | 0.7 - 1.0 | Float |
| `min_child_weight` | 1 - 7 | Integer |

#### Optuna Best Trial Results
| Metric | Value |
|--------|-------|
| **Trial #** | 7 |
| **ROC-AUC** | 0.8157 |
| **R² Score** | 0.2991 |

**Best Parameters from Optuna:**
```python
{
    'n_estimators': 55,
    'max_depth': 8,
    'learning_rate': 0.0795,
    'subsample': 0.8988,
    'colsample_bytree': 0.7935,
    'min_child_weight': 4
}
```

---

### Manual Hyperparameter Trials

Additionally, 8 predefined configurations were tested:

| Configuration | Accuracy | R² | AUC |
|---------------|----------|-----|-----|
| **Balanced** | **0.7785** | **0.2863** | **0.8263** |
| More Trees + Low LR | 0.7651 | 0.2784 | 0.8168 |
| High Trees + Medium Depth | 0.7584 | 0.2747 | 0.8171 |
| Shallow + Regularized | 0.7517 | 0.2621 | 0.8187 |
| Aggressive Regularization | 0.7450 | 0.2498 | 0.8088 |
| Baseline | 0.7651 | 0.2377 | 0.8092 |
| Deeper Trees | 0.7383 | 0.2344 | 0.8083 |
| Wide + Shallow | 0.7450 | 0.2266 | 0.8119 |

#### Best Manual Configuration: "Balanced"
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

## Model Performance Summary

### Baseline Model
| Metric | Score |
|--------|-------|
| **Accuracy** | 0.7651 |
| **R² Score** | 0.2377 |
| **ROC-AUC** | 0.8092 |

### Optimized Model (Best - "Balanced")
| Metric | Score |
|--------|-------|
| **Accuracy** | 0.7785 |
| **R² Score** | 0.2863 |
| **ROC-AUC** | 0.8263 |

### Classification Report (Baseline)
```
              precision    recall  f1-score   support
  Not Useful       0.76      0.77      0.77        75
      Useful       0.77      0.76      0.76        74
    accuracy                           0.77       149
```

---

## Top Predictive Features

| Rank | Gene | Importance |
|------|------|------------|
| 1 | SEPTIN6 | 0.0326 |
| 2 | PPIC | 0.0204 |
| 3 | PCDH1 | 0.0133 |
| 4 | ADAMTS15 | 0.0085 |
| 5 | ARHGAP9 | 0.0084 |
| 6 | TAF4B | 0.0077 |
| 7 | CRYBG2 | 0.0076 |
| 8 | GJB1 | 0.0076 |
| 9 | KDELR3 | 0.0070 |
| 10 | ABHD5 | 0.0070 |

---

## Dependencies

```
pandas
numpy
xgboost
scikit-learn
matplotlib
seaborn
optuna
tqdm
```

---

## Usage

```python
# Load and train optimized model
from xgboost import XGBClassifier

best_model = XGBClassifier(
    n_estimators=250,
    max_depth=7,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    tree_method='hist',
    device='cuda',
    random_state=42
)
best_model.fit(X_train, y_train)

# Make predictions
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
```

---

## Notes

- Model uses **median-based threshold** for balanced class distribution (50/50 split)
- **GPU acceleration** via CUDA significantly speeds up training on large feature sets (41K+ features)
- **Optuna** with TPE sampler and median pruner enables efficient Bayesian hyperparameter search
- Feature importance analysis identifies key genes associated with cisplatin sensitivity
- The "Balanced" configuration achieved the best overall performance with **AUC = 0.8263**
