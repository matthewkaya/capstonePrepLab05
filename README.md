# Lab 05 — Predict Late Deliveries

## Project Overview

This project predicts whether a delivery order will arrive late using information available when the order is placed.

The dataset contains 3,000 synthetic orders generated with the provided data generator.

The objective is to minimize business costs associated with offering vouchers and missing late deliveries.

## Business Cost

The cost policy is:

- Flagged order: $4 voucher
- Missed late delivery: $15
- Other outcomes: $0

Total cost:

`4 × flagged orders + 15 × missed late deliveries`

## Setup

Clone the repository:

```bash
git clone https://github.com/matthewkaya/capstonePrepLab05.git
cd capstonePrepLab05
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate lab05-late-deliveries
```

Alternatively, install dependencies using pip:

```bash
python -m pip install -r requirements.txt
```

## Run

Reproduce model training and evaluation:

```bash
python -m latedeliveries.train
```

The training workflow generates model artifacts and evaluation reports.

Open the analysis notebook:

```text
notebooks/analysis.ipynb
```

The notebook contains executed outputs and visualizations.

## Check It

Run automated tests:

```bash
python -m pytest -q
```

Check code quality:

```bash
ruff check src/ tests/
ruff format --check src/ tests/
```

## Decisions

### Features

The model uses seven features available at order placement:

- `distance_km`
- `items`
- `hour`
- `courier_months`
- `weather`
- `day_type`
- `vehicle`

The following columns are excluded:

- `order_id`: identifier
- `prep_minutes`: unavailable at prediction time
- `customer_rating`: unavailable at prediction time
- `late`: prediction target

Courier experience and vehicle type are assumed to be known when the order is placed.

### Data Preparation

The dataset is split into 75% training and 25% testing, using stratification and `random_state=42`.

Numerical features use median imputation and standard scaling.

Categorical features use most-frequent imputation and one-hot encoding with unknown-category handling.

Preprocessing is included in the model pipeline to prevent data leakage.

### Model Selection

The following classifiers are compared using five-fold stratified cross-validation:

1. Dummy Classifier
2. Logistic Regression
3. Logistic Regression with balanced class weights
4. Random Forest
5. HistGradientBoosting

The comparison includes accuracy, precision, recall, F1, and ROC-AUC, with mean and standard deviation.

Logistic Regression is selected for hyperparameter tuning.

GridSearchCV evaluates eight combinations of regularization strength and class weighting.

The best cross-validation ROC-AUC is approximately 0.7801. Its improvement over the default Logistic Regression is approximately 0.0006, which is small relative to fold-to-fold variation.

### Threshold Selection

The classification threshold is optimized using out-of-fold probabilities from the training set.

Candidate thresholds range from 0.05 to 0.95 in increments of 0.05.

The selected threshold is **0.25**, minimizing the specified business cost on the training predictions.

The test set is not used for threshold selection.

## Results

The final model is evaluated on the held-out test set of 750 orders.

| Metric | Value |
|---|---:|
| Accuracy | 0.8360 |
| Precision | 0.4706 |
| Recall | 0.4103 |
| F1 | 0.4384 |
| ROC-AUC | 0.7742 |

### Confusion Matrix

| | Predicted On Time | Predicted Late |
|---|---:|---:|
| Actual On Time | 579 | 54 |
| Actual Late | 69 | 48 |

### Business Cost Comparison

| Policy | Total Cost |
|---|---:|
| Flag no orders | $1,755 |
| Flag every order | $3,000 |
| Model with threshold 0.25 | **$1,443** |

The model saves **$312 (17.8%)** compared with flagging no orders.

It also saves **$1,557 (51.9%)** compared with flagging every order.

## Generated Artifacts

- `reports/cv_results.csv`: cross-validation comparison
- `reports/threshold_results.json`: threshold optimization results
- `reports/metrics.json`: final evaluation metrics and business costs
- `models/late_orders.joblib`: saved model pipeline
- `models/model_info.json`: model metadata
- `notebooks/analysis.ipynb`: executed analysis notebook

## Tests

Automated tests cover:

- Feature selection and excluded columns
- Stratified train/test splitting
- Pipeline predictions and probabilities
- Missing values and unseen categories
- Business cost calculations
- Edge cases for threshold selection
- Model serialization and loading

The project currently contains 17 passing tests.

## Limitations

- The dataset is synthetic and may not represent real delivery operations.
- The model identifies approximately 41% of late deliveries.
- Courier experience and vehicle information must be available at prediction time.
- Business costs are simplified and may change in production.
- Performance and data drift should be monitored before wider deployment.

## Recommendation

The tuned Logistic Regression pipeline is suitable for a limited pilot using the selected 0.25 threshold.

The model reduces estimated business costs on the held-out test set, but its recall remains limited.

Before production deployment, validate performance using real operational data and monitor prediction quality, business costs, and changes in delivery patterns.