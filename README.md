# StreamLoop – Customer Churn Prediction

A machine learning project focused on identifying customers belonging to the churn class (`Yes`) using the IBM Telco Customer Churn dataset. The main objective is to improve the detection of positive cases by prioritizing **Recall** during hyperparameter optimization.

## Project objective

This project develops and evaluates a Random Forest classifier to identify customers who have churned. Recall for the `Yes` class is the primary optimization metric because the project prioritizes reducing false negatives: customers who belong to the churn class but are incorrectly classified as `No`.

This prioritization involves a trade-off: increasing Recall can also increase false positives and reduce Precision.

## Dataset

The project uses the [IBM Telco Customer Churn dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv).

- Initial size: 7,043 records and 21 columns.
- Target variable: `Churn` (`Yes` / `No`).
- `customerID` is removed because it is a unique identifier rather than a useful predictive feature.
- `TotalCharges` is converted to numeric; 11 blank values become missing values and are handled during preprocessing.

## Methodology

### Data preparation

The data is split into training and test sets before preprocessing: 80% for training and 20% for testing, using `random_state=42` and stratification by the target variable.

Preprocessing is implemented inside a Scikit-learn `Pipeline` and `ColumnTransformer` to prevent data leakage during cross-validation:

- Numerical features: median imputation.
- Categorical features: most-frequent-value imputation followed by One-Hot Encoding.
- `OneHotEncoder(handle_unknown="ignore")` handles categories not seen during fitting.

### Algorithm selection

Random Forest was selected because it can model non-linear relationships and interactions between features. By combining multiple decision trees, it can reduce the variability associated with relying on a single tree. It also does not require numerical feature scaling.

Logistic Regression and a single Decision Tree were considered conceptually as alternatives. This project does not include an experimental benchmark in which all three algorithms are trained and compared; the comparison is based on their methodological characteristics.

| Algorithm | Advantages | Limitations for this problem |
|---|---|---|
| Logistic Regression | Simple, fast, and interpretable; models relationships between features and class probability. | Its basic formulation is linear and may not capture complex relationships without additional transformations. |
| Decision Tree | Interpretable and able to capture non-linear relationships. | A single tree can be unstable and may overfit if its complexity is not controlled. |
| Random Forest | Captures non-linear relationships and feature interactions; combining trees helps reduce variability compared with a single tree. | Less interpretable than one decision tree and requires tuning several hyperparameters. |

### Baseline model

A baseline `RandomForestClassifier` was trained using its default classifier hyperparameters, with `random_state=42`.

### Hyperparameter optimization

The optimization process uses:

- Primary scoring metric: Recall for `Churn="Yes"`.
- Cross-validation: 5-fold `StratifiedKFold`, with shuffling and `random_state=42`.
- `RandomizedSearchCV`: 20 sampled configurations, resulting in 100 cross-validation fits.
- `GridSearchCV`: 64 configurations in a refined search space, resulting in 320 cross-validation fits.
- `refit=True`: the best estimator is refitted on the full training set.

Cross-validation results are reviewed using both mean Recall and standard deviation across folds. The final configuration is selected based on the highest mean Recall. The test set remains separate from hyperparameter tuning and is used for baseline and final model evaluation.

## Results

### Baseline vs. optimized model

| Metric | Baseline | Optimized |
|---|---:|---:|
| Accuracy | 0.7779 | 0.7367 |
| Precision | 0.6034 | 0.5026 |
| Recall | 0.4759 | 0.7861 |
| F1-score | 0.5321 | 0.6131 |

The optimized model improves test-set Recall from **0.4759 to 0.7861**, an increase of approximately **31.02 percentage points**. F1-score also increases, while Accuracy and Precision decrease. This reflects the project's chosen trade-off: the optimized model identifies a substantially larger proportion of customers in the positive class, at the cost of generating more false positives.

### Best hyperparameters

The best configuration found by `GridSearchCV` is:

```python
{
    "classifier__class_weight": "balanced",
    "classifier__max_depth": 5,
    "classifier__max_features": "sqrt",
    "classifier__min_samples_leaf": 2,
    "classifier__min_samples_split": 2,
    "classifier__n_estimators": 300
}
```

Best mean cross-validation Recall: **0.8080**.

## Running the project

1. Clone or download this repository.
2. Open `streamloop_churn_tuning.ipynb` in Jupyter Notebook, JupyterLab, or a compatible environment such as GitHub Codespaces.
3. Install the required packages if they are not already available:

   ```bash
   pip install pandas scikit-learn jupyter
   ```

4. Run the notebook cells from top to bottom.

The notebook uses a fixed random state for reproducibility. The reported results were reproduced after restarting the kernel and running all cells in order.

### Environment used for the recorded run

- Python 3.14.2
- pandas 3.1.1
- scikit-learn 1.9.1

Other compatible versions may also work, but could produce small differences in results.

## Limitations and interpretation

- The target variable represents the churn label in the dataset. The project does not define a future prediction horizon, so the model should not be interpreted as predicting exactly when a customer will leave.
- The model is a classification aid, not a guarantee about an individual customer's behavior.
- Higher Recall comes with lower Precision. The business impact depends on the relative costs of false positives and false negatives, which have not been quantified in this project.
- The comparison between Random Forest, Logistic Regression, and Decision Tree is conceptual; only Random Forest was tuned and evaluated in this notebook.
- Results are based on the provided dataset and should be validated with current business data before operational use.

## Technologies

- Python
- pandas
- Scikit-learn
- Jupyter Notebook

## Project file

- `streamloop_churn_tuning.ipynb` – data preparation, baseline training, hyperparameter optimization, cross-validation analysis, and final evaluation.
- `tuning_report.md` y `tuning_report.es.md` - project documentation archive

---
**Developed by Wendy Sanchez**