# StreamLoop – Churn Model Tuning Report

## 1. Objective

The objective was to improve the detection of customers likely to cancel their subscriptions.

Recall for the positive class (`Yes`) was selected as the primary evaluation metric because missing a customer who is about to churn is more costly than incorrectly identifying a customer as a potential churner.

## 2. Baseline Model

A Random Forest classifier was implemented using a preprocessing and classification pipeline.

The baseline model used default classifier hyperparameters and was evaluated on the test set.

| Metric | Baseline |
|---|---:|
| Accuracy | 0.7779 |
| Precision | 0.6034 |
| Recall | 0.4759 |
| F1-score | 0.5321 |

The baseline detected only 47.59% of customers who actually churned.

## 3. Hyperparameter Optimization

Two search strategies were applied using 5-fold Stratified Cross-Validation and Recall as the scoring metric.

### RandomizedSearchCV

- Number of parameter combinations tested: 20
- Total cross-validation fits: 100
- Best mean Recall: 0.7980

### GridSearchCV

The search space was refined based on the RandomizedSearchCV results to reduce computational cost while focusing on promising parameter values.

- Number of parameter combinations tested: 64
- Total cross-validation fits: 320
- Best mean Recall: 0.8080

## 4. Final Hyperparameters

The selected model was the best-performing GridSearchCV candidate.

| Hyperparameter | Selected value |
|---|---|
| n_estimators | 300 |
| max_depth | 5 |
| max_features | sqrt |
| min_samples_split | 2 |
| min_samples_leaf | 2 |
| class_weight | balanced |

## 5. Model Stability and Selection

The selected candidate achieved:

- Mean cross-validation Recall: 0.808027
- Standard deviation: 0.031841

A second candidate achieved a slightly lower mean Recall of 0.807358 and a standard deviation of 0.030550.

Although the second candidate showed marginally lower variability, the difference in stability was small. The first candidate was selected because it achieved the highest mean Recall, which better supports the primary business objective.

## 6. Final Test Evaluation

The tuned model was evaluated on the test set and compared with the baseline.

| Metric | Baseline | Tuned Model |
|---|---:|---:|
| Accuracy | 0.7779 | 0.7367 |
| Precision | 0.6034 | 0.5026 |
| Recall | 0.4759 | 0.7861 |
| F1-score | 0.5321 | 0.6131 |

## 7. Business Interpretation and Conclusion

The tuned model increased Recall from 47.59% to 78.61%, an improvement of 31.02 percentage points.

This means the optimized model detects a substantially larger proportion of customers who are likely to cancel their subscriptions.

Although Precision and Accuracy decreased, this trade-off is acceptable in the StreamLoop business context because failing to identify a customer who churns is more costly than contacting a customer who was not actually going to cancel.

The F1-score also improved from 0.5321 to 0.6131.

**Conclusion:** The tuned Random Forest model is preferred over the baseline because it better addresses StreamLoop's primary business objective: identifying as many potential churners as possible.