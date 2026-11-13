# StreamLoop Customer Churn Prediction – Model Optimization Report

## 1. Project Objective

The objective of this project was to develop and optimize a machine learning model capable of identifying customers who are likely to leave the StreamLoop subscription service.

The main goal was to improve the detection of customers belonging to the positive class (`Churn = Yes`). Identifying these customers can help the business take proactive retention measures and reduce customer loss.

For this reason, **Recall was selected as the primary evaluation metric**. In this context, a false negative occurs when the model predicts that a customer will stay, even though the customer belongs to the churn class. Missing these customers could mean losing an opportunity to intervene. False positives, on the other hand, may lead to unnecessary retention actions, but these can be considered an acceptable trade-off depending on their cost.

The project therefore prioritizes identifying as many churn cases as possible rather than maximizing overall accuracy.

## 2. Dataset and Preprocessing

The IBM Telco Customer Churn dataset was used for this project. It contains customer information, subscription details, service usage, billing information, and the target variable `Churn`.

The dataset initially contained 7,043 records and 21 columns.

The following preparation steps were performed:

- Column names were cleaned by removing unnecessary spaces.
- The `customerID` column was removed because it is an identifier and does not provide meaningful predictive information.
- The `TotalCharges` column was converted to numeric format. Eleven empty values were converted to missing values (`NaN`).
- The target variable was defined as `Churn`, with `Yes` representing customers who churned and `No` representing customers who did not churn.

The data was divided into training and testing sets before preprocessing:

| Dataset | Number of records | Percentage |
| :--- | ---: | ---: |
| Training set | 5,634 | 80% |
| Test set | 1,409 | 20% |
| Total | 7,043 | 100% |

A stratified split was used with `test_size=0.2`, `random_state=42`, and `stratify=y`. Stratification preserves the proportion of the target classes in both sets, reducing the risk of an unrepresentative split.

### 2.1. Preprocessing Pipeline

Preprocessing was implemented inside a Scikit-learn `ColumnTransformer` and integrated into a machine learning `Pipeline`.

This approach ensures that preprocessing operations are fitted using only the training data during cross-validation, preventing information leakage from validation folds or the test set.

The following transformations were applied:

| Feature type | Transformation | Reason |
| :--- | :--- | :--- |
| Numerical features | Median imputation | Replaces missing values using a robust measure that is less sensitive to extreme values than the mean. |
| Categorical features | Most frequent imputation | Replaces missing values with the most common category. |
| Categorical features | One-Hot Encoding | Converts categorical variables into numerical features that can be processed by the classifier. |

The numerical features were:

- `SeniorCitizen`
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

The remaining 15 features were treated as categorical variables.

`OneHotEncoder(handle_unknown="ignore")` was used to handle categories that might appear in validation or test data but were not present during fitting.

Keeping preprocessing inside the pipeline ensures that the same transformations are consistently applied during baseline training, cross-validation, hyperparameter optimization, and final evaluation.

## 3. Model Selection: Why Random Forest?

Random Forest was selected as the classification algorithm because customer churn can be influenced by several factors that may interact in complex ways. These relationships are not necessarily linear, so a model capable of capturing nonlinear patterns and interactions is appropriate for this problem.

Before selecting the algorithm, three commonly used classification approaches were considered: Logistic Regression, Decision Tree, and Random Forest.

The following table compares their main characteristics in the context of customer churn prediction:

| Algorithm | Advantages | Limitations | Suitability for this project |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Simple, computationally efficient, and easy to interpret. Provides a useful baseline for binary classification. | Models linear relationships between features and the log-odds of the outcome unless additional transformations or interactions are introduced. | Useful as a baseline, but potentially less flexible when churn depends on complex feature interactions. |
| **Decision Tree** | Easy to interpret and capable of capturing nonlinear relationships and feature interactions. Requires little feature scaling. | A single tree can be sensitive to changes in training data and may overfit if its complexity is not controlled. | Suitable for capturing nonlinear patterns, but its predictions may be less stable than those of an ensemble. |
| **Random Forest** | Combines multiple decision trees, captures nonlinear relationships and feature interactions, and generally reduces the variance associated with individual trees. Does not require feature scaling. | Less interpretable than a single decision tree and can require more computational resources. | A suitable choice because it offers flexibility and more robust predictions while handling complex relationships between customer characteristics. |
| **Gradient Boosting** | Sequentially learns from previous errors and can capture complex patterns with strong predictive performance. | More sensitive to hyperparameter tuning and may overfit if model complexity is not controlled. | A relevant alternative for tabular classification that could have been evaluated experimentally, although it was not trained or compared in this project. |

**→ Random Forest** was therefore chosen as a practical balance between predictive flexibility and model stability.

```
Random Forest was selected as the primary algorithm because it offers a practical balance between predictive capability, training efficiency, and ease of optimization. Its independent tree structure allows parallel training, while its ensemble approach helps reduce variance and provides robust performance on tabular datasets. It also offers relatively straightforward hyperparameter configuration and built-in class weighting options, making it particularly suitable for this project, where the main objective was to prioritize churn detection through Recall optimization.

Although Gradient Boosting is another strong candidate for tabular classification, its sequential training process and greater sensitivity to hyperparameter configuration can make optimization more demanding. Random Forest was therefore considered a more practical starting point for this project, providing a robust and manageable approach without requiring a more complex optimization strategy.
```

## 4. Baseline Model

A baseline model was trained before performing hyperparameter optimization. This established a reference point against which the tuned model could later be evaluated.

The selected algorithm was `RandomForestClassifier` using its default classifier hyperparameters, with `random_state=42` to ensure reproducibility.

The preprocessing steps and classifier were combined into a single Scikit-learn pipeline.

The baseline model was evaluated on the test set using the following metrics:

| Metric | Baseline result |
| :--- | ---: |
| Accuracy | 0.7779 |
| Precision | 0.6034 |
| Recall | 0.4759 |
| F1-score | 0.5321 |

The confusion matrix was:

| Actual / Predicted | No | Yes |
| :--- | ---: | ---: |
| No | 918 | 117 |
| Yes | 196 | 178 |

The baseline model correctly identified 178 customers belonging to the churn class but failed to identify 196 actual churn cases.

Its Recall of 47.59% indicated that the model was missing more than half of the positive cases. This highlighted the need to optimize the model with a stronger focus on identifying customers who churn.

## 5. Hyperparameter Optimization Strategy

Hyperparameter optimization was performed to improve the baseline model's ability to identify churn cases.

Since the objective was to maximize Recall, this metric was used as the primary scoring criterion throughout the optimization process.

A custom scorer was defined using `recall_score` with `pos_label="Yes"` to ensure that the positive class corresponded specifically to customers who churned.

A `StratifiedKFold` cross-validation strategy was used with the following configuration:

| Parameter | Value |
| :--- | :--- |
| Number of folds | 5 |
| Shuffle | True |
| Random state | 42 |
| Scoring metric | Recall (`Yes`) |

Stratified cross-validation was selected to preserve the proportion of churn and non-churn customers across the folds. Using five folds provides multiple performance estimates, making it possible to evaluate whether the model performs consistently across different subsets of the training data.

The test set was kept separate from the optimization process and was not used to select hyperparameters.

### 5.1. RandomizedSearchCV: Broad Hyperparameter Search

The first optimization stage used `RandomizedSearchCV`.

This method was selected because it explores a broad hyperparameter space by sampling a predefined number of combinations. It allows several configurations to be investigated without evaluating every possible combination.

The search evaluated 20 candidate configurations using five-fold cross-validation, resulting in 100 total CV fits.

The search focused on the following Random Forest hyperparameters:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `class_weight`

The `class_weight` parameter was included because the dataset contains an imbalance between churn and non-churn customers. Using balanced class weights allows the model to assign greater importance to the minority class during training.

The best configuration obtained was:

| Hyperparameter | Best value |
| :--- | :--- |
| `n_estimators` | 500 |
| `max_depth` | 5 |
| `min_samples_split` | 10 |
| `min_samples_leaf` | 2 |
| `max_features` | `log2` |
| `class_weight` | `balanced_subsample` |

The best mean cross-validation Recall was:

**0.79799 (79.80%)**

This result represented a substantial improvement over the baseline model's Recall of 47.59%.

The results of this first search were then used to define a more focused hyperparameter space for the next optimization stage.

### 5.2. GridSearchCV: Refined Hyperparameter Search

After reviewing the RandomizedSearchCV results, a more focused search space was defined using `GridSearchCV`.

GridSearchCV evaluates every combination within the specified parameter grid. This makes it suitable for systematically examining a refined set of configurations identified during the initial search.

The refined search evaluated 64 candidate configurations using five-fold cross-validation, resulting in 320 total CV fits.

The best configuration obtained was:

| Hyperparameter | Best value |
| :--- | :--- |
| `n_estimators` | 300 |
| `max_depth` | 5 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 2 |
| `max_features` | `sqrt` |
| `class_weight` | `balanced` |

The best mean cross-validation Recall was:

**0.80803 (80.80%)**

Compared with RandomizedSearchCV, GridSearchCV improved the mean cross-validation Recall from 79.80% to 80.80%.

The refined search required more total CV fits than the initial randomized search. Its purpose was not to reduce the number of fits, but to examine every combination within a narrower parameter space informed by the initial results.

## 6. Cross-Validation Results and Model Stability

The `cv_results_` attribute was reviewed to examine not only the best mean Recall but also the variability across folds.

This is important because a model with a high average score may still perform inconsistently across different data subsets.

The best GridSearchCV configuration achieved the following results:

| Indicator | Result |
| :--- | ---: |
| Mean Recall | 0.808027 |
| Standard deviation | 0.031841 |
| Fold 1 | 0.806020 |
| Fold 2 | 0.769231 |
| Fold 3 | 0.816054 |
| Fold 4 | 0.862876 |
| Fold 5 | 0.785953 |

The standard deviation of approximately 0.0318 indicates that the Recall scores varied moderately across the five folds.

The top-performing configurations were also examined:

| Candidate | Mean Recall | Standard deviation |
| :--- | ---: | ---: |
| Candidate 1 | 0.808027 | 0.031841 |
| Candidate 5 | 0.807358 | 0.030550 |
| Candidate 2 | 0.807358 | 0.034795 |
| Candidate 4 | 0.806689 | 0.035103 |
| Candidate 7 | 0.806689 | 0.034524 |

Candidate 1 was selected because it achieved the highest mean Recall across the five folds.

Candidate 5 had a slightly lower standard deviation, indicating marginally more consistent results, but its mean Recall was also slightly lower. The difference between the two configurations was small.

The selection therefore prioritized the highest mean Recall, in accordance with the project's main objective, while also considering the variability across folds.

## 7. Final Model Selection and Training

The final model was obtained from:

`grid_search.best_estimator_`

The use of `best_estimator_` ensures that the selected configuration corresponds to the best-performing candidate identified during GridSearchCV.

The search was configured with `refit=True`, meaning that Scikit-learn automatically retrained the selected pipeline using the entire training set after completing cross-validation.

The final model therefore includes both the preprocessing transformations and the optimized Random Forest classifier.

The test set remained untouched during hyperparameter selection and was used only for the baseline evaluation and the final model evaluation.

## 8. Final Model Evaluation

After optimization, the final model was evaluated on the same test set used for the baseline evaluation.

This allows a direct comparison between the baseline and optimized models using the same evaluation data and metrics.

The final results were:

| Metric | Baseline Model | Tuned Model |
| :--- | ---: | ---: |
| Accuracy | 0.7779 | 0.7367 |
| Precision | 0.6034 | 0.5026 |
| Recall | 0.4759 | 0.7861 |
| F1-score | 0.5321 | 0.6131 |

The tuned model achieved a Recall of 78.61%, compared with 47.59% for the baseline model.

This represents an improvement of approximately **31.02 percentage points** in Recall.

However, Accuracy decreased from 77.79% to 73.67%, and Precision decreased from 60.34% to 50.26%.

The F1-score increased from 53.21% to 61.31%, indicating an improvement in the balance between Precision and Recall.

These results demonstrate the trade-off introduced by prioritizing Recall: the tuned model identifies a substantially larger proportion of churn cases, although it also generates more false positives.

## 9. Business Interpretation

The optimized model is better aligned with StreamLoop's objective of identifying customers who churn.

Its Recall of 78.61% means that it correctly identifies approximately 79 out of every 100 customers belonging to the churn class in the test set.

This improvement could help the business identify more customers for potential retention campaigns and reduce the number of churn cases that go undetected.

However, the lower Precision means that a greater proportion of customers flagged by the model will not belong to the actual churn class. Consequently, StreamLoop may need to contact more customers who would have stayed anyway.

Whether this trade-off is financially beneficial depends on the cost of retention campaigns, the value of each customer, and the financial impact of losing a customer. These business costs were not quantified in this project.

The model should therefore be considered a decision-support tool rather than a guarantee that a customer will leave the service.

## 10. Conclusion

The project successfully developed and optimized a Random Forest classification pipeline for customer churn prediction.

The model selection was based on the suitability of Random Forest for capturing nonlinear relationships and feature interactions, as well as its ability to reduce the instability associated with individual decision trees through an ensemble approach.

The preprocessing pipeline was integrated with the classifier to ensure consistent transformations and prevent data leakage during cross-validation.

The optimization process was conducted in two stages. RandomizedSearchCV was first used to explore a broad range of hyperparameter configurations, followed by GridSearchCV to examine a refined parameter space.

The best configuration achieved a mean cross-validation Recall of 80.80%. When evaluated on the independent test set, the optimized model achieved a Recall of 78.61%, compared with 47.59% for the baseline model.

Although Accuracy and Precision decreased, Recall and F1-score improved. This trade-off is consistent with the project's primary objective of identifying as many churn cases as possible.

Overall, the optimized model represents an improvement over the baseline for the specific business objective established in this project. Further work could include comparing Random Forest against other algorithms through empirical experiments, evaluating alternative classification thresholds, and incorporating business costs to determine the most appropriate balance between false positives and false negatives.