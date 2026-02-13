# Bank Marketing Subscription Prediction

## Problem Statement

The objective of this project is to build machine learning models that
can predict whether a customer will subscribe to a term deposit based on
marketing campaign data. Accurate prediction can help banking
institutions focus on potential customers, improve campaign efficiency,
and reduce operational costs.

------------------------------------------------------------------------

## Dataset Description

**Dataset:** Bank Marketing Data Set

The dataset contains marketing campaign data from a Portuguese banking
institution. The goal is to predict whether a client will subscribe to a
term deposit.

**Key Characteristics:**

-   **Instances:** 45,211\
-   **Features:** 16 input features + 1 target variable\
-   **Target Variable:** `y` (Yes/No subscription)

**Feature Examples:**

-   Demographic information (age, job, marital status, education)\
-   Financial attributes (balance, housing loan, personal loan)\
-   Campaign-related details (contact type, duration, previous outcomes)

The dataset exceeds the minimum assignment requirements for both feature
size and instance count, making it suitable for robust model evaluation.

------------------------------------------------------------------------

## Models Used

The following machine learning models were trained and evaluated:

-   Logistic Regression\
-   Decision Tree\
-   K-Nearest Neighbors (KNN)\
-   Naive Bayes\
-   Random Forest\
-   XGBoost

------------------------------------------------------------------------

## Model Comparison Table

## Model Comparison Table

| ML Model            | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
|--------------------|----------|--------|-----------|--------|----------|--------|
| Logistic Regression| 0.8910   | 0.8716 | 0.5940    | 0.2157 | 0.3165   | 0.3126 |
| Decision Tree      | 0.9751   | 0.9403 | 0.8925    | 0.8949 | 0.8937   | 0.8796 |
| KNN                | 0.9140   | 0.9287 | 0.7221    | 0.4299 | 0.5390   | 0.5152 |
| Naive Bayes        | 0.8373   | 0.8163 | 0.3515    | 0.4628 | 0.3996   | 0.3114 |
| Random Forest      | 0.9804   | 0.9925 | 0.9494    | 0.8790 | 0.9128   | 0.9026 |
| XGBoost            | 0.9476   | 0.9724 | 0.8359    | 0.6867 | 0.7540   | 0.7294 |


------------------------------------------------------------------------

## Observations on Model Performance

  | ML Model            | Observation |
|--------------------|-------------|
| Logistic Regression | Provides a good baseline accuracy but struggles with recall, indicating difficulty in identifying positive subscription cases. |
| Decision Tree       | Performs very well with balanced precision and recall, showing strong capability in capturing decision boundaries within the dataset. |
| KNN                 | Achieves moderate performance but lower recall suggests it may miss several potential subscribers. Performance is sensitive to feature scaling and neighbor selection. |
| Naive Bayes         | Shows comparatively lower accuracy and precision, likely due to the independence assumption between features which may not hold true for this dataset. |
| Random Forest       | Delivers the best overall performance with the highest accuracy, AUC, F1 score, and MCC. Demonstrates strong generalization and robustness against overfitting. |
| XGBoost             | Produces strong results with high AUC and accuracy, though slightly lower recall than Random Forest. Remains a powerful model for structured data. |


------------------------------------------------------------------------

## Screenshots
![alt text](<Screenshot 2026-02-13 161118.jpg>)
![alt text](<Screenshot 2026-02-13 161342.jpg>)
![alt text](<Screenshot 2026-02-13 174839.jpg>)




------------------------------------------------------------------------

## Conclusion

Among all the evaluated models, **Random Forest** achieved the best
overall performance across most evaluation metrics. Its ability to
handle feature interactions and reduce variance makes it the most
suitable model for this prediction task.

XGBoost and Decision Tree also demonstrated strong performance and can
be considered reliable alternatives depending on deployment constraints
such as training time and interpretability.

This project highlights how machine learning can assist financial
institutions in making data-driven marketing decisions.
