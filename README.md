# Bank Marketing Subscription Prediction


Live App:
https://stroke-ml-app-2025aa05139.streamlit.app/


## Problem Statement

The objective of this assignment is to build machine learning models that
can predict whether a customer will subscribe to a term deposit based on
marketing campaign data. Accurate prediction can help banking
institutions focus on potential customers, improve campaign efficiency,
and reduce operational costs.



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


## Models Used

The following machine learning models were trained and evaluated:

-   Logistic Regression\
-   Decision Tree\
-   K-Nearest Neighbors (KNN)\
-   Naive Bayes\
-   Random Forest\
-   XGBoost



## Model Comparison Table

| ML Model            | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
|--------------------|----------|--------|-----------|--------|----------|--------|
| Logistic Regression| 0.8910   | 0.8716 | 0.5940    | 0.2157 | 0.3165   | 0.3126 |
| Decision Tree      | 0.9751   | 0.9403 | 0.8925    | 0.8949 | 0.8937   | 0.8796 |
| KNN                | 0.9140   | 0.9287 | 0.7221    | 0.4299 | 0.5390   | 0.5152 |
| Naive Bayes        | 0.8373   | 0.8163 | 0.3515    | 0.4628 | 0.3996   | 0.3114 |
| Random Forest      | 0.9804   | 0.9925 | 0.9494    | 0.8790 | 0.9128   | 0.9026 |
| XGBoost            | 0.9476   | 0.9724 | 0.8359    | 0.6867 | 0.7540   | 0.7294 |




## Observations on Model Performance

  | ML Model            | Observation |
|--------------------|-------------|
| Logistic Regression | Provides a good baseline accuracy but struggles with recall, indicating difficulty in identifying positive subscription cases. |
| Decision Tree       | Performs very well with balanced precision and recall, showing strong capability in capturing decision boundaries within the dataset. |
| KNN                 | Achieves moderate performance but lower recall suggests it may miss several potential subscribers. Performance is sensitive to feature scaling and neighbor selection. |
| Naive Bayes         | Shows comparatively lower accuracy and precision, likely due to the independence assumption between features which may not hold true for this dataset. |
| Random Forest       | Delivers the best overall performance with the highest accuracy, AUC, F1 score, and MCC. Demonstrates strong generalization and robustness against overfitting. |
| XGBoost             | Produces strong results with high AUC and accuracy, though slightly lower recall than Random Forest. Remains a powerful model for structured data. |



## Screenshots
![alt text](<Screenshot 2026-02-13 161118.jpg>)
![alt text](<Screenshot 2026-02-13 161342.jpg>)
![alt text](<Screenshot 2026-02-13 174839.jpg>)





Conclusion

This assignment successfully implemented and evaluated multiple machine learning models to predict customer subscription behavior using marketing campaign data. The comparative analysis revealed significant differences in predictive capability, model stability, and generalization performance.

Among all models, Random Forest emerged as the top performer, achieving the highest scores across Accuracy, AUC, F1 Score, and MCC. Its ensemble methodology reduces variance, captures complex feature relationships, and enhances overall reliability — making it highly suitable for real-world deployment.

The Decision Tree model also demonstrated excellent predictive strength but may be more prone to overfitting compared to ensemble techniques. XGBoost showed outstanding discriminative power with a very high AUC score and remains a strong candidate for production environments requiring optimized performance.

Logistic Regression provided a dependable baseline; however, its lower recall suggests that linear models may struggle with the non-linear patterns present in marketing datasets. KNN delivered reasonable accuracy but may face scalability challenges as data volume grows. Naive Bayes showed comparatively weaker performance due to its assumption of feature independence.

Key Findings

Ensemble methods consistently outperformed individual algorithms.

Evaluating models using multiple metrics provides a more comprehensive understanding than relying on accuracy alone.

Strong AUC values indicate effective separation between subscribing and non-subscribing customers.

Marketing datasets often contain non-linear relationships that benefit from tree-based approaches.

Business Impact

Accurate prediction of term deposit subscriptions enables banks to:

Target high-probability customers

Optimize marketing expenditure

Reduce unnecessary outreach

Improve campaign conversion rates

Support data-driven strategic decision-making

Final Recommendation

Based on overall evaluation, Random Forest is recommended as the most suitable model due to its robustness, strong predictive capability, and consistent performance. XGBoost serves as an excellent alternative when computational complexity can be accommodated.

This assignment demonstrates the practical value of machine learning in enhancing marketing effectiveness and highlights the importance of systematic model evaluation when selecting an optimal predictive solution.
