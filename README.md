# Iris Dataset Classification with PySpark MLlib

## Overview of the Project
This project fulfills Assignment 1 for STQD6324 Data Management. It demonstrates an end-to-end machine learning workflow using PySpark MLlib to classify the Iris dataset into three distinct flower species based on their physical measurements.

## Description of Dataset and Methodology
* **Dataset**: The Iris dataset is fetched directly from the UC Irvine Machine Learning Repository. It contains 150 instances, each with 4 predictive attributes (sepal length, sepal width, petal length, petal width) and a categorical class label (Iris Setosa, Iris Versicolour, Iris Virginica).
* **Methodology**: 
  1. **Preprocessing**: Handled via Spark MLlib's `StringIndexer` (to convert labels to numerical classes) and `VectorAssembler` (to combine features).
  2. **Data Splitting**: 80% Training, 20% Testing.
  3. **Modeling**: Three algorithms were implemented: Logistic Regression, Decision Tree, and Random Forest.
  4. **Tuning**: 3-fold Cross-Validation alongside Grid Search was utilized to optimize hyperparameters (`regParam` for LR, `maxDepth` for Trees).
  5. **Evaluation**: Evaluated using Accuracy, Precision, Recall, and F1-Score using `MulticlassClassificationEvaluator`.

## Methodological Justification & Critical Analysis

While the Iris dataset is widely recognized for its clean, low-dimensional structure (150 instances, 4 features), the methodological choices in this workflow were deliberately selected to demonstrate a robust, scalable approach to machine learning classification.

**1. Algorithm Selection Strategy**
From a pure computational efficiency standpoint, highly complex ensemble methods are not strictly necessary for a dataset that is largely linearly separable; lighter models like K-Nearest Neighbors (KNN) or simple Support Vector Machines (SVM) could easily achieve similar baseline accuracy. However, the three selected models represent a deliberate, educational progression of algorithmic complexity:
* **Logistic Regression** establishes a strong, highly interpretable linear baseline.
* **Decision Trees** introduce non-linear, hierarchical logic and feature thresholding.
* **Random Forest** applies advanced ensemble techniques (bagging) to directly address and mitigate the high-variance and overfitting vulnerabilities inherent to standalone decision trees. 

**2. Hyperparameter Optimization Rationale**
In large-scale enterprise environments, utilizing an exhaustive Grid Search is often computationally prohibitive, forcing data scientists to rely on randomized search or Bayesian optimization. However, the compact dimensionality of the Iris dataset presents a unique opportunity. It allows us to fully leverage PySpark MLlib's exhaustive `ParamGridBuilder` to pinpoint the absolute mathematical optimum for our hyperparameters without latency penalties. 

Coupling this exhaustive search with 3-fold Cross-Validation was a critical necessity, not an option. Given the small size of the 20% holdout set (~32 rows), cross-validation ensures that our near-perfect evaluation metrics are structurally sound and robust against sampling bias, rather than mere statistical artifacts of a "lucky" train-test split.

## Summary of Results and Key Findings
All models achieved near-perfect accuracy due to the clean and simple nature of the Iris dataset. However, **Random Forest** was selected as the superior model theoretically due to its ensemble nature, mitigating the overfitting risks associated with standalone decision trees while easily capturing non-linear boundaries.

## Instructions to Reproduce the Analysis
1. Ensure you have a working Python environment with `pyspark` and `pandas` installed.
   ```bash
   pip install pyspark pandas
