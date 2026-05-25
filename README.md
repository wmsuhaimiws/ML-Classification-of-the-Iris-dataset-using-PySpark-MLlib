# Iris Flower Classification with Apache Spark MLlib

**STQD6324 Data Management — Assignment 1**

An end-to-end multiclass classification project on the classic **Iris** dataset, built entirely with **Apache Spark MLlib**. Three classifiers — Logistic Regression, Decision Tree, and Random Forest — are trained, tuned with cross-validation and grid search, evaluated on a held-out test set, and compared to justify a best model.

---

## 1. Project Overview

The goal is to predict the species of an iris flower (*setosa*, *versicolor*, or *virginica*) from four morphological measurements. The project demonstrates a complete, reproducible Spark MLlib workflow: data loading into a Spark DataFrame, preprocessing with `StringIndexer`/`VectorAssembler`/`StandardScaler`, pipeline construction, hyperparameter tuning with 5-fold `CrossValidator` + `ParamGridBuilder`, multi-metric evaluation (accuracy, precision, recall, F1, confusion matrix), prediction on unseen data, and a comparative analysis of the three algorithms.

Every step in the notebook is accompanied by a markdown explanation of *what* is done, *why*, and *how to interpret* the output.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository (also bundled with scikit-learn) |
| Rows | 150 |
| Features | 4 numeric (cm): `sepal_length`, `sepal_width`, `petal_length`, `petal_width` |
| Target | `species` — 3 balanced classes (50 each): setosa, versicolor, virginica |
| Missing values | None |

A clean copy, `iris.csv`, is included in the repository so the notebook runs **offline and reproducibly**. If the file is absent, the notebook automatically falls back to downloading the canonical UCI copy and normalising it.

**Key data characteristics that shape the analysis:** the dataset is small, perfectly class-balanced, and nearly linearly separable. *Setosa* is trivially separable from the other two; essentially all classification difficulty lies in the modest overlap between *versicolor* and *virginica*.

---

## 3. Methodology

1. **Spark session** — local-mode `SparkSession` with fixed shuffle partitions for deterministic, fast small-data runs.
2. **Load** — read `iris.csv` into a Spark DataFrame with an explicit schema.
3. **EDA & preprocessing** — verify no missing values and class balance; **visualise** the data (pairwise scatter matrix, per-feature box plots, correlation heatmap) to expose class separability; encode the label with `StringIndexer`; assemble features with `VectorAssembler`; add `StandardScaler` only to the Logistic Regression pipeline (tree models are scale-invariant).
4. **Split** — 70% train / 30% test, `seed=42`.
5. **Pipelines** — one `Pipeline` per model so preprocessing is fit per cross-validation fold (no data leakage).
6. **Tuning** — 5-fold `CrossValidator` with `ParamGridBuilder`, optimising **F1**:
   - **Logistic Regression:** `regParam`, `elasticNetParam`, `maxIter`
   - **Decision Tree:** `maxDepth`, `impurity`
   - **Random Forest:** `numTrees`, `maxDepth`, `featureSubsetStrategy`
7. **Evaluation** — accuracy, weighted precision, weighted recall, F1, and a confusion matrix per model on the test set.
8. **Prediction** — readable predicted-vs-actual species via `IndexToString`.
9. **Comparison** — metrics table (incl. cross-validated F1), **visual analysis** (side-by-side confusion-matrix heatmaps, CV-vs-test F1 grouped bars, Random Forest feature importances), strengths/limitations, and a justified best-model choice.

---

## 4. Results & Key Findings

Results from the executed run (70/30 split, `seed=42`):

| Model | Accuracy | Precision | Recall | Test F1 | CV F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.9783 | 0.9804 | 0.9783 | 0.9785 | **0.9591** |
| Decision Tree | 0.9783 | 0.9804 | 0.9783 | 0.9785 | 0.9281 |
| Random Forest | 0.9783 | 0.9804 | 0.9783 | 0.9785 | 0.9303 |

**Key finding — the three models tie on the test set.** All produced *identical* test metrics and *identical* confusion matrices: every model classified all setosa and all virginica test flowers correctly and misclassified the **same single versicolor** flower. With only 46 test rows and one error, the test set cannot rank the models.

- **Setosa** is perfectly separable and is never misclassified by any model; the lone error sits on the **versicolor ↔ virginica** boundary, exactly as expected.
- Because the test scores tie, the model is chosen on the **cross-validated F1** (averaged over the 5 training folds — a more reliable discriminator than a single small test split) and on parsimony:
  - **Logistic Regression is selected** — it has the clearly highest CV F1 (0.959 vs ~0.930) and is the simplest, fastest, most interpretable model, with the lowest overfitting risk on 104 training rows.
  - **Random Forest** would be the stronger default on larger, noisier data (variance reduction via ensembling), but that advantage does not materialise on a dataset this small and nearly linearly separable.
  - **Decision Tree** is the most interpretable as explicit rules but the highest-variance single model.

> Important: the correct statement is **not** "Logistic Regression had the best test accuracy" (it tied). It is: *all three tie on the test set; Logistic Regression wins the tie-break on cross-validated F1 and is the most parsimonious choice.* The notebook detects the tie automatically and reports it this way.

Note: the split is not stratified, giving an uneven 22/15/9 test-class mix. Using `stratified`-style sampling would distribute classes more evenly — a reasonable enhancement, though it would not change the overall conclusion on this dataset.

---

## 5. Repository Structure

```
.
├── Iris_Spark_MLlib.ipynb   # Complete, commented PySpark workflow with explanations + visualizations
├── iris.csv                 # Iris dataset (offline-reproducible copy)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 6. How to Reproduce

### Requirements
- Python 3.8+
- Java 8, 11, or 17 (required by Spark)
- PySpark 3.x

### Setup
```bash
# (optional) create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# install dependencies
pip install -r requirements.txt   # pyspark, pandas, matplotlib, seaborn, jupyter

# verify Java is available (Spark needs it)
java -version
```

### Run
```bash
jupyter notebook Iris_Spark_MLlib.ipynb
```
Then run all cells top to bottom (Kernel → Restart & Run All). The notebook ends by stopping the Spark session.

To run headless instead:
```bash
jupyter nbconvert --to notebook --execute Iris_Spark_MLlib.ipynb \
  --output Iris_Spark_MLlib_executed.ipynb
```

### Notes
- The notebook uses `local[*]` (all local cores); no cluster is required.
- A fixed `seed=42` is used for the split and the models so results are reproducible.
- If `iris.csv` is missing, the loader downloads the UCI copy automatically (needs internet for that fallback only).

---

## 7. Tools & Libraries

- **Apache Spark MLlib** — DataFrames, `Pipeline`, `StringIndexer`, `VectorAssembler`, `StandardScaler`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `CrossValidator`, `ParamGridBuilder`, `MulticlassClassificationEvaluator`, `MulticlassMetrics`.
- **pandas** — formatting the final comparison table.
- **matplotlib / seaborn** — exploratory and results visualizations (scatter matrix, box plots, correlation heatmap, confusion-matrix heatmaps, F1 bar chart, feature importances).
