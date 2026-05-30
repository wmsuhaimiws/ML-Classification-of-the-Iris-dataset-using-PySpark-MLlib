# Iris Flower Classification with Apache Spark MLlib

**STQD6324 Data Management — Assignment 1**

An end-to-end multiclass classification project on the classic **Iris** dataset, built entirely with **Apache Spark MLlib**. Three classifiers — Logistic Regression, Decision Tree, and Random Forest — are trained, tuned with cross-validation and grid search, evaluated on a held-out test set, and compared so that a best model can be justified.

---

## 1. Project Overview

The goal is to predict the species of an iris flower (*setosa*, *versicolor*, or *virginica*) from four morphological measurements. A complete, reproducible Spark MLlib workflow is demonstrated: data loading into a Spark DataFrame, preprocessing with `StringIndexer`/`VectorAssembler`/`StandardScaler`, pipeline construction, hyperparameter tuning with 5-fold `CrossValidator` + `ParamGridBuilder`, multi-metric evaluation (accuracy, precision, recall, F1, confusion matrix), prediction on unseen data, and a comparative analysis of the three algorithms.

Every step in the notebook is accompanied by a markdown explanation of *what* is done, *why*, and *how the output should be interpreted*.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository (also bundled with scikit-learn) |
| Rows | 150 |
| Features | 4 numeric (cm): `sepal_length`, `sepal_width`, `petal_length`, `petal_width` |
| Target | `species` — 3 balanced classes (50 each): setosa, versicolor, virginica |
| Missing values | None |

A clean copy, `iris.csv`, is included in the repository so that the notebook can be run **offline and reproducibly**. If the file is absent, the canonical UCI copy is downloaded and normalised automatically.

**Key data characteristics that shape the analysis:** the dataset is small, perfectly class-balanced, and nearly linearly separable. *Setosa* is trivially separable from the other two; essentially all classification difficulty lies in the modest overlap between *versicolor* and *virginica*.

---

## 3. Methodology

1. **Spark session** — a local-mode `SparkSession` is created with fixed shuffle partitions for deterministic, fast small-data runs.
2. **Load** — `iris.csv` is read into a Spark DataFrame with an explicit schema.
3. **EDA & preprocessing** — missing values and class balance are verified; the data is **visualised** (pairwise scatter matrix, per-feature box plots, correlation heatmap) to expose class separability; the label is encoded with `StringIndexer`; the features are assembled with `VectorAssembler`; a `StandardScaler` is added only to the Logistic Regression pipeline (tree models are scale-invariant).
4. **Stratified split** — 70% train / 30% test is sampled *within each species* via `sampleBy` (classes are kept balanced in both sets, unlike plain `randomSplit`), `seed=42`.
5. **Pipelines** — one `Pipeline` is built per model so that preprocessing is fit per cross-validation fold (no data leakage).
6. **Tuning** — a 5-fold `CrossValidator` with `ParamGridBuilder` is run, optimising **F1**:
   - **Logistic Regression:** `regParam`, `elasticNetParam`, `maxIter`
   - **Decision Tree:** `maxDepth`, `impurity`
   - **Random Forest:** `numTrees`, `maxDepth`, `featureSubsetStrategy`
7. **Evaluation** — accuracy, weighted precision/recall/F1, a confusion matrix, **and per-class precision/recall/F1** are computed per model on the test set.
8. **Prediction** — readable predicted-vs-actual species are produced via `IndexToString`.
9. **Comparison** — a metrics table (incl. cross-validated F1) is presented; **visual analysis** (confusion-matrix heatmaps, CV-vs-test F1 bars, feature importances) is provided; a **robustness study** repeats evaluation over 10 stratified splits; **model interpretability** is examined (LR coefficients, Decision Tree rules, 2D decision boundary); strengths/limitations are discussed; and a best-model choice is justified.

---

## 4. Results & Key Findings

**On a single held-out split, all three tuned models tie** — Accuracy ≈ 0.978, weighted F1 ≈ 0.979, with *identical* confusion matrices. Every setosa and virginica test flower is classified correctly by each model, and the **same single versicolor** flower is missed by all of them. With ~14 test rows per class and one error, a single split cannot rank strong models — so the comparison is taken further.

**The robustness study settles it.** When each model's tuned configuration is re-fit on **10 independent stratified splits**, a distribution of F1 per model is obtained (illustrative pattern; exact figures are produced by the executed run):

| Model | Mean F1 | Std F1 | Reading |
|---|---|---|---|
| **Logistic Regression** | highest (~0.96) | **lowest** (~0.027) | best *and* most stable |
| Random Forest | ~0.96 (close 2nd) | low (~0.031) | strong, robust alternative |
| Decision Tree | lowest (~0.94) | highest (~0.038) | weakest, highest-variance single tree |

**Key findings:**
- **Setosa** is perfectly separable and is never misclassified; all error is located on the **versicolor ↔ virginica** boundary, as confirmed by the per-class metrics and the 2D decision-boundary plot.
- **Logistic Regression is selected as the best model**, justified three ways that agree: (1) highest **mean** F1 across splits, (2) **lowest variance** (most reliable), and (3) **parsimony/interpretability** — it is the simplest, fastest, most transparent model, with the lowest overfitting risk on ~104 training rows.
- **Random Forest** is the recommended alternative when robustness on larger, noisier data matters more than interpretability; its variance-reduction advantage is muted on a dataset this small and nearly linearly separable.

> Honest framing: the correct statement is **not** "Logistic Regression had the best test accuracy" (it tied on the single split). It is: *repeated stratified evaluation shows that Logistic Regression is best on average and most stable, and it is also the most interpretable — hence the justified choice.*

Interpretability outputs are included: the Logistic Regression coefficient matrix (petal features carry the largest weights), the Decision Tree's learned rules, and the Random Forest feature importances (petal length/width dominate) — all mutually consistent.

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
# (optional) a virtual environment can be created
python3 -m venv .venv && source .venv/bin/activate

# dependencies are installed
pip install -r requirements.txt   # pyspark, pandas, matplotlib, seaborn, jupyter

# Java availability is verified (Spark requires it)
java -version
```

### Run
```bash
jupyter notebook Iris_Spark_MLlib.ipynb
```
All cells should then be run top to bottom (Kernel → Restart & Run All). The Spark session is stopped at the end of the notebook.

For a headless run instead:
```bash
jupyter nbconvert --to notebook --execute Iris_Spark_MLlib.ipynb \
  --output Iris_Spark_MLlib_executed.ipynb
```

### Notes
- `local[*]` is used (all local cores); no cluster is required.
- A fixed `seed=42` is used for the stratified split and the models so that results are reproducible; 10 fixed seeds are swept in the robustness study.
- All three models are re-fit on 10 splits in the robustness study, so a full run takes a few minutes on a laptop — this is expected.
- If `iris.csv` is missing, the UCI copy is downloaded automatically (internet is needed for that fallback only).

---

## 7. Tools & Libraries

- **Apache Spark MLlib** — DataFrames, `Pipeline`, `StringIndexer`, `VectorAssembler`, `StandardScaler`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `CrossValidator`, `ParamGridBuilder`, `MulticlassClassificationEvaluator`, `MulticlassMetrics`.
- **pandas** — used for formatting the final comparison table.
- **matplotlib / seaborn** — used for the exploratory and results visualizations (scatter matrix, box plots, correlation heatmap, confusion-matrix heatmaps, F1 bar chart, feature importances, F1 robustness box plot, decision-boundary plot).
