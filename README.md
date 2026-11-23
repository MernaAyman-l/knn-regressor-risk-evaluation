# KNN Regressor & Risk Classification Evaluation

A Jupyter Notebook (`task.ipynb`) implementing a **K-Nearest Neighbors Regressor** (`KNeighborsRegressor` with `n_neighbors=5`) to predict continuous values, evaluate regression performance (`MSE`, `MAE`, and `R²`), and threshold predictions into binary risk classes (`Low Risk` vs. `High Risk`) for Confusion Matrix and ROC/AUC analysis.

## Project Structure & Key Files
* `task.ipynb`: Jupyter Notebook containing model training, prediction, regression metrics calculation, binary thresholding, Confusion Matrix visualization, and ROC/AUC plotting.

## Model & Evaluation Workflow
* **Regression Training & Prediction:** Fits `KNeighborsRegressor(n_neighbors=5)` on `(x_train, y_train)` and generates predictions `y_pred` on `x_test`.
* **Regression Metrics:** Evaluates `y_test` against `y_pred` using:
  * **Mean Squared Error (MSE):** `2.2077`
  * **Mean Absolute Error (MAE):** `1.1187`
  * **R² Score:** `0.7734`
* **Binary Risk Thresholding & Classification Evaluation:** Uses `y_test.mean()` as a threshold to convert continuous target and predicted values into binary classes (`Low Risk` / `High Risk`), computes the Confusion Matrix (`[[2830, 794], [453, 2440]]`), plots a seaborn heatmap, and calculates ROC curve points and AUC score.

## Built With
* **Python**
* **scikit-learn** (`KNeighborsRegressor`, `mean_squared_error`, `mean_absolute_error`, `r2_score`, `roc_curve`, `auc`, `confusion_matrix`)
* **Matplotlib & Seaborn** (Confusion Matrix heatmap and ROC curve visualization)

## How to Run

### Requirements

- Python 3.9+
- Jupyter Notebook or JupyterLab
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, and `seaborn`

Install the required packages:

```bash
python -m pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Run the notebook

1. Open `task.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
2. Run the notebook cells in order **after preparing the input arrays**: `x_train`, `y_train`, `x_test`, and `y_test`.

> **Reproducibility note:** The repository currently contains the notebook only; it does not include the source dataset or the preprocessing steps that create the train/test arrays. Therefore, the notebook is not fully reproducible from a fresh clone until those inputs are supplied. The reported metrics and plots are the saved notebook outputs, not a guarantee that a new run will reproduce the same values.

## Scope & Possible Improvements

- Add the source dataset or link to its public source, subject to its license.
- Include the preprocessing and train/test split steps so the notebook runs from a clean environment.
- Document the target variable and explain why the regression output is converted into low/high risk classes.
- Use a domain-defined risk threshold where possible, rather than deriving it from the test target mean.
- Add a clear evaluation strategy and avoid using test data to choose the classification threshold.
