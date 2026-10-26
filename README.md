# AIP Labs — Semester 1 CSE: Time-Series Retail Sales Forecasting

## Academic Lab Project Overview
This repository folder contains the complete, self-contained implementation of the **AIP (Artificial Intelligence Project / Practices), Semester 1 Computer Science & Engineering (CSE)** lab assignment on **Time-Series Sales Forecasting**.

Inspired by the step-by-step structure of `sales-forecasting.ipynb`, this project:
1. **Removes all Kaggle dependencies** — fully reproducible anywhere (JupyterLab, VS Code, Google Colab).
2. **Uses our own custom datasheet** (`retail_sales_dataset.csv`) comprising 5 years of retail transactions across 5 stores and 5 products ($N=45,650$ rows).
3. **Implements three supervised machine learning algorithms**:
   - **Linear Regression** (Ordinary Least Squares baseline)
   - **Random Forest Regressor** (Bagging ensemble with 100 estimators)
   - **XGBoost Regressor** (Gradient boosted decision trees)
4. **Calculates empirical evaluation metrics** (RMSE, MAE, $R^2$) on a held-out 12-month test partition.

---

## Directory Contents
* [retail_sales_dataset.csv](file:///d:/NewProject/sales-forecasting/AiP%20labs/retail_sales_dataset.csv): Our custom 5-year retail sales datasheet ($N=45,650$ rows).
* [sales_forecasting_aip_sem1.ipynb](file:///d:/NewProject/sales-forecasting/AiP%20labs/sales_forecasting_aip_sem1.ipynb): The main Google Colab / Jupyter Notebook containing all 46 educational markdown and code cells.
* [sales-forecasting.ipynb](file:///d:/NewProject/sales-forecasting/AiP%20labs/sales-forecasting.ipynb): Identical notebook copy matching the requested filename convention.

---

## Datasheet Schema (`retail_sales_dataset.csv`)
* `date`: Daily timestamp from `2019-01-01` to `2023-12-31` (1,826 daily timestamps).
* `store`: Retail store identifier (`1` to `5`).
* `item`: Product SKU identifier (`1` to `5`).
* `sales`: Daily units sold (accounting for annual trend, weekend lift, seasonal oscillation, and noise).

---

## Model Evaluation Benchmark (12-Month Held-Out Test Set)

| Model | RMSE (Units) | MAE (Units) | $R^2$ Score | Verdict |
|---|:---:|:---:|:---:|:---|
| **Linear Regression** | **346.24** | **318.09** | **0.9957** | **Best Baseline (Lowest Error)** |
| **Random Forest Regressor** | **636.61** | **512.15** | **0.9856** | **Strong Bagging Ensemble** |
| **XGBoost Regressor** | **677.31** | **559.75** | **0.9837** | **Competitive Gradient Boosting** |

---

## How to Run in Google Colab / Jupyter
1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com/)
2. Upload `sales_forecasting_aip_sem1.ipynb` (or `sales-forecasting.ipynb`).
3. Click **Runtime** -> **Run All**.
4. If `retail_sales_dataset.csv` is not uploaded, the notebook automatically auto-generates the datasheet inline without throwing any `FileNotFoundError`.
