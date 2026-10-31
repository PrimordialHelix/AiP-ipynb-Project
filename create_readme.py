import re

readme_content = """# Retail Sales Forecasting — Presenter Guide & Walkthrough

A complete, slide-by-slide presentation guide for the **Retail Sales Forecasting** project using machine learning.

---

## 1. Project Summary (30-Second Elevator Pitch)

* **Objective:** Predict monthly retail sales for the next 12 months using historical sales patterns.
* **Core Approach:**
  1. Take 5 years of daily store sales and sum them into monthly totals.
  2. Use first-order differencing to remove the upward trend and make the data stationary.
  3. Create 12 monthly lag features to capture year-over-year seasonality.
  4. Train three regression models: **Linear Regression**, **Random Forest**, and **XGBoost**.
  5. Test on a held-out 12-month period and compare accuracy using RMSE, MAE, and $R^2$.
* **Key Finding:** **Linear Regression** performed best ($R^2 = 0.9957$), accurately capturing smooth seasonal cycles without overfitting.

---

## 2. Step-by-Step Presentation Walkthrough

Use this section to guide your audience through the notebook from start to finish.

```
Raw Daily Data  ──>  Monthly Totals  ──>  Differencing (Stationary)
      │
      └──>  12-Month Lags  ──>  Train/Test Split  ──>  Models  ──>  Forecast Evaluation
```

### Step 1: Environment & Setup
* **What to say:** "We import standard data science tools: Pandas and NumPy for data handling, Matplotlib for charts, and Scikit-Learn plus XGBoost for machine learning."

### Step 2: Data Ingestion (Automatic GitHub Ingestion)
* **What to say:** "The notebook automatically downloads the 5-year sales dataset directly from our public GitHub repository. This means anyone can open the notebook in Google Colab or Jupyter and click Run All without uploading files manually."
* **Dataset details:**
  * 5 years of daily records (2019 to 2023, 1,826 days).
  * 5 stores and 5 products, totaling 45,650 daily transactions.
  * Shows steady year-over-year sales growth, weekend sales bumps, and summer/holiday peaks.

### Step 3: Data Preparation & Monthly Aggregation
* **What to say:** "First, we drop the store and item columns because we are forecasting total business sales, not individual items."
* **Why group daily sales into monthly totals?**
  * Daily data fluctuates wildly because of weekends and weekdays.
  * Summing sales by month smooths out daily noise while keeping the true annual seasonal patterns clear.
  * Reduces 45,650 daily rows into 60 clean monthly totals.

### Step 4: Making Data Stationary (First-Order Differencing)
* **What to say:** "Before training machine learning models, the time series must be stationary."
* **What is differencing?**
  $$\\Delta y_t = y_t - y_{t-1}$$
  Each month's value is replaced by the change from the previous month.
* **Why is this necessary?**
  * Raw sales grow higher every year. If we train on raw sales, the model struggles to predict future values that are higher than anything in the training set.
  * Differencing removes the upward drift and centers the data around zero. The models learn the momentum and seasonal swings instead of memorizing large numbers.

### Step 5: Feature Engineering (12-Month Lags)
* **What to say:** "To let supervised regression models learn time-series patterns, we turn the single column of differences into a feature table using 12 past months."
* **How it works:**
  $$X = [\\Delta y_{t-1}, \\Delta y_{t-2}, \\dots, \\Delta y_{t-12}] \\longrightarrow y = \\Delta y_t$$
* **Why 12 lags?**
  * Retail follows a yearly cycle.
  * 12 lags give the model the exact value from the same month of the previous year (for example, using last December to predict this December).
  * The first 12 rows are dropped because they do not have 12 past months, leaving 47 complete monthly samples.

### Step 6: Train/Test Split & Feature Scaling
* **What to say:** "We divide the 47 samples into two chronological parts: the first 35 months for training, and the final 12 months for testing."
* **Two strict rules applied:**
  1. **No random shuffling:** Time-series data must stay in chronological order. Shuffling causes data leakage by using future data to predict the past.
  2. **MinMax Scaling (-1 to 1):** Scaled strictly on training data and then applied to test data to prevent leakage. Keeps all input values on an equal scale.

### Step 7: The 3 Models
* **Model 1 — Linear Regression (OLS):**
  * A clear, straightforward baseline that fits a linear relationship between the 12 lags and the target month.
* **Model 2 — Random Forest Regressor:**
  * An ensemble of 100 decision trees. It combines predictions from multiple trees to model non-linear interactions.
* **Model 3 — XGBoost Regressor:**
  * A gradient boosted model. It builds 100 small trees sequentially, where each new tree corrects the errors made by previous trees.

### Step 8: Reconstructing Actual Sales Figures
* **What to say:** "The models predict the differenced value (the change in sales). To convert this back into actual units sold, we reverse the scaling and add the predicted change to the previous month's actual sales:"
  $$\\widehat{\\text{Sales}}_t = \\widehat{\\Delta y}_t + \\text{Sales}_{t-1}$$

---

## 3. Results & Performance Benchmark

Evaluated on the 12-month held-out test period:

| Model | RMSE (Units) | MAE (Units) | $R^2$ Score | Verdict |
|---|:---:|:---:|:---:|:---|
| **Linear Regression** | **346.24** | **318.09** | **0.9957** | **Best Performance (Lowest Error)** |
| **Random Forest Regressor** | **636.61** | **512.15** | **0.9856** | **Strong Non-Linear Ensemble** |
| **XGBoost Regressor** | **677.31** | **559.75** | **0.9837** | **Competitive Gradient Boosting** |

* **RMSE (Root Mean Squared Error):** Measures average deviation in sales units, penalizing large misses.
* **MAE (Mean Absolute Error):** Average unit error per month.
* **$R^2$ Score:** Measures the proportion of sales variation explained by the model (1.0 is a perfect match).

---

## 4. Key Questions & Answers for the Presentation Panel (Viva Guide)

Prepare for questions from evaluators or the audience with these clear answers:

### Q1: Why did Linear Regression beat more complex models like Random Forest and XGBoost?
> **Answer:**
> Aggregated monthly sales follow a smooth, continuous wave with clear annual peaks. Linear Regression fits smooth lines naturally without overfitting.
> Decision trees (Random Forest and XGBoost) predict using flat, step-like splits. On a smaller dataset ($N=47$), these step boundaries create small approximation errors around smooth cyclical peaks.

### Q2: Why was first-order differencing necessary before scaling?
> **Answer:**
> Raw sales have an upward trend. If we scaled raw data between -1 and 1, future sales that grow past historical records would fall outside the $[-1, 1]$ bounds.
> Differencing removes the trend and centers the numbers around zero, keeping all values stable and within safe bounds before scaling.

### Q3: Why did we choose 12 lag months as features?
> **Answer:**
> Retail sales are seasonal and follow a 12-month cycle. Using 12 lags gives the model direct access to the same month from the prior year (lag 12), enabling it to learn annual seasonality naturally.

### Q4: How did we ensure there is zero data leakage?
> **Answer:**
> Two ways:
> 1. We used a strict chronological train/test split (first 35 months to train, last 12 months to test) rather than random K-Fold shuffling.
> 2. The `MinMaxScaler` was fitted exclusively on the training set and only transformed onto the test set.

---

## 5. Quick-Start Instructions

1. **Open the Notebook:** Open [`sales_forecasting_aip_sem1.ipynb`](./sales_forecasting_aip_sem1.ipynb) in Google Colab, VS Code, or JupyterLab.
2. **Execute:** Click **Runtime** -> **Run All**.
3. The dataset is pulled automatically from GitHub, requiring zero manual setup.
"""

prohibited = [
    "beginner", "beginners", "frndly", "friendly", "user friendly", 
    "beginner friendly", "made by student", "made by students", 
    "for students", "student project", "student details", "student sign-off",
    "tutorial", "novice", "dummy"
]

found = []
for p in prohibited:
    matches = re.findall(rf"\b{p}\b", readme_content, re.IGNORECASE)
    if matches:
        found.append((p, len(matches)))

if found:
    print(f"ERROR: Prohibited tokens found: {found}")
    exit(1)
else:
    print("SUCCESS: 0 prohibited tokens found!")

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print("Updated README.md successfully.")

