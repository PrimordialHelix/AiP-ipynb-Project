import json
import re

with open("sales_forecasting_aip_sem1.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Cell 0 overview to mention automated GitHub data loading
cell_0_text = """# Retail Sales Forecasting using Supervised Machine Learning
## Time-Series Analysis with 12-Month Lag Features & Multi-Model Benchmarking

---

### Project Overview
* **Goal:** Forecast monthly retail sales for the next 12 months using historical sales patterns.
* **Core Strategy:**
  1. **Automated Data Ingestion:** Automatically fetch the 5-year retail sales dataset from the public GitHub repository.
  2. **Aggregate:** Convert daily store records into total monthly sales.
  3. **Stationarize:** Use first-order differencing ($\\Delta y_t = y_t - y_{t-1}$) to remove long-term trend and center sales changes around zero.
  4. **Feature Engineering:** Build 12 monthly lag features ($t-1$ through $t-12$) to capture annual seasonality.
  5. **Train/Test Split:** Train on the first 35 months, test on the final 12 months (out-of-time evaluation with no data leakage).
  6. **Benchmark:** Compare Linear Regression, Random Forest, and XGBoost on test accuracy (RMSE, MAE, $R^2$).
"""
nb["cells"][0]["source"] = [l + "\n" for l in cell_0_text.splitlines()]

# Update Cell 3 markdown
cell_3_text = """## Phase 2: Load the Sales Dataset from Public GitHub Repository

**Goal:** Automatically ingest the 5-year retail sales dataset directly from the public GitHub repository ([`retail_sales_dataset.csv`](https://github.com/PrimordialHelix/AiP-ipynb-Project/blob/main/retail_sales_dataset.csv)).

* **Data Source:** Hosted on GitHub and loaded via raw URL for instant execution in any environment (Google Colab, Jupyter, Kaggle).
* **Timeframe:** 5 years of daily sales (2019 to 2023, 1,826 days).
* **Structure:** 5 stores, 5 product items, yielding 45,650 daily sales records.
* **Pattern:** Shows steady yearly growth, seasonal summer/holiday peaks, and weekend sales spikes.
"""
nb["cells"][3]["source"] = [l + "\n" for l in cell_3_text.splitlines()]

# Update Cell 4 code
cell_4_code = """# Ingest dataset automatically from public GitHub repository
github_raw_url = 'https://raw.githubusercontent.com/PrimordialHelix/AiP-ipynb-Project/main/retail_sales_dataset.csv'
local_candidates = [
    'retail_sales_dataset.csv',
    'AiP labs/retail_sales_dataset.csv',
    '../AiP labs/retail_sales_dataset.csv',
    'data/sample_sales.csv'
]

# Attempt direct GitHub ingestion with local fallback
dataset_loaded = False

try:
    print(f"Fetching dataset directly from public GitHub URL:\\n{github_raw_url}")
    store_sales = pd.read_csv(github_raw_url)
    dataset_loaded = True
    print(f"Successfully loaded {len(store_sales):,} records from GitHub!")
    # Cache locally for offline execution
    store_sales.to_csv('retail_sales_dataset.csv', index=False)
except Exception as e:
    print(f"Online fetch failed ({e}). Checking local files...")
    for p in local_candidates:
        if os.path.exists(p):
            print(f"Loading local dataset from: '{p}'")
            store_sales = pd.read_csv(p)
            dataset_loaded = True
            break

if not dataset_loaded:
    print("Generating fallback retail sales datasheet...")
    np.random.seed(42)
    dates = pd.date_range(start="2019-01-01", end="2023-12-31", freq="D")
    rows = []
    for s in [1, 2, 3, 4, 5]:
        store_mult = 0.8 + 0.1 * s
        for itm in [1, 2, 3, 4, 5]:
            item_base = 15.0 + 5.0 * itm
            t = np.arange(len(dates))
            trend = 1.0 + 0.08 * (t / 365.25)
            dow = dates.dayofweek
            weekend_boost = np.where(dow == 5, 1.25, np.where(dow == 6, 1.30, 1.0))
            annual_seasonality = 1.0 + 0.18 * np.sin(2 * np.pi * dates.dayofyear / 365.25 - 0.5) + 0.12 * np.cos(4 * np.pi * dates.dayofyear / 365.25)
            noise = np.random.normal(1.0, 0.08, size=len(dates))
            daily_sales = np.clip(np.round(item_base * store_mult * trend * weekend_boost * annual_seasonality * noise), 1, None).astype(int)
            for d_idx, d in enumerate(dates):
                rows.append({'date': d.strftime('%Y-%m-%d'), 'store': s, 'item': itm, 'sales': int(daily_sales[d_idx])})
    store_sales = pd.DataFrame(rows).sort_values(['date', 'store', 'item']).reset_index(drop=True)
    store_sales.to_csv('retail_sales_dataset.csv', index=False)
    print("Datasheet saved as 'retail_sales_dataset.csv'!")

# Display the first 10 records
store_sales.head(10)
"""
nb["cells"][4]["source"] = [l + "\n" for l in cell_4_code.splitlines()]

# Prohibited scan
prohibited = [
    "beginner", "beginners", "frndly", "friendly", "user friendly", 
    "beginner friendly", "made by student", "made by students", 
    "for students", "student project", "student details", "student sign-off",
    "tutorial", "novice", "dummy"
]
all_text = ""
for c in nb["cells"]:
    all_text += "".join(c.get("source", [])) + "\n"

found = []
for p in prohibited:
    matches = re.findall(rf"\b{p}\b", all_text, re.IGNORECASE)
    if matches:
        found.append((p, len(matches)))

if found:
    print(f"ERROR: Prohibited phrases detected: {found}")
    exit(1)
else:
    print("SUCCESS: 0 prohibited phrases detected!")

with open("sales_forecasting_aip_sem1.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated sales_forecasting_aip_sem1.ipynb successfully.")

