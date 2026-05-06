# --- Barclay case: load weekly returns + run regressions (CNC on VOO, BLK on VOO) ---

import os
import pandas as pd
import statsmodels.api as sm

# 1) Exact file path
FILEPATH = r"/Users/pinnytu/Documents/IVEY MMA/Finance/08_Barclays Investment Exhibit 2 2021-11-17.xlsx"

if not os.path.exists(FILEPATH):
    raise FileNotFoundError(f"File not found: {FILEPATH}")

# 2) Read Sheet1 explicitly
df = pd.read_excel(FILEPATH, sheet_name="Sheet1")

# 3) Clean column names
df.columns = [c.strip() for c in df.columns]

print("Columns found:", df.columns)

# 4) Convert date column
df["Week Ending"] = pd.to_datetime(df["Week Ending"])

# 5) Convert percent strings to decimals
def to_decimal(x):
    if isinstance(x, str):
        x = x.strip().replace("%", "")
    return float(x) / 100

for col in ["BLK", "CNC", "VOO"]:
    df[col] = df[col].apply(to_decimal)

# 6) Sort and keep most recent 104 weeks (~2 years)
df = df.sort_values("Week Ending")
df_2y = df.tail(104).dropna()

print("Rows used:", len(df_2y))

# 7) Regression function
def run_regression(y_col):
    X = sm.add_constant(df_2y["VOO"])  # adds alpha
    y = df_2y[y_col]
    model = sm.OLS(y, X).fit()
    return model

# 8) Run regressions
model_cnc = run_regression("CNC")
model_blk = run_regression("BLK")

print("\n--- CNC on VOO ---")
print(model_cnc.summary())

print("\n--- BLK on VOO ---")
print(model_blk.summary())