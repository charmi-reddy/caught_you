import pandas as pd
import numpy as np

# ===============================
# 1. LOAD FIRST 50K ROWS
# ===============================

file_path = r"C:\Users\Charmi\Desktop\fraud detection\PaySim.csv"

df = pd.read_csv(file_path, nrows=50000)

print("Dataset shape:", df.shape)

# ===============================
# 2. KEEP ONLY NECESSARY COLUMNS
# ===============================

df = df[[
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    "isFraud"
]]

# Reduce memory usage
df["step"] = df["step"].astype("int32")
df["amount"] = df["amount"].astype("float32")
df["isFraud"] = df["isFraud"].astype("int8")

# ===============================
# 3. SORT BY TIME
# ===============================

df = df.sort_values("step").reset_index(drop=True)

# ===============================
# 4. FILTER ONLY TRANSFER & CASH_OUT
# ===============================

df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].reset_index(drop=True)

print("After filtering:", df.shape)

# ===============================
# 5. FAST PAIRWISE HISTORY FEATURES
# ===============================

group = df.groupby(["nameOrig", "nameDest"])

# Transaction count BEFORE current transaction
df["pair_tx_count"] = group.cumcount()

# Cumulative amount BEFORE current transaction
df["cum_amount"] = group["amount"].cumsum() - df["amount"]

# Historical average
df["pair_avg"] = df["cum_amount"] / df["pair_tx_count"].replace(0, np.nan)

# Historical max BEFORE current transaction
df["pair_max"] = group["amount"].cummax().shift()

# Fill NaNs
df["pair_avg"] = df["pair_avg"].fillna(0)
df["pair_max"] = df["pair_max"].fillna(0)

# ===============================
# 6. SPIKE DETECTION FEATURE
# ===============================

df["spike_ratio"] = df["amount"] / (df["pair_avg"] + 1)

# ===============================
# 7. RULE-BASED FLAG
# ===============================

df["rule_flag"] = (
    (df["spike_ratio"] > 20) |
    (df["amount"] > 5 * (df["pair_max"] + 1))
).astype(int)

# ===============================
# 8. VIEW SUSPICIOUS TRANSACTIONS
# ===============================

suspicious = df[df["rule_flag"] == 1]

print("Suspicious spike transactions:", suspicious.shape[0])

print(suspicious[[
    "step",
    "nameOrig",
    "nameDest",
    "amount",
    "pair_avg",
    "pair_max",
    "spike_ratio",
    "isFraud"
]].head(20))
