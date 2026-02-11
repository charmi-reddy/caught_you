import pandas as pd
import numpy as np

# ===============================
# 1. LOAD DATA
# ===============================

file_path = r"C:\Users\Charmi\Desktop\fraud detection\PaySim.xlsx"
df = pd.read_excel(file_path, sheet_name="PaySim")

print("Dataset shape:", df.shape)
print(df.head())

# ===============================
# 2. SORT BY TIME
# ===============================

df = df.sort_values("step").reset_index(drop=True)

# ===============================
# 3. FILTER ONLY TRANSFER & CASH_OUT
# ===============================

df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].reset_index(drop=True)

print("After filtering:", df.shape)

# ===============================
# 4. CREATE PAIRWISE HISTORY FEATURES
# ===============================

# Historical average amount per sender-receiver pair
df["pair_avg"] = (
    df.groupby(["nameOrig", "nameDest"])["amount"]
      .transform(lambda x: x.shift().expanding().mean())
)

# Historical max amount per pair
df["pair_max"] = (
    df.groupby(["nameOrig", "nameDest"])["amount"]
      .transform(lambda x: x.shift().expanding().max())
)

# Count of transactions between pair
df["pair_tx_count"] = (
    df.groupby(["nameOrig", "nameDest"]).cumcount()
)

# Fill NaNs for first transactions
df["pair_avg"] = df["pair_avg"].fillna(0)
df["pair_max"] = df["pair_max"].fillna(0)

# ===============================
# 5. SPIKE DETECTION FEATURE
# ===============================

df["spike_ratio"] = df["amount"] / (df["pair_avg"] + 1)

# ===============================
# 6. RULE-BASED FLAG
# ===============================

df["rule_flag"] = np.where(
    (df["spike_ratio"] > 20) | 
    (df["amount"] > 5 * (df["pair_max"] + 1)),
    1,
    0
)

# ===============================
# 7. VIEW SUSPICIOUS TRANSACTIONS
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
