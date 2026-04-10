import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/Air_Quality.csv")

# =========================
# BASIC CLEANING
# =========================

df = df.drop(columns=["Unique ID", "Geo Join ID", "Message"])

# =========================
# DATE ENGINEERING
# =========================

df["Start_Date"] = pd.to_datetime(df["Start_Date"])

df["year"] = df["Start_Date"].dt.year
df["month"] = df["Start_Date"].dt.month
df["day_of_week"] = df["Start_Date"].dt.dayofweek

# =========================
# MISSING VALUES
# =========================

df["Data Value"] = df.groupby("Name")["Data Value"].transform(
    lambda x: x.fillna(x.mean())
)

# =========================
# OUTLIER REMOVAL (IQR)
# =========================

Q1 = df["Data Value"].quantile(0.25)
Q3 = df["Data Value"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["Data Value"] >= lower) & (df["Data Value"] <= upper)]

# =========================
# TARGET ENCODING
# =========================

le = LabelEncoder()
df["target"] = le.fit_transform(df["Name"])

# =========================
# ONE HOT ENCODING
# =========================

df = pd.get_dummies(
    df,
    columns=["Measure", "Geo Type Name", "Geo Place Name", "Time Period"],
)

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

df["Data Value"] = scaler.fit_transform(df[["Data Value"]])

# =========================
# SAVE DATASET
# =========================

df.to_csv("data/air_quality_processed.csv", index=False)

print("Preprocessing completed")
print("Final shape:", df.shape)