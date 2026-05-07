"""
precompute_stats.py
Pre-compute model stats and save to stats_cache.json so app startup is instant.
Run this locally after training, then commit the JSON file.
"""

import io
import json
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.model_selection import cross_val_score

from config import FEATURE_COLS, DORM_MAP, SIZE_MAP, ANOMALY_THRESHOLD

# ── Load models ───────────────────────────────────────────────────────────────
model = joblib.load("electricity_model.pkl")
try:
    lr_model = joblib.load("electricity_model_lr.pkl")
    HAS_LR = True
except Exception:
    HAS_LR = False

# ── Load and prepare data ─────────────────────────────────────────────────────
df = pd.read_csv("smart_meter_data.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp")

# Verify feature columns
missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

X = df[FEATURE_COLS]
y = df["Electricity_Consumed"]

train_size = int(len(df) * 0.7)
X_test = X.iloc[train_size:]
y_test = y.iloc[train_size:]

# ── RF metrics ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

mae  = round(mean_absolute_error(y_test, y_pred), 4)
rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
r2   = round(r2_score(y_test, y_pred), 4)

print("Computing 5-fold CV (this takes a moment)...")
cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
cv = round(float(cv_scores.mean()), 4)

# ── LR baseline metrics ──────────────────────────────────────────────────────
lr_metrics = {}
if HAS_LR:
    lr_pred = lr_model.predict(X_test)
    lr_metrics = {
        "lr_mae": round(mean_absolute_error(y_test, lr_pred), 4),
        "lr_rmse": round(float(np.sqrt(mean_squared_error(y_test, lr_pred))), 4),
        "lr_r2": round(r2_score(y_test, lr_pred), 4),
        "lr_cv": round(float(cross_val_score(lr_model, X, y, cv=5, scoring="r2", n_jobs=-1).mean()), 4),
    }

# ── Feature importances ──────────────────────────────────────────────────────
if hasattr(model, "feature_importances_"):
    values = list(model.feature_importances_)
    max_val = max(values) or 1
    importances = [
        {
            "feature": f,
            "coef": round(v, 4),
            "width": max(2, round(v / max_val * 100)),
            "color_class": "coeff-bar-pos",
        }
        for f, v in zip(FEATURE_COLS, values)
    ]
else:
    values = list(model.coef_)
    max_abs = max(abs(c) for c in values) or 1
    importances = [
        {
            "feature": f,
            "coef": round(c, 4),
            "width": max(2, round(abs(c) / max_abs * 100)),
            "color_class": "coeff-bar-pos" if c >= 0 else "coeff-bar-neg",
        }
        for f, c in zip(FEATURE_COLS, values)
    ]

# ── Top consuming rooms ──────────────────────────────────────────────────────
agg = (
    df.groupby(["Dorm_ID", "Room_ID"])["Electricity_Consumed"]
      .mean()
      .reset_index()
      .rename(columns={"Electricity_Consumed": "value"})
      .sort_values("value", ascending=False)
      .head(5)
)

top_rooms = [
    {
        "dorm": row.Dorm_ID,
        "room": row.Room_ID,
        "value": round(row.value, 4),
    }
    for row in agg.itertuples()
]

# ── Monthly consumption data (for Chart.js in the dashboard) ─────────────────
monthly_data = (
    df.groupby("Month")["Electricity_Consumed"]
      .agg(["mean", "std", "count"])
      .reset_index()
)
monthly_chart = {
    "months": monthly_data["Month"].tolist(),
    "means": [round(v, 4) for v in monthly_data["mean"].tolist()],
    "stds": [round(v, 4) for v in monthly_data["std"].tolist()],
    "counts": monthly_data["count"].tolist(),
}

# ── Hourly consumption data ───────────────────────────────────────────────────
hourly_data = (
    df.groupby("Hour")["Electricity_Consumed"]
      .mean()
      .reindex(range(24), fill_value=0)
      .tolist()
)
hourly_data = [round(v, 4) for v in hourly_data]

# ── Dorm consumption data ─────────────────────────────────────────────────────
dorm_data = (
    df.groupby("Dorm_ID")["Electricity_Consumed"]
      .agg(["mean", "std"])
      .reset_index()
)
dorm_chart = {
    "dorms": dorm_data["Dorm_ID"].tolist(),
    "means": [round(v, 4) for v in dorm_data["mean"].tolist()],
    "stds": [round(v, 4) for v in dorm_data["std"].tolist()],
}

# ── Chart (Actual vs Predicted - dark theme) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 3.5))
fig.patch.set_facecolor("#0f172a")
ax.set_facecolor("#0f172a")
ax.plot(y_test.values[:100], color="#38bdf8", linewidth=1.5, label="Actual")
ax.plot(y_pred[:100], color="#818cf8", linewidth=1.5, label="Predicted", linestyle="--")
ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=9)
ax.set_xlabel("Sample", color="#94a3b8", fontsize=9)
ax.set_ylabel("Consumption", color="#94a3b8", fontsize=9)
ax.tick_params(colors="#64748b")
for spine in ax.spines.values():
    spine.set_edgecolor("#1e293b")
plt.tight_layout()

buf = io.BytesIO()
plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0f172a")
plt.close(fig)
buf.seek(0)
chart_b64 = base64.b64encode(buf.read()).decode("utf-8")

# ── Classification metrics ────────────────────────────────────────────────────
clf_metrics = {}
try:
    classifier = joblib.load("electricity_classifier.pkl")
    HAS_CLF = True
except Exception:
    HAS_CLF = False

if HAS_CLF:
    # Generate binary labels from dataset
    y_binary = (df["Electricity_Consumed"] > ANOMALY_THRESHOLD).astype(int)
    y_test_binary = y_binary.iloc[train_size:]
    clf_pred = classifier.predict(X_test)

    clf_cm = confusion_matrix(y_test_binary, clf_pred).tolist()
    clf_metrics = {
        "clf_model": type(classifier).__name__,
        "clf_accuracy": round(accuracy_score(y_test_binary, clf_pred), 4),
        "clf_precision": round(precision_score(y_test_binary, clf_pred, zero_division=0), 4),
        "clf_recall": round(recall_score(y_test_binary, clf_pred, zero_division=0), 4),
        "clf_f1": round(f1_score(y_test_binary, clf_pred, zero_division=0), 4),
        "clf_confusion_matrix": clf_cm,
    }
    print(f"Classifier: {clf_metrics['clf_model']} — "
          f"Acc: {clf_metrics['clf_accuracy']}, F1: {clf_metrics['clf_f1']}")

# ── Build stats cache ─────────────────────────────────────────────────────
stats_cache = {
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
    "cv": cv,
    **lr_metrics,
    **clf_metrics,
    "importances": importances,
    "chart": chart_b64,
    "model_type": type(model).__name__,
    "top_rooms": top_rooms,
    "monthly_chart": monthly_chart,
    "hourly_data": hourly_data,
    "dorm_chart": dorm_chart,
    "dataset_info": {
        "records": len(df),
        "months": int(df["Month"].nunique()),
        "dorms": int(df["Dorm_ID"].nunique()),
        "rooms": int(df["Room_ID"].nunique()),
        "date_range": f"{df['Timestamp'].min()} to {df['Timestamp'].max()}",
    },
}

with open("stats_cache.json", "w", encoding="utf-8") as f:
    json.dump(stats_cache, f, ensure_ascii=False, indent=2)

print(f"\nStats cached to stats_cache.json")
print(f"RF  — MAE: {mae}, RMSE: {rmse}, R²: {r2}, CV R²: {cv}")
if HAS_LR:
    print(f"LR  — MAE: {lr_metrics['lr_mae']}, RMSE: {lr_metrics['lr_rmse']}, "
          f"R²: {lr_metrics['lr_r2']}, CV R²: {lr_metrics['lr_cv']}")
print(f"Chart size: {len(chart_b64)} chars")
print(f"Top rooms: {len(top_rooms)}")
print(f"Monthly data: {len(monthly_chart['months'])} months")
print(f"Hourly data: {len(hourly_data)} hours")
