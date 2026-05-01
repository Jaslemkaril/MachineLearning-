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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

model = joblib.load("electricity_model.pkl")

APPLIANCE_COLS = [
    "App_Electric_Fan", "App_Air_Conditioner", "App_Laptop_PC",
    "App_Refrigerator", "App_TV_Monitor", "App_Phone_Charger",
    "App_Electric_Kettle", "App_Rice_Cooker", "App_Study_Lamp",
]

FEATURE_COLS = [
    "Avg_Past_Consumption",
    "Hour", "Day", "IsWeekend", "TimeOfDay",
    "Dorm_Enc", "Room_Enc", "RoomSize_Enc", "Num_Occupants",
    *APPLIANCE_COLS,
]

DORM_MAP = {"Dorm A": 0, "Dorm B": 1, "Dorm C": 2}
SIZE_MAP = {"Small": 0, "Medium": 1, "Large": 2}

df = pd.read_csv("smart_meter_data.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Hour"]      = df["Timestamp"].dt.hour
df["Day"]       = df["Timestamp"].dt.day
df["IsWeekend"] = df["Timestamp"].dt.dayofweek.isin([5, 6]).astype(int)
df["TimeOfDay"] = pd.cut(df["Hour"], bins=[-1, 5, 11, 17, 23],
                          labels=[0, 1, 2, 3]).astype(int)
df["Dorm_Enc"]    = df["Dorm_ID"].map(DORM_MAP)
room_num = df["Room_ID"].str.extract(r"(\d+)").astype(int)
room_num = room_num.where(room_num < 100, room_num - 100)
df["Room_Enc"] = room_num - 1
df["RoomSize_Enc"] = df["Room_Size_Cat"].map(SIZE_MAP)
df = df.sort_values("Timestamp")

X = df[FEATURE_COLS]
y = df["Electricity_Consumed"]

train_size = int(len(df) * 0.7)
X_test = X.iloc[train_size:]
y_test = y.iloc[train_size:]

y_pred = model.predict(X_test)

mae  = round(mean_absolute_error(y_test, y_pred), 4)
rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
r2   = round(r2_score(y_test, y_pred), 4)

print("Computing 5-fold CV (this takes ~30s)...")
cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
cv = round(float(cv_scores.mean()), 4)

# Feature importances
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

# Top consuming rooms
agg = (
    df.groupby(["Dorm_ID", "Room_ID"])["Electricity_Consumed"]
      .mean()
      .reset_index()
      .rename(columns={"Electricity_Consumed": "value"})
      .sort_values("value", ascending=False)
      .head(5)
)
def _normalize_room_label(room_id: str) -> str:
    match = pd.Series([room_id]).str.extract(r"(\d+)").iloc[0, 0]
    if match is None:
        return room_id
    room_num = int(match)
    if room_num >= 100:
        room_num -= 100
    return f"Room {room_num}"

top_rooms = [
    {
        "dorm": row.Dorm_ID,
        "room": _normalize_room_label(row.Room_ID),
        "value": round(row.value, 4),
    }
    for row in agg.itertuples()
]

# Chart
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

stats_cache = {
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
    "cv": cv,
    "importances": importances,
    "chart": chart_b64,
    "model_type": type(model).__name__,
    "top_rooms": top_rooms,
}

with open("stats_cache.json", "w", encoding="utf-8") as f:
    json.dump(stats_cache, f, ensure_ascii=False, indent=2)

print(f"\nStats cached to stats_cache.json")
print(f"MAE: {mae}, RMSE: {rmse}, R²: {r2}, CV R²: {cv}")
print(f"Chart size: {len(chart_b64)} chars")
print(f"Top rooms: {len(top_rooms)}")
