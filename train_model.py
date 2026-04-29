import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# ── Load & feature-engineer ──────────────────────────────────────────────────
df = pd.read_csv("smart_meter_data.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Handle missing values (realistic smart meter data has gaps)
print("Handling missing values...")
print(f"Missing values before: {df.isnull().sum().sum()}")

# Fill missing environmental values with forward fill then backward fill
for col in ['Temperature', 'Humidity', 'Wind_Speed', 'Avg_Past_Consumption']:
    if col in df.columns:
        df[col] = df[col].ffill().bfill().fillna(df[col].mean())

print(f"Missing values after: {df.isnull().sum().sum()}")

df["Hour"]      = df["Timestamp"].dt.hour
df["Day"]       = df["Timestamp"].dt.day
df["Month"]     = df["Timestamp"].dt.month
df["IsWeekend"] = df["Timestamp"].dt.dayofweek.isin([5, 6]).astype(int)
df["Season"]    = df["Month"].map({12:0,1:0,2:0, 3:1,4:1,5:1,
                                    6:2,7:2,8:2, 9:3,10:3,11:3})
df["TimeOfDay"] = pd.cut(df["Hour"], bins=[-1,5,11,17,23],
                          labels=[0,1,2,3]).astype(int)
df["Is_Anomaly"] = (df["Anomaly_Label"] != "Normal").astype(int)

# Encode categorical room fields
df["Dorm_Enc"]     = df["Dorm_ID"].map({"Dorm A": 0, "Dorm B": 1, "Dorm C": 2})
df["Room_Enc"]     = df["Room_ID"].str.extract(r"(\d+)").astype(int) - 101
df["RoomSize_Enc"] = df["Room_Size_Cat"].map({"Small": 0, "Medium": 1, "Large": 2})

APPLIANCE_COLS = [
    "App_Electric_Fan", "App_Air_Conditioner", "App_Laptop_PC",
    "App_Refrigerator", "App_TV_Monitor", "App_Phone_Charger",
    "App_Electric_Kettle", "App_Rice_Cooker", "App_Study_Lamp",
]

FEATURE_COLS = [
    # Environmental
    "Temperature", "Humidity", "Wind_Speed", "Avg_Past_Consumption",
    # Time
    "Hour", "Day", "Month", "IsWeekend", "Season", "TimeOfDay",
    # Anomaly flag
    "Is_Anomaly",
    # Room details
    "Dorm_Enc", "Room_Enc", "RoomSize_Enc", "Num_Occupants",
    # Appliances
    *APPLIANCE_COLS,
    # Actual appliance load
    "Appliance_kWh_Active",
]

df = df.sort_values("Timestamp")
X = df[FEATURE_COLS]
y = df["Electricity_Consumed"]

train_size = int(len(df) * 0.7)
X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
X_test,  y_test  = X.iloc[train_size:], y.iloc[train_size:]

# ── Multiple Linear Regression ───────────────────────────────────────────────
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_mae  = mean_absolute_error(y_test, lr_pred)
lr_rmse = float(np.sqrt(mean_squared_error(y_test, lr_pred)))
lr_r2   = r2_score(y_test, lr_pred)
lr_cv   = cross_val_score(lr, X, y, cv=5, scoring="r2", n_jobs=-1).mean()

print("=== Multiple Linear Regression ===")
print(f"MAE : {lr_mae:.4f}")
print(f"RMSE: {lr_rmse:.4f}")
print(f"R²  : {lr_r2:.4f}")
print(f"CV R²: {lr_cv:.4f}")
print("\nFeature Coefficients:")
for f, c in zip(FEATURE_COLS, lr.coef_):
    print(f"  {f}: {c:.4f}")

# ── Random Forest ────────────────────────────────────────────────────────────
rf = RandomForestRegressor(n_estimators=30, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_pred)))
rf_r2   = r2_score(y_test, rf_pred)
rf_cv   = cross_val_score(rf, X, y, cv=5, scoring="r2", n_jobs=-1).mean()

print("\n=== Random Forest ===")
print(f"MAE : {rf_mae:.4f}")
print(f"RMSE: {rf_rmse:.4f}")
print(f"R²  : {rf_r2:.4f}")
print(f"CV R²: {rf_cv:.4f}")

print("\n=== Feature Importances (RF) ===")
for f, imp in sorted(zip(FEATURE_COLS, rf.feature_importances_),
                     key=lambda x: x[1], reverse=True):
    print(f"  {f:<28} {imp:.4f}")

# ── Model comparison ─────────────────────────────────────────────────────────
print("\n=== Model Comparison ===")
print(f"{'Metric':<10} {'Linear Reg':>12} {'Random Forest':>14}")
print(f"{'MAE':<10} {lr_mae:>12.4f} {rf_mae:>14.4f}")
print(f"{'RMSE':<10} {lr_rmse:>12.4f} {rf_rmse:>14.4f}")
print(f"{'R²':<10} {lr_r2:>12.4f} {rf_r2:>14.4f}")
print(f"{'CV R²':<10} {lr_cv:>12.4f} {rf_cv:>14.4f}")

# ── Save model ───────────────────────────────────────────────────────────────
joblib.dump(rf, "electricity_model.pkl")
print("\nRandom Forest model saved to electricity_model.pkl")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, pred, title in zip(axes, [lr_pred, rf_pred],
                            ["Linear Regression", "Random Forest"]):
    ax.plot(y_test.values[:100], label="Actual")
    ax.plot(pred[:100], label="Predicted", linestyle="--")
    ax.set_title(f"Actual vs Predicted — {title}")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Consumption")
    ax.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved to actual_vs_predicted.png")
