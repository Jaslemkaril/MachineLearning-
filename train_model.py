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

# Load dataset
df = pd.read_csv("smart_meter_data.csv")

# Convert timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Extract time features
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.day
df['Month'] = df['Timestamp'].dt.month

# Additional engineered features
df['IsWeekend']  = df['Timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
df['Season']     = df['Month'].map({12:0,1:0,2:0, 3:1,4:1,5:1,
                                     6:2,7:2,8:2, 9:3,10:3,11:3})
df['TimeOfDay']  = pd.cut(df['Hour'], bins=[-1,5,11,17,23],
                           labels=[0,1,2,3]).astype(int)

# Encode Anomaly_Label
df['Is_Anomaly'] = (df['Anomaly_Label'] != 'Normal').astype(int)

# Synthetic Dorm_ID and Room_ID (3 dorms, 10 rooms each)
np.random.seed(42)
df['Dorm_ID'] = np.random.randint(0, 3, size=len(df))
df['Room_ID'] = np.random.randint(0, 10, size=len(df))

# Define features and target
feature_cols = ['Temperature', 'Humidity', 'Wind_Speed', 'Avg_Past_Consumption',
                'Hour', 'Day', 'Month', 'IsWeekend', 'Season', 'TimeOfDay', 'Is_Anomaly',
                'Dorm_ID', 'Room_ID']
X = df[feature_cols]
y = df['Electricity_Consumed']

# Time-based split (correct for forecasting)
df = df.sort_values("Timestamp")
train_size = int(len(df) * 0.7)
train = df[:train_size]
test  = df[train_size:]

X_train = train[feature_cols]
y_train = train["Electricity_Consumed"]
X_test  = test[feature_cols]
y_test  = test["Electricity_Consumed"]

# ── Multiple Linear Regression ──────────────────────────────────────────────
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_mae  = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2   = r2_score(y_test, lr_pred)
lr_cv   = cross_val_score(lr, X, y, cv=5, scoring='r2').mean()

print("=== Multiple Linear Regression ===")
print("MAE :", lr_mae)
print("RMSE:", lr_rmse)
print("R²  :", lr_r2)
print("CV R²:", lr_cv)

# Feature coefficients
print("\nFeature Coefficients:")
for f, c in zip(feature_cols, lr.coef_):
    print(f"  {f}: {round(c, 4)}")

# ── Random Forest ────────────────────────────────────────────────────────────
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2   = r2_score(y_test, rf_pred)

# Save RF model immediately after training (CV skipped — too slow locally)
joblib.dump(rf, "electricity_model.pkl")
print("\nRandom Forest model saved to electricity_model.pkl")

print("\n=== Random Forest ===")
print("MAE :", rf_mae)
print("RMSE:", rf_rmse)
print("R²  :", rf_r2)

# ── Cross Validation summary ────────────────────────────────────────────────
print("\n=== Model Comparison ===")
print(f"{'Metric':<10} {'Linear Reg':>12} {'Random Forest':>14}")
print(f"{'MAE':<10} {lr_mae:>12.4f} {rf_mae:>14.4f}")
print(f"{'RMSE':<10} {lr_rmse:>12.4f} {rf_rmse:>14.4f}")
print(f"{'R²':<10} {lr_r2:>12.4f} {rf_r2:>14.4f}")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, pred, title in zip(axes, [lr_pred, rf_pred],
                            ['Linear Regression', 'Random Forest']):
    ax.plot(y_test.values[:100], label='Actual')
    ax.plot(pred[:100], label='Predicted', linestyle='--')
    ax.set_title(f'Actual vs Predicted — {title}')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Consumption')
    ax.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved to actual_vs_predicted.png")