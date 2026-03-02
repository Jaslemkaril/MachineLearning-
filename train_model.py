import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
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

# Define features and target
X = df[['Temperature', 'Humidity', 'Wind_Speed',
        'Avg_Past_Consumption', 'Hour', 'Day', 'Month']]

y = df['Electricity_Consumed']

# Time-based split (correct for forecasting)
df = df.sort_values("Timestamp")
train_size = int(len(df) * 0.7)
train = df[:train_size]
test = df[train_size:]

X_train = train[X.columns]
y_train = train["Electricity_Consumed"]
X_test = test[X.columns]
y_test = test["Electricity_Consumed"]

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

# Feature importance
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print("\nFeature Coefficients:")
print(coefficients)

# Cross Validation
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("\nCross Validation R²:", scores.mean())

# Plot Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Electricity Consumption")
plt.xlabel("Samples")
plt.ylabel("Electricity Consumption")
plt.show()

# Save model
joblib.dump(model, "electricity_model.pkl")
print("\nModel saved to electricity_model.pkl")