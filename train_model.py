import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             confusion_matrix, classification_report, 
                             precision_score, recall_score, f1_score, accuracy_score)
from sklearn.model_selection import cross_val_score
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

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
df["IsWeekend"] = df["Timestamp"].dt.dayofweek.isin([5, 6]).astype(int)
df["TimeOfDay"] = pd.cut(df["Hour"], bins=[-1,5,11,17,23],
                          labels=[0,1,2,3]).astype(int)
# Month and Season removed: dataset spans only 1.45 months (insufficient for seasonal patterns)

# Encode categorical room fields
df["Dorm_Enc"] = df["Dorm_ID"].map({"Dorm A": 0, "Dorm B": 1, "Dorm C": 2})
room_num = df["Room_ID"].str.extract(r"(\d+)").astype(int)
room_num = room_num.where(room_num < 100, room_num - 100)
df["Room_Enc"] = room_num - 1
df["RoomSize_Enc"] = df["Room_Size_Cat"].map({"Small": 0, "Medium": 1, "Large": 2})

APPLIANCE_COLS = [
    "App_Electric_Fan", "App_Air_Conditioner", "App_Laptop_PC",
    "App_Refrigerator", "App_TV_Monitor", "App_Phone_Charger",
    "App_Electric_Kettle", "App_Rice_Cooker", "App_Study_Lamp",
]

FEATURE_COLS = [
    # Environmental - REMOVED per professor's feedback
    # "Temperature", "Humidity", "Wind_Speed", 
    # Professor's note: "Environmental conditions are not constant and need more analysis"
    # These appear to be normalized/synthetic values (0-1 range) rather than real measurements
    
    "Avg_Past_Consumption",  # Keep this - it's historical consumption data
    
    # Time - Month and Season removed: only 1.45 months of data, insufficient for seasonal patterns
    "Hour", "Day", "IsWeekend", "TimeOfDay",
    # Room details
    "Dorm_Enc", "Room_Enc", "RoomSize_Enc", "Num_Occupants",
    # Appliances
    *APPLIANCE_COLS,
    # NOTE: Removed "Appliance_kWh_Active" to avoid data leakage
    # This feature is essentially the target variable, making prediction too easy
    # Real-world prediction should use appliance states, not actual consumption
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

# ── Save regression model ────────────────────────────────────────────────────
joblib.dump(rf, "electricity_model.pkl")
print("\nRandom Forest regression model saved to electricity_model.pkl")

# ── Plot regression results ──────────────────────────────────────────────────
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

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4.2: Model Comparison and Performance Evaluation (Classification)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SECTION 4.2: MODEL COMPARISON AND PERFORMANCE EVALUATION")
print("="*80)

# Create binary classification target: High consumption (anomaly) prediction
# Similar to paper's "Retained (0)" vs "Dropped Out (1)" classification
# We'll predict if consumption is anomalously high (above 75th percentile)
consumption_threshold = df["Electricity_Consumed"].quantile(0.75)
df["High_Consumption"] = (df["Electricity_Consumed"] > consumption_threshold).astype(int)

# Prepare classification data
X_clf = df[FEATURE_COLS]
y_clf = df["High_Consumption"]

# Split using same indices as regression
X_train_clf, y_train_clf = X_clf.iloc[:train_size], y_clf.iloc[:train_size]
X_test_clf, y_test_clf = X_clf.iloc[train_size:], y_clf.iloc[train_size:]

print(f"\nClassification Task: Predicting High Consumption (>{consumption_threshold:.2f} kWh)")
print(f"Class distribution in training set:")
print(f"  Normal (0): {(y_train_clf == 0).sum()} ({(y_train_clf == 0).mean()*100:.1f}%)")
print(f"  High (1):   {(y_train_clf == 1).sum()} ({(y_train_clf == 1).mean()*100:.1f}%)")

# ── Model 1: Random Forest Classifier ────────────────────────────────────────
print("\n" + "-"*80)
print("Training Random Forest Classifier...")
print("-"*80)
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, 
                                 n_jobs=-1, class_weight='balanced')
rf_clf.fit(X_train_clf, y_train_clf)
rf_clf_pred = rf_clf.predict(X_test_clf)

# ── Model 2: Support Vector Machine (SVM) ────────────────────────────────────
print("\nTraining Support Vector Machine (SVM)...")
print("-"*80)
svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, 
              class_weight='balanced')
svm_clf.fit(X_train_clf, y_train_clf)
svm_clf_pred = svm_clf.predict(X_test_clf)

# ── Model 3: XGBoost ─────────────────────────────────────────────────────────
if XGBOOST_AVAILABLE:
    print("\nTraining XGBoost Classifier...")
    print("-"*80)
    # Calculate scale_pos_weight for imbalanced classes
    scale_pos_weight = (y_train_clf == 0).sum() / (y_train_clf == 1).sum()
    xgb_clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                           random_state=42, scale_pos_weight=scale_pos_weight,
                           eval_metric='logloss')
    xgb_clf.fit(X_train_clf, y_train_clf)
    xgb_clf_pred = xgb_clf.predict(X_test_clf)

# ── Calculate Metrics for Each Model ─────────────────────────────────────────
def calculate_classification_metrics(y_true, y_pred, model_name):
    """Calculate precision, recall, F1-score, and accuracy for both classes"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Metrics for class 0 (Normal/Retained)
    precision_0 = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall_0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_0 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    # Metrics for class 1 (High/Dropped Out)
    precision_1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    return {
        'model': model_name,
        'confusion_matrix': cm,
        'normal_precision': precision_0,
        'normal_recall': recall_0,
        'normal_f1': f1_0,
        'high_precision': precision_1,
        'high_recall': recall_1,
        'high_f1': f1_1,
        'accuracy': accuracy
    }

# Calculate metrics for all models
results = []
results.append(calculate_classification_metrics(y_test_clf, rf_clf_pred, "Random Forest"))
results.append(calculate_classification_metrics(y_test_clf, svm_clf_pred, "Support Vector Machine (SVM)"))
if XGBOOST_AVAILABLE:
    results.append(calculate_classification_metrics(y_test_clf, xgb_clf_pred, "XGBoost"))

# ── Table 1: Comparative Performance Metrics ─────────────────────────────────
print("\n" + "="*80)
print("TABLE 1: Comparative Performance Metrics")
print("(Accuracy, Precision, Recall, F1-Score)")
print("="*80)

# Print table header
print(f"\n{'Machine Learning':<25} {'Target Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Overall':<12}")
print(f"{'Model':<25} {'':<15} {'':<12} {'':<12} {'':<12} {'Accuracy':<12}")
print("-"*88)

# Print results for each model
for result in results:
    model_name = result['model']
    
    # Row 1: Normal/Retained (0)
    print(f"{model_name:<25} {'Normal (0)':<15} {result['normal_precision']:<12.2f} "
          f"{result['normal_recall']:<12.2f} {result['normal_f1']:<12.2f} {result['accuracy']*100:<11.2f}%")
    
    # Row 2: High/Dropped Out (1)
    print(f"{'':<25} {'High (1)':<15} {result['high_precision']:<12.2f} "
          f"{result['high_recall']:<12.2f} {result['high_f1']:<12.2f}")
    print()

# ── Figure 1: Confusion Matrices ─────────────────────────────────────────────
print("\n" + "="*80)
print("FIGURE 1: Confusion Matrices")
print("="*80)

num_models = len(results)
fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 5))
if num_models == 1:
    axes = [axes]

for idx, result in enumerate(results):
    cm = result['confusion_matrix']
    model_name = result['model']
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal (0)', 'High (1)'],
                yticklabels=['Normal (0)', 'High (1)'],
                ax=axes[idx], cbar=True, square=True)
    
    axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=10)
    axes[idx].set_xlabel('Predicted Label', fontsize=10)
    
    # Add TP, TN, FP, FN labels
    tn, fp, fn, tp = cm.ravel()
    axes[idx].text(0.5, 0.5, f'TN={tn}', ha='center', va='center', 
                   fontsize=9, color='darkblue', weight='bold')
    axes[idx].text(1.5, 0.5, f'FP={fp}', ha='center', va='center', 
                   fontsize=9, color='darkred', weight='bold')
    axes[idx].text(0.5, 1.5, f'FN={fn}', ha='center', va='center', 
                   fontsize=9, color='darkred', weight='bold')
    axes[idx].text(1.5, 1.5, f'TP={tp}', ha='center', va='center', 
                   fontsize=9, color='darkblue', weight='bold')

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nConfusion matrices saved to confusion_matrices.png")

# ── Detailed Classification Reports ──────────────────────────────────────────
print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORTS")
print("="*80)

print("\n--- Random Forest ---")
print(classification_report(y_test_clf, rf_clf_pred, 
                           target_names=['Normal (0)', 'High (1)']))

print("\n--- Support Vector Machine (SVM) ---")
print(classification_report(y_test_clf, svm_clf_pred, 
                           target_names=['Normal (0)', 'High (1)']))

if XGBOOST_AVAILABLE:
    print("\n--- XGBoost ---")
    print(classification_report(y_test_clf, xgb_clf_pred, 
                               target_names=['Normal (0)', 'High (1)']))

# ── Key Insights (as per paper methodology) ──────────────────────────────────
print("\n" + "="*80)
print("KEY INSIGHTS (Paper Methodology)")
print("="*80)

print("""
1. RECALL (Sensitivity/True Positive Rate):
   - Measures fraction of positive samples correctly predicted
   - Critical metric for minimizing false negatives
   - High recall = fewer dangerous misclassifications

2. PRECISION:
   - Measures fraction of predictions classified as positive that are actually positive
   - Monitors rate of false positives
   - Ensures educators are not overwhelmed by false alerts

3. ACCURACY:
   - Ratio of correct predictions to total predictions
   - Useful for global overview but affected by class imbalance

4. F1-SCORE:
   - Harmonic mean of precision and recall
   - Maintains balance between sensitivity and precision
   - Ensures model doesn't achieve high sensitivity at cost of false positives
""")

# Save best classification model
best_model_idx = np.argmax([r['accuracy'] for r in results])
best_model_name = results[best_model_idx]['model']
best_accuracy = results[best_model_idx]['accuracy']

if best_model_name == "Random Forest":
    joblib.dump(rf_clf, "electricity_classifier.pkl")
elif best_model_name == "Support Vector Machine (SVM)":
    joblib.dump(svm_clf, "electricity_classifier.pkl")
elif XGBOOST_AVAILABLE and best_model_name == "XGBoost":
    joblib.dump(xgb_clf, "electricity_classifier.pkl")

print(f"\nBest performing model: {best_model_name} (Accuracy: {best_accuracy*100:.2f}%)")
print(f"Classification model saved to electricity_classifier.pkl")

print("\n" + "="*80)
print("MODEL COMPARISON AND EVALUATION COMPLETE")
print("="*80)
