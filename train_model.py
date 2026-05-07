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

from config import FEATURE_COLS, APPLIANCE_COLS, APPLIANCE_WATT_COLS, DORM_MAP, SIZE_MAP

# ── Load & prepare ───────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("smart_meter_data.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Handle missing values
print(f"Missing values: {df.isnull().sum().sum()}")
for col in FEATURE_COLS:
    if col in df.columns and df[col].isnull().any():
        df[col] = df[col].ffill().bfill().fillna(df[col].mean())

print(f"Records: {len(df):,}")
print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
print(f"Months: {df['Month'].nunique()}")

# ── Verify all feature columns exist ─────────────────────────────────────────
missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

df = df.sort_values("Timestamp")
X = df[FEATURE_COLS]
y = df["Electricity_Consumed"]

train_size = int(len(df) * 0.7)
X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
X_test,  y_test  = X.iloc[train_size:], y.iloc[train_size:]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Features: {len(FEATURE_COLS)}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4.1: REGRESSION — Multiple Linear Regression (Baseline) vs Random Forest
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SECTION 4.1: REGRESSION MODEL COMPARISON")
print("="*80)

# ── Multiple Linear Regression (Baseline) ────────────────────────────────────
print("\n--- Training Multiple Linear Regression (Baseline) ---")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_mae  = mean_absolute_error(y_test, lr_pred)
lr_rmse = float(np.sqrt(mean_squared_error(y_test, lr_pred)))
lr_r2   = r2_score(y_test, lr_pred)
lr_cv   = float(cross_val_score(lr, X, y, cv=5, scoring="r2", n_jobs=-1).mean())

print(f"MAE : {lr_mae:.4f}")
print(f"RMSE: {lr_rmse:.4f}")
print(f"R²  : {lr_r2:.4f}")
print(f"CV R²: {lr_cv:.4f}")
print("\nFeature Coefficients (top 10 by magnitude):")
coef_pairs = sorted(zip(FEATURE_COLS, lr.coef_), key=lambda x: abs(x[1]), reverse=True)
for f, c in coef_pairs[:10]:
    print(f"  {f:<30} {c:+.6f}")

# ── Random Forest Regressor (Primary Model) ──────────────────────────────────
print("\n--- Training Random Forest Regressor (Primary Model) ---")
rf = RandomForestRegressor(n_estimators=150, max_depth=20, random_state=42,
                           min_samples_leaf=5, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_pred)))
rf_r2   = r2_score(y_test, rf_pred)
rf_cv   = float(cross_val_score(rf, X, y, cv=5, scoring="r2", n_jobs=-1).mean())

print(f"MAE : {rf_mae:.4f}")
print(f"RMSE: {rf_rmse:.4f}")
print(f"R²  : {rf_r2:.4f}")
print(f"CV R²: {rf_cv:.4f}")

print("\n=== Feature Importances (Random Forest) ===")
imp_pairs = sorted(zip(FEATURE_COLS, rf.feature_importances_),
                   key=lambda x: x[1], reverse=True)
for f, imp in imp_pairs:
    print(f"  {f:<30} {imp:.4f}")

# ── Model Comparison Table ────────────────────────────────────────────────────
print("\n" + "="*80)
print("REGRESSION MODEL COMPARISON TABLE")
print("="*80)
print(f"\n{'Metric':<10} {'Linear Regression':>18} {'Random Forest':>14}")
print("-"*44)
print(f"{'MAE':<10} {lr_mae:>18.4f} {rf_mae:>14.4f}")
print(f"{'RMSE':<10} {lr_rmse:>18.4f} {rf_rmse:>14.4f}")
print(f"{'R²':<10} {lr_r2:>18.4f} {rf_r2:>14.4f}")
print(f"{'CV R²':<10} {lr_cv:>18.4f} {rf_cv:>14.4f}")

improvement_r2 = ((rf_r2 - lr_r2) / max(abs(lr_r2), 0.001)) * 100
print(f"\nRandom Forest R² improvement over Linear Regression: {improvement_r2:+.1f}%")

# ── Save regression model ────────────────────────────────────────────────────
joblib.dump(rf, "electricity_model.pkl")
joblib.dump(lr, "electricity_model_lr.pkl")
print("\nModels saved:")
print("  - electricity_model.pkl (Random Forest - primary)")
print("  - electricity_model_lr.pkl (Linear Regression - baseline)")

# ── Plot regression results ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, pred, title in zip(axes, [lr_pred, rf_pred],
                            ["Linear Regression (Baseline)", "Random Forest (Primary)"]):
    ax.plot(y_test.values[:100], label="Actual", linewidth=1.5)
    ax.plot(pred[:100], label="Predicted", linestyle="--", linewidth=1.5)
    ax.set_title(f"Actual vs Predicted — {title}")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Consumption")
    ax.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved to actual_vs_predicted.png")

# ── Residual plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, pred, title in zip(axes, [lr_pred, rf_pred],
                            ["Linear Regression", "Random Forest"]):
    residuals = y_test.values - pred
    ax.scatter(pred, residuals, alpha=0.3, s=10)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.set_title(f"Residuals — {title}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
plt.tight_layout()
plt.savefig("residual_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved to residual_plot.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4.2: CLASSIFICATION — High Consumption Detection
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SECTION 4.2: CLASSIFICATION MODEL COMPARISON")
print("="*80)

# Create binary classification target: High consumption (above 75th percentile)
consumption_threshold = df["Electricity_Consumed"].quantile(0.75)
df["High_Consumption"] = (df["Electricity_Consumed"] > consumption_threshold).astype(int)

# Prepare classification data
X_clf = df[FEATURE_COLS]
y_clf = df["High_Consumption"]

# Split using same indices as regression
X_train_clf, y_train_clf = X_clf.iloc[:train_size], y_clf.iloc[:train_size]
X_test_clf, y_test_clf = X_clf.iloc[train_size:], y_clf.iloc[train_size:]

print(f"\nClassification Task: Predicting High Consumption (>{consumption_threshold:.4f})")
print(f"Class distribution in training set:")
print(f"  Normal (0): {(y_train_clf == 0).sum()} ({(y_train_clf == 0).mean()*100:.1f}%)")
print(f"  High (1):   {(y_train_clf == 1).sum()} ({(y_train_clf == 1).mean()*100:.1f}%)")

# ── Model 1: Random Forest Classifier ────────────────────────────────────────
print("\n--- Training Random Forest Classifier ---")
rf_clf = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42,
                                 n_jobs=-1, class_weight='balanced')
rf_clf.fit(X_train_clf, y_train_clf)
rf_clf_pred = rf_clf.predict(X_test_clf)

# ── Model 2: Support Vector Machine (SVM) ────────────────────────────────────
print("--- Training Support Vector Machine (SVM) ---")
svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42,
              class_weight='balanced')
svm_clf.fit(X_train_clf, y_train_clf)
svm_clf_pred = svm_clf.predict(X_test_clf)

# ── Model 3: XGBoost ─────────────────────────────────────────────────────────
if XGBOOST_AVAILABLE:
    print("--- Training XGBoost Classifier ---")
    scale_pos_weight = (y_train_clf == 0).sum() / max((y_train_clf == 1).sum(), 1)
    xgb_clf = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                           random_state=42, scale_pos_weight=scale_pos_weight,
                           eval_metric='logloss')
    xgb_clf.fit(X_train_clf, y_train_clf)
    xgb_clf_pred = xgb_clf.predict(X_test_clf)

# ── Calculate Metrics for Each Model ─────────────────────────────────────────
def calculate_classification_metrics(y_true, y_pred, model_name):
    """Calculate precision, recall, F1-score, and accuracy for both classes"""
    cm = confusion_matrix(y_true, y_pred)

    precision_0 = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall_0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_0 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    precision_1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

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
results.append(calculate_classification_metrics(y_test_clf, svm_clf_pred, "SVM"))
if XGBOOST_AVAILABLE:
    results.append(calculate_classification_metrics(y_test_clf, xgb_clf_pred, "XGBoost"))

# ── Table: Comparative Performance Metrics ────────────────────────────────────
print("\n" + "="*80)
print("CLASSIFICATION PERFORMANCE COMPARISON")
print("="*80)

print(f"\n{'Model':<20} {'Class':<12} {'Precision':<11} {'Recall':<11} {'F1':<11} {'Accuracy':<11}")
print("-"*76)

for result in results:
    model_name = result['model']
    print(f"{model_name:<20} {'Normal (0)':<12} {result['normal_precision']:<11.4f} "
          f"{result['normal_recall']:<11.4f} {result['normal_f1']:<11.4f} {result['accuracy']*100:<10.2f}%")
    print(f"{'':<20} {'High (1)':<12} {result['high_precision']:<11.4f} "
          f"{result['high_recall']:<11.4f} {result['high_f1']:<11.4f}")
    print()

# ── Confusion Matrices Plot ───────────────────────────────────────────────────
num_models = len(results)
fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 5))
if num_models == 1:
    axes = [axes]

for idx, result in enumerate(results):
    cm = result['confusion_matrix']
    model_name = result['model']

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal (0)', 'High (1)'],
                yticklabels=['Normal (0)', 'High (1)'],
                ax=axes[idx], cbar=True, square=True)

    axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=10)
    axes[idx].set_xlabel('Predicted Label', fontsize=10)

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("Confusion matrices saved to confusion_matrices.png")

# ── Detailed Classification Reports ──────────────────────────────────────────
print("\n--- Random Forest ---")
print(classification_report(y_test_clf, rf_clf_pred,
                           target_names=['Normal (0)', 'High (1)']))

print("--- SVM ---")
print(classification_report(y_test_clf, svm_clf_pred,
                           target_names=['Normal (0)', 'High (1)']))

if XGBOOST_AVAILABLE:
    print("--- XGBoost ---")
    print(classification_report(y_test_clf, xgb_clf_pred,
                               target_names=['Normal (0)', 'High (1)']))

# ── Save best classification model ───────────────────────────────────────────
best_model_idx = np.argmax([r['accuracy'] for r in results])
best_model_name = results[best_model_idx]['model']
best_accuracy = results[best_model_idx]['accuracy']

if best_model_name == "Random Forest":
    joblib.dump(rf_clf, "electricity_classifier.pkl")
elif best_model_name == "SVM":
    joblib.dump(svm_clf, "electricity_classifier.pkl")
elif XGBOOST_AVAILABLE and best_model_name == "XGBoost":
    joblib.dump(xgb_clf, "electricity_classifier.pkl")

print(f"\nBest classifier: {best_model_name} (Accuracy: {best_accuracy*100:.2f}%)")
print(f"Saved to electricity_classifier.pkl")

print("\n" + "="*80)
print("ALL TRAINING COMPLETE")
print("="*80)
