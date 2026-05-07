"""
Generate comprehensive data visualization charts for presentation
Creates multiple PNG files showing different aspects of the electricity forecasting system
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import json

print("=" * 80)
print("GENERATING PRESENTATION CHARTS")
print("=" * 80)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
print("\n[1] Loading data...")
df = pd.read_csv('smart_meter_data.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
model = joblib.load('electricity_model.pkl')

with open('stats_cache.json', 'r') as f:
    stats = json.load(f)

print(f"✓ Loaded {len(df):,} records")

# Create output directory for charts
import os
if not os.path.exists('presentation_charts'):
    os.makedirs('presentation_charts')

# ═══════════════════════════════════════════════════════════════════════════
# CHART 1: Model Accuracy Comparison
# ═══════════════════════════════════════════════════════════════════════════
print("\n[2] Creating Model Accuracy Comparison chart...")

fig, ax = plt.subplots(figsize=(10, 6))
models = ['Random Forest', 'XGBoost', 'SVM']
accuracies = [92.03, 91.23, 80.38]  # Updated with 22-feature model results
colors = ['#2E7D32', '#558B2F', '#00838F']

bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Machine Learning Model', fontsize=14, fontweight='bold')
ax.set_title('Overall Accuracy Comparison by Model', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('presentation_charts/1_model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 1_model_accuracy_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 2: Model Performance Metrics
# ═══════════════════════════════════════════════════════════════════════════
print("\n[3] Creating Model Performance Metrics chart...")

fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['MAE', 'RMSE', 'R² Score', 'CV Score']
values = [stats['mae'], stats['rmse'], stats['r2'], stats['cv']]
colors = ['#1976D2', '#388E3C', '#D32F2F', '#F57C00']

bars = ax.barh(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(0, 1.0)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for bar, val in zip(bars, values):
    width = bar.get_width()
    ax.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
            f'{val:.4f}', ha='left', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('presentation_charts/2_model_performance_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 2_model_performance_metrics.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 3: Consumption by Hour of Day
# ═══════════════════════════════════════════════════════════════════════════
print("\n[4] Creating Consumption by Hour chart...")

df['Hour'] = df['Timestamp'].dt.hour
hourly_consumption = df.groupby('Hour')['Electricity_Consumed'].mean()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(hourly_consumption.index, hourly_consumption.values, marker='o', 
        linewidth=2.5, markersize=8, color='#1976D2')
ax.fill_between(hourly_consumption.index, hourly_consumption.values, alpha=0.3, color='#1976D2')
ax.set_xlabel('Hour of Day', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Consumption (normalized)', fontsize=14, fontweight='bold')
ax.set_title('Electricity Consumption Pattern by Hour of Day', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(range(0, 24, 2))
ax.grid(True, alpha=0.3)

# Highlight peak hours
peak_hour = hourly_consumption.idxmax()
ax.axvline(x=peak_hour, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Peak Hour: {peak_hour}:00')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('presentation_charts/3_consumption_by_hour.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 3_consumption_by_hour.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 4: Consumption by Dorm
# ═══════════════════════════════════════════════════════════════════════════
print("\n[5] Creating Consumption by Dorm chart...")

dorm_consumption = df.groupby('Dorm_ID')['Electricity_Consumed'].agg(['mean', 'std'])

fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(dorm_consumption))
bars = ax.bar(x, dorm_consumption['mean'], yerr=dorm_consumption['std'],
              color=['#E53935', '#43A047', '#1E88E5'], edgecolor='black', 
              linewidth=1.5, capsize=5, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(dorm_consumption.index, fontsize=12)
ax.set_ylabel('Average Consumption (normalized)', fontsize=14, fontweight='bold')
ax.set_xlabel('Dormitory', fontsize=14, fontweight='bold')
ax.set_title('Average Electricity Consumption by Dormitory', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, mean_val in zip(bars, dorm_consumption['mean']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{mean_val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('presentation_charts/4_consumption_by_dorm.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 4_consumption_by_dorm.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 5: Consumption by Room Size
# ═══════════════════════════════════════════════════════════════════════════
print("\n[6] Creating Consumption by Room Size chart...")

room_size_consumption = df.groupby('Room_Size_Cat')['Electricity_Consumed'].mean().sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
colors_map = {'Small': '#FFA726', 'Medium': '#66BB6A', 'Large': '#42A5F5'}
colors = [colors_map[cat] for cat in room_size_consumption.index]

bars = ax.bar(room_size_consumption.index, room_size_consumption.values, 
              color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Average Consumption (normalized)', fontsize=14, fontweight='bold')
ax.set_xlabel('Room Size Category', fontsize=14, fontweight='bold')
ax.set_title('Electricity Consumption by Room Size', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, room_size_consumption.values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('presentation_charts/5_consumption_by_room_size.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 5_consumption_by_room_size.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 6: Appliance Usage Distribution
# ═══════════════════════════════════════════════════════════════════════════
print("\n[7] Creating Appliance Usage Distribution chart...")

appliance_cols = [col for col in df.columns if col.startswith('App_')]
appliance_usage = {}
for col in appliance_cols:
    name = col.replace('App_', '').replace('_', ' ')
    usage_pct = (df[col].sum() / len(df)) * 100
    appliance_usage[name] = usage_pct

appliance_usage = dict(sorted(appliance_usage.items(), key=lambda x: x[1], reverse=True))

fig, ax = plt.subplots(figsize=(12, 7))
colors = plt.cm.Set3(range(len(appliance_usage)))
bars = ax.barh(list(appliance_usage.keys()), list(appliance_usage.values()), 
               color=colors, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Usage Percentage (%)', fontsize=14, fontweight='bold')
ax.set_title('Appliance Usage Distribution', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for bar, val in zip(bars, appliance_usage.values()):
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
            f'{val:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('presentation_charts/6_appliance_usage_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 6_appliance_usage_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 7: Normal vs Anomaly Distribution
# ═══════════════════════════════════════════════════════════════════════════
print("\n[8] Creating Normal vs Anomaly Distribution chart...")

anomaly_counts = df['Anomaly_Label'].value_counts()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart
colors_pie = ['#4CAF50', '#F44336']
explode = (0.05, 0.05)
wedges, texts, autotexts = ax1.pie(anomaly_counts.values, labels=anomaly_counts.index,
                                     autopct='%1.1f%%', startangle=90, colors=colors_pie,
                                     explode=explode, shadow=True, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax1.set_title('Consumption Pattern Distribution', fontsize=14, fontweight='bold', pad=20)

# Bar chart
bars = ax2.bar(anomaly_counts.index, anomaly_counts.values, color=colors_pie, 
               edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Number of Records', fontsize=12, fontweight='bold')
ax2.set_title('Normal vs Abnormal Consumption Count', fontsize=14, fontweight='bold', pad=20)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, anomaly_counts.values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 20,
            f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('presentation_charts/7_normal_vs_anomaly.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 7_normal_vs_anomaly.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 8: Feature Importance (Top 10)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[9] Creating Feature Importance chart...")

importances_df = pd.DataFrame(stats['importances'])
top_10 = importances_df.nlargest(10, 'coef')

fig, ax = plt.subplots(figsize=(12, 7))
colors = plt.cm.viridis(np.linspace(0, 1, len(top_10)))
bars = ax.barh(top_10['feature'], top_10['coef'], color=colors, 
               edgecolor='black', linewidth=1.5)
ax.set_xlabel('Importance Score', fontsize=14, fontweight='bold')
ax.set_title('Top 10 Most Important Features for Prediction', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for bar, val in zip(bars, top_10['coef']):
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
            f'{val:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('presentation_charts/8_feature_importance_top10.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 8_feature_importance_top10.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 9: Consumption Distribution Histogram
# ═══════════════════════════════════════════════════════════════════════════
print("\n[10] Creating Consumption Distribution chart...")

fig, ax = plt.subplots(figsize=(12, 6))
n, bins, patches = ax.hist(df['Electricity_Consumed'], bins=50, color='#2196F3', 
                            edgecolor='black', linewidth=0.5, alpha=0.7)

# Color gradient
cm = plt.cm.RdYlGn_r
for i, patch in enumerate(patches):
    patch.set_facecolor(cm(i / len(patches)))

ax.set_xlabel('Electricity Consumption (normalized)', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
ax.set_title('Distribution of Electricity Consumption', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

# Add mean and median lines
mean_val = df['Electricity_Consumed'].mean()
median_val = df['Electricity_Consumed'].median()
ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('presentation_charts/9_consumption_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 9_consumption_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 10: Monthly Consumption Trend
# ═══════════════════════════════════════════════════════════════════════════
print("\n[11] Creating Monthly Consumption Trend chart...")

df['Month'] = df['Timestamp'].dt.month
monthly_consumption = df.groupby('Month')['Electricity_Consumed'].agg(['mean', 'std', 'count'])

fig, ax = plt.subplots(figsize=(12, 6))
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_labels = [months[m-1] for m in monthly_consumption.index]

ax.plot(month_labels, monthly_consumption['mean'], marker='o', linewidth=2.5, 
        markersize=10, color='#FF5722', label='Average Consumption')
ax.fill_between(range(len(month_labels)), 
                monthly_consumption['mean'] - monthly_consumption['std'],
                monthly_consumption['mean'] + monthly_consumption['std'],
                alpha=0.3, color='#FF5722', label='±1 Std Dev')

ax.set_xlabel('Month', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Consumption (normalized)', fontsize=14, fontweight='bold')
ax.set_title('Monthly Electricity Consumption Trend (March-April 2024)', fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('presentation_charts/10_monthly_consumption_trend.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 10_monthly_consumption_trend.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 11: Weekday vs Weekend Comparison
# ═══════════════════════════════════════════════════════════════════════════
print("\n[12] Creating Weekday vs Weekend Comparison chart...")

df['IsWeekend'] = df['Timestamp'].dt.dayofweek.isin([5, 6])
weekend_comparison = df.groupby('IsWeekend')['Electricity_Consumed'].agg(['mean', 'std'])

fig, ax = plt.subplots(figsize=(10, 6))
labels = ['Weekday', 'Weekend']
means = weekend_comparison['mean'].values
stds = weekend_comparison['std'].values
colors = ['#3F51B5', '#FF9800']

bars = ax.bar(labels, means, yerr=stds, color=colors, edgecolor='black', 
              linewidth=1.5, capsize=5, alpha=0.8)
ax.set_ylabel('Average Consumption (normalized)', fontsize=14, fontweight='bold')
ax.set_title('Weekday vs Weekend Electricity Consumption', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, mean_val in zip(bars, means):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{mean_val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('presentation_charts/11_weekday_vs_weekend.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 11_weekday_vs_weekend.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 12: Temperature vs Consumption Correlation
# ═══════════════════════════════════════════════════════════════════════════
print("\n[13] Creating Temperature vs Consumption chart...")

fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(df['Temperature'], df['Electricity_Consumed'], 
                     c=df['Electricity_Consumed'], cmap='coolwarm', 
                     alpha=0.5, s=20, edgecolors='black', linewidth=0.5)

# Add trend line
z = np.polyfit(df['Temperature'].dropna(), df['Electricity_Consumed'][df['Temperature'].notna()], 1)
p = np.poly1d(z)
ax.plot(df['Temperature'].sort_values(), p(df['Temperature'].sort_values()), 
        "r--", linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')

ax.set_xlabel('Temperature (normalized)', fontsize=14, fontweight='bold')
ax.set_ylabel('Electricity Consumption (normalized)', fontsize=14, fontweight='bold')
ax.set_title('Temperature vs Electricity Consumption Correlation', fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Consumption Level', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('presentation_charts/12_temperature_vs_consumption.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 12_temperature_vs_consumption.png")

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("✓ ALL CHARTS GENERATED SUCCESSFULLY!")
print("=" * 80)
print(f"\nTotal charts created: 12")
print(f"Location: presentation_charts/")
print("\nChart List:")
print("  1. Model Accuracy Comparison")
print("  2. Model Performance Metrics")
print("  3. Consumption by Hour of Day")
print("  4. Consumption by Dormitory")
print("  5. Consumption by Room Size")
print("  6. Appliance Usage Distribution")
print("  7. Normal vs Anomaly Distribution")
print("  8. Top 10 Feature Importance")
print("  9. Consumption Distribution Histogram")
print(" 10. Monthly Consumption Trend")
print(" 11. Weekday vs Weekend Comparison")
print(" 12. Temperature vs Consumption Correlation")
print("\n" + "=" * 80)
print("These charts are ready for your presentation!")
print("=" * 80)
