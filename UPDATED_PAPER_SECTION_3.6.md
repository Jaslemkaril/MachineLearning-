# 3.6 Feature Engineering

## Overview
A comprehensive feature engineering process was conducted to transform raw smart meter data into meaningful predictors for electricity consumption classification. The final feature set comprises **22 features** across five distinct categories, designed to capture temporal patterns, appliance usage, room characteristics, historical consumption trends, and anomaly detection.

---

## Extracted Features

### **Temporal Features (4 features)**

**Hour** – Represents the specific hour of the day (0–23) and captures intra-day consumption patterns, such as peak usage during evening hours (18:00–23:00) when occupants are present and active. This is the most critical temporal feature for predicting consumption.

**Day** – Represents the day of the month (1–31) and captures short-term temporal variation in electricity usage, such as patterns related to academic schedules, exam periods, or billing cycles.

**IsWeekend** – A binary feature (1 = weekend, 0 = weekday) that distinguishes days when occupants are more likely to remain in their rooms for extended periods, typically resulting in higher electricity consumption due to increased appliance usage.

**TimeOfDay** – A categorical feature derived from the hour (0 = Night [0–5], 1 = Morning [6–11], 2 = Afternoon [12–17], 3 = Evening [18–23]) that groups hours into meaningful consumption periods aligned with typical dormitory occupancy patterns.

---

### **Appliance Usage Features (9 features)**

Binary indicators (0 = off, 1 = on) representing the operational status of major electrical appliances:

**App_Electric_Fan** – Cooling appliance with moderate power consumption, commonly used in tropical climates.

**App_Air_Conditioner** – Primary cooling appliance and highest power consumer, accounting for 56% of feature importance in the predictive model.

**App_Laptop_PC** – Computing device with moderate consumption, essential for student activities.

**App_Refrigerator** – Continuous operation appliance for food storage.

**App_TV_Monitor** – Entertainment and display device.

**App_Phone_Charger** – Low power consumption device for mobile device charging.

**App_Electric_Kettle** – High wattage heating appliance, accounting for 35% of feature importance, second only to air conditioning.

**App_Rice_Cooker** – Cooking appliance with moderate to high consumption, accounting for 4% of feature importance.

**App_Study_Lamp** – Lighting device with low consumption for study activities.

---

### **Room Characteristics Features (4 features)**

**Dorm_Enc** – Label-encoded identifier representing the dormitory building (Dorm A = 0, Dorm B = 1, Dorm C = 2), allowing the model to learn building-level consumption differences due to infrastructure variations, insulation quality, and location-specific factors.

**Room_Enc** – Label-encoded identifier representing the specific room within a dormitory (Room 101–108 encoded as 0–7), enabling the model to capture room-level consumption variation based on orientation, window exposure, and individual usage patterns.

**RoomSize_Enc** – Categorical encoding of room size (Small = 0, Medium = 1, Large = 2), reflecting the relationship between room area and cooling/heating requirements.

**Num_Occupants** – Number of occupants in the room (1–4), directly influencing appliance usage frequency and overall consumption levels.

---

### **Historical Consumption Feature (1 feature)**

**Avg_Past_Consumption** – Represents the average historical electricity demand over previous time periods, capturing temporal dependency and consumption trends. This feature enables the model to learn from past behavior patterns and detect deviations from normal usage.

---

### **Anomaly Detection Feature (1 feature)**

**Is_Anomaly** – A binary feature (1 = anomalous, 0 = normal) derived from the Anomaly_Label column in the dataset, indicating whether a recorded consumption value was flagged as abnormal based on statistical thresholds. This feature helps the model distinguish between normal consumption patterns and unusual events such as power fluctuations, brownouts, or equipment malfunctions.

---

## Feature Selection and Exclusion

### **Excluded Features and Rationale**

**Environmental Features (Temperature, Humidity, Wind_Speed)** – Initially considered but excluded due to:
1. Data quality concerns – measurements exhibited normalized ranges (0–1) that may not represent actual sensor readings
2. Temporal stability requirements – environmental conditions require continuous real-time monitoring
3. Model performance – appliance-based features and temporal patterns provided sufficient predictive power (92.03% accuracy)
4. Practical deployment – excluding environmental features simplifies system architecture

**Temporal Features (Month, Season)** – Excluded due to:
1. Insufficient data range – dataset spans only 1.45 months (March 1 - April 14, 2024)
2. No seasonal variation – both months fall within the same season
3. Tropical climate – Zamboanga City exhibits minimal seasonal temperature variation
4. Noise vs. signal – preliminary testing showed these features added noise rather than predictive value

**Data Leakage Prevention (Appliance_kWh_Active)** – Deliberately removed to avoid data leakage, as it represents a direct measurement of the target variable. Real-world prediction systems must rely on appliance operational states rather than actual consumption measurements.

---

## Feature Importance Analysis

Feature importance rankings from Random Forest and XGBoost models revealed:

1. **App_Air_Conditioner** – 56% (dominant predictor)
2. **App_Electric_Kettle** – 35% (secondary predictor)
3. **App_Rice_Cooker** – 4% (tertiary predictor)
4. **Other features** – 5% (combined contribution)

The top three appliance features account for **95% of predictive power**, confirming that high-wattage appliances are the primary drivers of electricity consumption in dormitory settings.

---

## Data Preprocessing

### **Missing Value Handling**
The dataset contained 291 missing values (2.5%) due to realistic smart meter imperfections such as sensor communication errors and network outages. Missing values were handled using:
- Forward fill – propagate last valid observation forward
- Backward fill – propagate next valid observation backward
- Mean imputation – fill remaining gaps with feature mean

### **Class Imbalance Handling**
The classification task exhibits class imbalance (75% normal consumption, 25% high consumption). Addressed through:
1. Balanced class weights applied to all classifiers
2. Scale_pos_weight parameter in XGBoost
3. Stratified sampling in cross-validation

### **Feature Scaling**
Feature scaling was not required as tree-based models (Random Forest, XGBoost) are invariant to feature scales. For SVM, scikit-learn's default standardization was applied internally.

---

## Final Feature Set Summary

| Category | Features | Count |
|----------|----------|-------|
| **Temporal** | Hour, Day, IsWeekend, TimeOfDay | 4 |
| **Appliances** | 9 appliance binary indicators | 9 |
| **Room Characteristics** | Dorm_Enc, Room_Enc, RoomSize_Enc, Num_Occupants | 4 |
| **Historical** | Avg_Past_Consumption | 1 |
| **Anomaly** | Is_Anomaly | 1 |
| **TOTAL** | | **22** |

### **Excluded Features**
- Temperature, Humidity, Wind_Speed (data quality concerns)
- Month, Season (insufficient data range: 1.45 months)
- Appliance_kWh_Active (data leakage prevention)

---

## Conclusion

This feature engineering approach balances predictive power, interpretability, and practical deployment considerations. The resulting 22-feature model achieves 92.03% accuracy with Random Forest, demonstrating that a focused set of high-quality features outperforms a larger set containing noisy or redundant variables. The model is suitable for real-world electricity consumption prediction in dormitory environments.
