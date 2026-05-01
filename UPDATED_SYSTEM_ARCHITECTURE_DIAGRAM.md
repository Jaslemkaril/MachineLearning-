# Updated System Architecture Diagrams

## 3.3.1 Overall Architecture (Updated)

```
┌─────────────────────────────────────────────────────────────┐
│                         USER                                │
│                  (Web Browser Interface)                    │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTPS
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    RENDER CLOUD PLATFORM                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         Flask Web Application (Python)               │  │
│  │  - Input validation (22 features)                    │  │
│  │  - Feature engineering (IsWeekend, TimeOfDay)        │  │
│  │  - Model inference (Random Forest)                   │  │
│  │  - Response formatting (JSON)                        │  │
│  └─────────────────────┬─────────────────────────────────┘  │
│                        │                                     │
│  ┌─────────────────────▼─────────────────────────────────┐  │
│  │         Machine Learning Models (.pkl)               │  │
│  │  - electricity_model.pkl (Random Forest Regressor)   │  │
│  │  - electricity_classifier.pkl (RF Classifier)        │  │
│  │  - Accuracy: 92.03%                                  │  │
│  └─────────────────────┬─────────────────────────────────┘  │
│                        │                                     │
│  ┌─────────────────────▼─────────────────────────────────┐  │
│  │              Data Storage (Files)                    │  │
│  │  - smart_meter_data.csv (2,089 records)              │  │
│  │  - room_config.json (24 rooms)                       │  │
│  │  - stats_cache.json (model statistics)               │  │
│  │  - prediction_history.json (user history)            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3.3.2 ML Pipeline Architecture (Updated)

```
┌─────────────────────────────────────────────────────────────┐
│              Smart Meter Dataset                            │
│         (Time-Series Consumption Data)                      │
│  - 2,089 records from March-April 2024                      │
│  - 8 rooms across 3 dormitories                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Preprocessing                             │
│  - Cleaning: Remove inconsistencies                         │
│  - Missing Values: Forward/backward fill (291 values)       │
│  - Datetime Conversion: Parse timestamps                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Extraction                             │
│  Temporal (4): Hour, Day, IsWeekend, TimeOfDay              │
│  Appliances (9): AC, Kettle, Rice Cooker, Fan, etc.        │
│  Room (4): Dorm_Enc, Room_Enc, RoomSize_Enc, Occupants     │
│  Historical (1): Avg_Past_Consumption                       │
│  Anomaly (1): Is_Anomaly                                    │
│  ────────────────────────────────────────────────────       │
│  EXCLUDED: Month, Season (insufficient data: 1.45 months)   │
│  EXCLUDED: Temperature, Humidity, Wind (data quality)       │
│  EXCLUDED: Appliance_kWh_Active (data leakage)              │
│  ────────────────────────────────────────────────────       │
│  TOTAL: 22 Features                                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Engineering                            │
│  - Label Encoding: Dorm (0-2), Room (0-7), Size (0-2)      │
│  - Binary Encoding: IsWeekend (0/1), Appliances (0/1)      │
│  - Categorical: TimeOfDay (0-3), Is_Anomaly (0/1)          │
│  - Normalization: Pre-normalized to 0-1 scale               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Splitting                                 │
│  Training Set: 70% (1,462 records) - March 1-31            │
│  Testing Set:  30% (627 records)   - April 1-14            │
│  Method: Temporal split (chronological)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Training & Comparison                    │
│  ┌─────────────────┬─────────────────┬──────────────────┐  │
│  │ Random Forest   │    XGBoost      │      SVM         │  │
│  │ Accuracy: 92.03%│ Accuracy: 91.23%│ Accuracy: 80.38% │  │
│  │ Recall: 0.88    │ Recall: 0.94    │ Recall: 0.74     │  │
│  │ Precision: 0.82 │ Precision: 0.77 │ Precision: 0.60  │  │
│  │ ⭐ BEST MODEL   │ High Recall     │ Baseline         │  │
│  └─────────────────┴─────────────────┴──────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Evaluation                               │
│  - Confusion Matrix Analysis                                │
│  - Cross-Validation (5-fold): R² = 0.96                     │
│  - Feature Importance: AC (56%), Kettle (35%), Rice (4%)   │
│  - Classification Metrics: Precision, Recall, F1-Score      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Persistence                              │
│  - electricity_model.pkl (Random Forest Regressor)          │
│  - electricity_classifier.pkl (Random Forest Classifier)    │
│  - Serialized using joblib                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Deployment (Render)                            │
│  - Gunicorn WSGI Server                                     │
│  - Automatic deployment from GitHub                         │
│  - Health monitoring (/health endpoint)                     │
│  - HTTPS/SSL enabled                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3.3.3 Web Application Flow (Updated)

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE                           │
│              (HTML5 + CSS3 + JavaScript)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ User Input (22 Features)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  INPUT FORM                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Appliances (9):                                      │  │
│  │  ☐ Air Conditioner    ☐ Electric Kettle             │  │
│  │  ☐ Rice Cooker        ☐ Electric Fan                │  │
│  │  ☐ Laptop/PC          ☐ Refrigerator                │  │
│  │  ☐ TV/Monitor         ☐ Phone Charger               │  │
│  │  ☐ Study Lamp                                        │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │ Temporal (4):                                        │  │
│  │  Hour: [0-23]         Day: [1-31]                    │  │
│  │  (IsWeekend & TimeOfDay auto-calculated)             │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │ Room (4):                                            │  │
│  │  Dorm: [A/B/C]        Room: [1-8]                    │  │
│  │  (Size & Occupancy auto-filled from config)          │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │ Historical (1):                                      │  │
│  │  Avg Past Consumption: [0.0 - 1.0] slider           │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │ Anomaly (1):                                         │  │
│  │  Is_Anomaly: Auto-set to 0 (normal)                 │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ HTTP POST
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              FLASK BACKEND (app.py)                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Input Validation                                  │  │
│  │    - Validate 22 feature values                      │  │
│  │    - Check ranges and data types                     │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │ 2. Feature Engineering                               │  │
│  │    - Calculate IsWeekend from Day                    │  │
│  │    - Calculate TimeOfDay from Hour                   │  │
│  │    - Encode Dorm, Room, Size                         │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │ 3. Model Inference                                   │  │
│  │    - Load electricity_classifier.pkl                 │  │
│  │    - Create feature vector (22 features)             │  │
│  │    - Predict: model.predict(features)                │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │ 4. Post-Processing                                   │  │
│  │    - Denormalize: value × 2.0 = kWh                  │  │
│  │    - Calculate cost: kWh × ₱10.50                    │  │
│  │    - Classify: Normal or High Consumption            │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │ 5. Response Formatting                               │  │
│  │    - JSON response with prediction                   │  │
│  │    - Save to prediction_history.json                 │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ HTTP Response (JSON)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  PREDICTION RESULT                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Prediction: 0.65 (normalized)                        │  │
│  │ Status: ⚠️ High Consumption                          │  │
│  │ Estimated kWh: 1.30 kWh per 30-min slot              │  │
│  │ Estimated Cost: ₱13.65                               │  │
│  │                                                       │  │
│  │ Appliances Active: AC, Kettle, Rice Cooker           │  │
│  │ Time: 20:00 · Day 15 · Mar                           │  │
│  │ Room: Dorm A - Room 3 (Medium, 2 occupants)          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  Prediction History (Last 5):                               │
│  1. 0.65 - High - ₱13.65 - 20:00 Day 15                    │
│  2. 0.45 - Normal - ₱9.45 - 18:00 Day 14                   │
│  3. 0.38 - Normal - ₱7.98 - 12:00 Day 14                   │
│  ...                                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3.3.4 Feature Flow Diagram (22 Features)

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW INPUT DATA                           │
└──────────┬──────────────────────────────────────────────────┘
           │
           ├─────────────────────────────────────────────────┐
           │                                                 │
           ▼                                                 ▼
┌──────────────────────┐                    ┌────────────────────────┐
│  TEMPORAL FEATURES   │                    │  APPLIANCE FEATURES    │
│  (4 features)        │                    │  (9 features)          │
├──────────────────────┤                    ├────────────────────────┤
│ • Hour (0-23)        │                    │ • App_Air_Conditioner  │
│ • Day (1-31)         │                    │ • App_Electric_Kettle  │
│ • IsWeekend (0/1)    │◄─── Derived        │ • App_Rice_Cooker      │
│ • TimeOfDay (0-3)    │     from Hour/Day  │ • App_Electric_Fan     │
└──────────┬───────────┘                    │ • App_Laptop_PC        │
           │                                │ • App_Refrigerator     │
           │                                │ • App_TV_Monitor       │
           │                                │ • App_Phone_Charger    │
           │                                │ • App_Study_Lamp       │
           │                                └────────────┬───────────┘
           │                                             │
           ├─────────────────────────────────────────────┤
           │                                             │
           ▼                                             ▼
┌──────────────────────┐                    ┌────────────────────────┐
│  ROOM FEATURES       │                    │  HISTORICAL FEATURE    │
│  (4 features)        │                    │  (1 feature)           │
├──────────────────────┤                    ├────────────────────────┤
│ • Dorm_Enc (0-2)     │◄─── Encoded        │ • Avg_Past_Consumption │
│ • Room_Enc (0-7)     │     from           │   (0.0-1.0)            │
│ • RoomSize_Enc (0-2) │     selection      │                        │
│ • Num_Occupants (1-4)│                    │ 24-hour rolling avg    │
└──────────┬───────────┘                    └────────────┬───────────┘
           │                                             │
           ├─────────────────────────────────────────────┤
           │                                             │
           ▼                                             ▼
┌──────────────────────┐                    ┌────────────────────────┐
│  ANOMALY FEATURE     │                    │   EXCLUDED FEATURES    │
│  (1 feature)         │                    │   (Not in model)       │
├──────────────────────┤                    ├────────────────────────┤
│ • Is_Anomaly (0/1)   │                    │ ✗ Month                │
│                      │                    │ ✗ Season               │
│ Auto-set to 0        │                    │ ✗ Temperature          │
│ (normal prediction)  │                    │ ✗ Humidity             │
└──────────┬───────────┘                    │ ✗ Wind_Speed           │
           │                                │ ✗ Appliance_kWh_Active │
           │                                └────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│              FEATURE VECTOR (22 features)                   │
│  [Hour, Day, IsWeekend, TimeOfDay, App1, App2, ..., App9,  │
│   Dorm_Enc, Room_Enc, RoomSize_Enc, Num_Occupants,         │
│   Avg_Past_Consumption, Is_Anomaly]                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              RANDOM FOREST MODEL                            │
│  - 100 decision trees                                       │
│  - Trained on 1,462 samples                                 │
│  - Accuracy: 92.03%                                         │
│  - Recall: 0.88 (catches 88% of high consumption)           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PREDICTION OUTPUT                              │
│  - Normalized value (0.0-1.0)                               │
│  - Classification (Normal/High)                             │
│  - kWh estimation (value × 2.0)                             │
│  - Cost calculation (kWh × ₱10.50)                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Updates in These Diagrams:

✅ **22 features** (not 24)
✅ **Removed features** clearly marked (Month, Season, environmental)
✅ **Random Forest** as best model (92.03%)
✅ **File-based storage** (not MySQL)
✅ **Render deployment** platform
✅ **Updated accuracy metrics**
✅ **Feature importance** (AC: 56%, Kettle: 35%, Rice: 4%)
✅ **Temporal split** (70-30)
✅ **Health endpoint** (/health)

---

## How to Use These Diagrams:

1. **Copy the ASCII diagrams** into your Google Doc
2. **Or recreate them** using Google Drawings/Lucidchart
3. **Use the structure** as a template for visual diagrams
4. **Include in Section 3.3** (System Architecture)

These diagrams accurately reflect your current implementation with 22 features and 92.03% accuracy!
