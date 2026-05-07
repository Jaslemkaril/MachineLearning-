# Electricity Consumption Prediction & Monitoring System

A machine learning-based web application for predicting and monitoring electricity consumption in dormitory environments. Developed as a thesis project for Zamboanga City (ZAMCELCO area).

## Project Overview

This system uses **Multiple Regression** (Random Forest Regressor as primary, Linear Regression as baseline) for continuous consumption prediction and **Classification** (Random Forest, SVM, XGBoost) for high-consumption detection. The application provides a full-featured dashboard with real-time predictions, monthly estimates, interactive charts, and room management.

## Features

- **Dual ML Models**: Random Forest Regressor (primary) vs Linear Regression (baseline) for consumption prediction
- **Classification**: High consumption detection using RF/SVM/XGBoost classifiers
- **Wattage-Based Features**: Per-appliance wattage with adjustable quantity (1-10 per type)
- **Monthly Prediction**: ML-driven 30-day consumption estimate (model runs across all 24 hours)
- **Interactive Dashboard**: Chart.js visualizations — monthly, hourly, dorm comparison
- **Dynamic Room Management**: Add rooms, auto-fill occupancy from room config
- **Auto-Derived History**: Avg Past Consumption computed from prediction history (no manual input)
- **Cost Estimation**: Per-slot and monthly cost using ZAMCELCO rates (₱10.50/kWh)

## Regression Model Performance

| Metric | Linear Regression (Baseline) | Random Forest (Primary) |
|--------|------------------------------|-------------------------|
| MAE    | 0.0278                       | **0.0238**              |
| RMSE   | 0.0387                       | **0.0369**              |
| R²     | 0.882                        | **0.8925**              |
| CV R²  | 0.8864                       | **0.889**               |

## Dataset

- **Source**: Synthetically generated based on real appliance specs, observed dorm usage patterns, and ZAMCELCO rates
- **Records**: 8,000 readings
- **Period**: 8 months (extended temporal coverage)
- **Coverage**: 24 rooms across 3 dormitory buildings
- **Sampling**: 30-minute intervals
- **Generator**: `generate_dataset.py` (reproducible with noise)

## Technology Stack

- **Backend**: Python 3.10, Flask 3.x
- **ML**: Scikit-learn, XGBoost, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Chart.js
- **Frontend**: HTML5, CSS3 (custom dark theme), JavaScript
- **Deployment**: Render, Gunicorn
- **Config**: Centralized `config.py` for all constants

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Jaslemkaril/MachineLearning-.git
cd MachineLearning-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate dataset (optional — CSV already included):
```bash
python generate_dataset.py
```

4. Train models (optional — .pkl files already included):
```bash
python train_model.py
```

5. Pre-compute stats for dashboard:
```bash
python precompute_stats.py
```

6. Run the application:
```bash
python app.py
```

7. Open browser to `http://localhost:5000`

## Project Structure

```
├── config.py                       # Centralized constants & feature definitions
├── app.py                          # Flask web application (routes, prediction logic)
├── train_model.py                  # Model training (RF, LR, classifiers)
├── generate_dataset.py             # Synthetic dataset generator (8k records)
├── precompute_stats.py             # Pre-compute dashboard statistics
├── smart_meter_data.csv            # Generated training dataset
├── room_config.json                # Room configurations (editable via UI)
├── prediction_history.json         # Prediction history (auto-managed)
├── stats_cache.json                # Cached model metrics & chart data
├── electricity_model.pkl           # Trained RF regressor
├── electricity_model_lr.pkl        # Trained LR baseline
├── electricity_classifier.pkl      # Trained classifier (best of RF/SVM/XGB)
├── templates/
│   └── index.html                  # Web interface (tabbed: Predict/Dashboard/Rooms)
├── requirements.txt                # Python dependencies
├── Procfile                        # Render deployment config
└── render.yaml                     # Render service config
```

## Usage

### Web Interface (3 Tabs)

1. **Predict Tab**: Select dorm/room, time (hour/day/month), toggle appliances with quantity & wattage. Get per-slot and monthly consumption predictions.
2. **Dashboard Tab**: View model metrics, classification results, Chart.js charts, feature importances, top consuming rooms, and prediction history.
3. **Manage Rooms Tab**: Add new rooms with size category, view all rooms per dorm.

### Training Pipeline

```bash
python generate_dataset.py   # 1. Generate 8k-row dataset
python train_model.py        # 2. Train all models + evaluation
python precompute_stats.py   # 3. Cache stats for dashboard
python app.py                # 4. Launch web app
```

## Feature Set (22 Features)

| Category | Features |
|----------|----------|
| **Historical** | Avg_Past_Consumption (auto-derived) |
| **Temporal** | Hour, Day, Month, IsWeekend, TimeOfDay |
| **Room Context** | Dorm_Enc, Room_Enc, RoomSize_Enc, Num_Occupants |
| **Appliance Aggregates** | Total_Active_Wattage, Num_Active_Appliances, Has_High_Power_Appliance |
| **Per-Appliance Wattage** | 9 features (W×qty if on, 0 if off) |

## Key Design Decisions

- **No temperature/humidity**: Removed per professor's feedback (synthetic/normalized data)
- **Fixed appliance types**: Only wattage and quantity are editable, not appliance categories
- **Multiple Regression + Baseline**: Random Forest vs Linear Regression for model comparison
- **Wattage × Quantity**: Supports multiple units of same appliance (e.g., 3 fans)
- **Monthly estimate**: Model-based (not simple multiplication) — runs 24 hourly predictions

## Deployment

The application is deployed on Render:
- **Auto-deployment**: Triggered on push to main branch
- **Health monitoring**: `/health` endpoint
- **WSGI**: Gunicorn

## License

This project is developed for academic purposes (thesis).

## Contact

For questions or issues, please contact the development team.
