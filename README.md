# Electricity Consumption Prediction System

A machine learning-based system for predicting electricity consumption in dormitory environments using smart meter data.

## Project Overview

This system uses Random Forest classification to predict electricity consumption patterns in dormitory rooms, achieving 92.03% accuracy. The application is deployed as a web interface where users can input room characteristics and appliance usage to receive consumption predictions.

## Features

- **Machine Learning Models**: Comparison of Random Forest, XGBoost, and SVM classifiers
- **22 Optimized Features**: Temporal patterns, appliance usage, room characteristics, historical data
- **Web Interface**: User-friendly Flask application for real-time predictions
- **Visualization**: Comprehensive charts for model performance and consumption patterns
- **Cloud Deployment**: Hosted on Render with automatic GitHub integration

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **92.03%** | 0.82 | 0.88 | 0.85 |
| XGBoost | 91.23% | 0.77 | 0.94 | 0.85 |
| SVM | 80.38% | 0.60 | 0.74 | 0.66 |

## Dataset

- **Source**: Synthetic smart meter data based on ZAMCELCO (Zamboanga City Electric Cooperative) parameters
- **Records**: 2,089 readings
- **Period**: March 1 - April 14, 2024 (1.45 months)
- **Coverage**: 8 dormitory rooms across 3 buildings
- **Sampling**: 30-minute intervals

## Technology Stack

- **Backend**: Python, Flask
- **ML Libraries**: Scikit-learn, XGBoost, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3, JavaScript
- **Deployment**: Render, Gunicorn
- **Version Control**: Git, GitHub

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

3. Run the application:
```bash
python app.py
```

4. Open browser to `http://localhost:5000`

## Project Structure

```
├── app.py                          # Flask web application
├── train_model.py                  # Model training script
├── precompute_stats.py            # Statistics pre-computation
├── generate_presentation_charts.py # Chart generation
├── smart_meter_data.csv           # Training dataset
├── room_config.json               # Room configurations
├── electricity_model.pkl          # Trained regression model
├── electricity_classifier.pkl     # Trained classification model
├── templates/
│   └── index.html                 # Web interface
├── presentation_charts/           # Generated visualizations
├── requirements.txt               # Python dependencies
├── Procfile                       # Render deployment config
└── render.yaml                    # Render service config
```

## Usage

### Training Models

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train Random Forest, XGBoost, and SVM models
- Generate confusion matrices and performance metrics
- Save trained models as `.pkl` files

### Generating Charts

```bash
python generate_presentation_charts.py
```

Creates 12 visualization charts including:
- Model accuracy comparison
- Feature importance
- Consumption patterns by hour, dorm, room size
- Appliance usage distribution

### Running Web Application

```bash
python app.py
```

Access the interface at `http://localhost:5000` to:
- Select dormitory and room
- Input appliance usage
- Specify temporal parameters
- Receive consumption predictions with cost estimates

## Model Features (22 Total)

### Temporal (4)
- Hour (0-23)
- Day (1-31)
- IsWeekend (0/1)
- TimeOfDay (0-3)

### Appliances (9)
- Electric Fan, Air Conditioner, Laptop/PC
- Refrigerator, TV/Monitor, Phone Charger
- Electric Kettle, Rice Cooker, Study Lamp

### Room Characteristics (4)
- Dorm ID, Room ID, Room Size, Number of Occupants

### Historical (1)
- Average Past Consumption

### Anomaly Detection (1)
- Anomaly Flag

## Deployment

The application is deployed on Render:
- **URL**: https://smartmeter-forecast.onrender.com
- **Auto-deployment**: Triggered on push to main branch
- **Health monitoring**: `/health` endpoint

## Results

- **Best Model**: Random Forest (92.03% accuracy)
- **Top Features**: Air Conditioner (56%), Electric Kettle (35%), Rice Cooker (4%)
- **Recall**: 88% (catches 88% of high consumption events)
- **Precision**: 82% (82% of high predictions are correct)

## License

This project is developed for academic purposes.

## Contact

For questions or issues, please contact the development team.
