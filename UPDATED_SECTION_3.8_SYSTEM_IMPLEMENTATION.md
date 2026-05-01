# 3.8 System Implementation

The forecasting system developed in this study is implemented as a functional software application that integrates the trained machine learning model with a user-accessible interface. The implementation follows a modular architecture that separates data processing, model inference, backend logic, and frontend presentation into distinct components.

---

## 3.8.1 Programming Language

Python 3.x is the primary programming language used throughout the system. Python is widely adopted in the machine learning and data science community due to its extensive library ecosystem, readable syntax, and strong community support. It is used for all stages of the pipeline, including data preprocessing, feature engineering, model training, evaluation, and web application development.

---

## 3.8.2 Machine Learning Framework

**Scikit-learn** is the core machine learning framework used in this study. It provides robust and well-documented implementations of:

- **Multiple Linear Regression** (LinearRegression) for baseline regression modeling
- **Random Forest** (RandomForestRegressor, RandomForestClassifier) for ensemble learning
- **Support Vector Machine** (SVC) for classification tasks
- **Evaluation metrics** including MAE, MSE, RMSE, R², confusion matrix, precision, recall, and F1-score
- **Cross-validation** utilities for model validation

**XGBoost** is additionally used as an advanced gradient boosting framework, providing:
- **XGBClassifier** for high-performance classification
- Built-in regularization to prevent overfitting
- Handling of class imbalance through scale_pos_weight parameter

Scikit-learn's consistent API makes it suitable for comparing multiple candidate models in a standardized manner, while XGBoost provides state-of-the-art performance for classification tasks.

---

## 3.8.3 Data Processing Libraries

**Pandas** is used for loading, cleaning, and transforming the smart meter dataset. It supports:
- CSV file reading and datetime parsing
- Handling of missing values through forward fill, backward fill, and mean imputation
- Feature extraction operations (hour, day, weekend detection, time-of-day categorization)
- Data aggregation and grouping for statistical analysis

**NumPy** complements Pandas by providing efficient numerical computations required during preprocessing, feature engineering, and model training operations.

---

## 3.8.4 Data Visualization

**Matplotlib** and **Seaborn** are used to generate visualizations throughout the analysis. These include:

- **Time-series plots** of electricity consumption trends by hour, day, and month
- **Confusion matrices** for classification model evaluation
- **Feature importance charts** showing the contribution of each feature to model predictions
- **Comparison charts** of actual versus predicted consumption values
- **Distribution plots** for consumption patterns and anomaly detection
- **Correlation heatmaps** for feature selection analysis

Visual outputs support the interpretation and communication of model results, with 12 presentation-ready charts generated for stakeholder communication.

---

## 3.8.5 Backend Technology

**Flask**, a lightweight Python web framework, is used to develop the backend API of the forecasting system. The Flask application serves as the middleware between the trained machine learning model and the frontend user interface. 

**Key functionalities include:**

1. **Model Loading**: Loads pre-trained Random Forest models (electricity_model.pkl, electricity_classifier.pkl) using joblib
2. **Input Validation**: Validates user input for 22 features including appliance states, temporal features, room characteristics, historical consumption, and anomaly flags
3. **Feature Engineering**: Derives additional features (IsWeekend, TimeOfDay, Season) from user input
4. **Prediction**: Applies the trained model to generate electricity consumption forecasts
5. **Response Formatting**: Returns predicted consumption values, classification status (Normal/High), estimated kWh, and cost in Philippine Peso
6. **Prediction History**: Maintains a rolling history of the last 5 predictions for user reference
7. **Health Monitoring**: Provides a /health endpoint for deployment monitoring

The Flask application is production-ready and deployed on **Render**, a cloud platform that provides automatic deployment from GitHub, SSL certificates, and continuous integration.

---

## 3.8.6 Frontend Interface

The frontend is built using **HTML5**, **CSS3**, and **JavaScript**. It provides an intuitive and accessible dashboard where users can:

**Input Features:**
- **Appliance states**: Toggle switches for 9 appliances (AC, kettle, rice cooker, fan, laptop, fridge, TV, charger, lamp)
- **Custom wattage**: Adjustable power ratings for each appliance
- **Temporal features**: Hour (0-23), Day (1-31)
- **Room characteristics**: Dormitory selection (Dorm A/B/C), Room selection (1-8), auto-filled size and occupancy
- **Historical consumption**: Slider input for average past consumption (0.0-1.0)

**Output Display:**
- **Prediction result**: Normalized consumption value (0.0-1.0)
- **Classification status**: Normal or High Consumption with color-coded badges
- **Estimated kWh**: Actual kilowatt-hour consumption per 30-minute slot
- **Estimated cost**: Cost in Philippine Peso (₱) based on ZAMCELCO rates (₱10.50/kWh)
- **Prediction history**: Last 5 predictions with timestamps and details
- **Model statistics**: MAE, RMSE, R², and cross-validation scores

**Design Features:**
- **Responsive layout**: Adapts to desktop, tablet, and mobile devices
- **Real-time feedback**: Instant updates as users adjust inputs
- **Visual indicators**: Color-coded status badges, progress bars, and charts
- **Accessibility**: ARIA labels, keyboard navigation, and screen reader support

The interface is designed for clarity and ease of use, reflecting the practical application objectives of this study.

---

## 3.8.7 Model Persistence

Trained models are serialized and stored using **joblib**, a Python library optimized for saving large NumPy arrays efficiently. The following model files are persisted:

- **electricity_model.pkl**: Random Forest regression model (97.32% R² score)
- **electricity_classifier.pkl**: Random Forest classification model (92.03% accuracy)

These files are loaded at application startup, enabling fast inference without retraining. Model versioning is managed through Git, with each model update tracked in the repository history.

---

## 3.8.8 Data Storage

**Development Phase:**
The smart meter dataset is stored and retrieved from **CSV files** (smart_meter_data.csv) during development and testing. This approach provides:
- Simple file-based storage for rapid prototyping
- Easy version control through Git
- Portability across development environments

**Configuration Storage:**
- **room_config.json**: Room characteristics (size, occupancy) for 24 rooms across 3 dormitories
- **stats_cache.json**: Pre-computed model statistics for fast application startup
- **prediction_history.json**: Persistent storage of user prediction history

**Production Deployment:**
For the current deployment on Render, the system uses **file-based storage** with JSON for configuration and history. This lightweight approach is suitable for the pilot study scale (2,089 records, 8 rooms).

**Future Scalability:**
For larger-scale deployment, the system architecture supports integration with relational databases such as **PostgreSQL** or **MySQL** to manage:
- Historical consumption records
- User accounts and authentication
- Prediction logs and analytics
- Real-time smart meter data ingestion

---

## 3.8.9 Deployment Platform

The system is deployed on **Render** (https://render.com), a cloud platform that provides:

- **Automatic deployment**: Continuous deployment from GitHub repository
- **HTTPS/SSL**: Automatic SSL certificate provisioning
- **Health monitoring**: Automatic restart on failure
- **Environment variables**: Secure configuration management
- **Build automation**: Automatic dependency installation from requirements.txt
- **Zero-downtime deployment**: Rolling updates without service interruption

**Deployment Configuration:**
- **Procfile**: Specifies the web server command (Gunicorn)
- **render.yaml**: Defines service configuration and build settings
- **requirements.txt**: Lists all Python dependencies

The deployment process is fully automated: pushing code to the main branch triggers automatic build, test, and deployment to production.

---

## 3.8.10 System Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (HTML/CSS/JS)            │
│  - Input form (22 features)                                 │
│  - Prediction display (kWh, cost, status)                   │
│  - History panel (last 5 predictions)                       │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP POST
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Flask Backend (Python)                     │
│  - Input validation                                         │
│  - Feature engineering (IsWeekend, TimeOfDay)               │
│  - Model inference (Random Forest)                          │
│  - Response formatting (JSON)                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Machine Learning Models                        │
│  - electricity_model.pkl (Random Forest Regressor)          │
│  - electricity_classifier.pkl (Random Forest Classifier)    │
│  - Trained on 22 features, 2,089 records                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Storage                              │
│  - smart_meter_data.csv (training data)                     │
│  - room_config.json (room characteristics)                  │
│  - stats_cache.json (model statistics)                      │
│  - prediction_history.json (user history)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3.8.11 Technology Stack Summary

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Programming Language** | Python 3.x | Core development language |
| **ML Framework** | Scikit-learn, XGBoost | Model training and inference |
| **Data Processing** | Pandas, NumPy | Data manipulation and computation |
| **Visualization** | Matplotlib, Seaborn | Chart generation |
| **Web Framework** | Flask | Backend API development |
| **Frontend** | HTML5, CSS3, JavaScript | User interface |
| **Model Persistence** | Joblib | Model serialization |
| **Data Storage** | CSV, JSON | File-based storage |
| **Deployment** | Render | Cloud hosting platform |
| **Version Control** | Git, GitHub | Code management and CI/CD |
| **Web Server** | Gunicorn | Production WSGI server |

---

## 3.8.12 Key Implementation Decisions

**1. Feature Count Optimization**
- Reduced from 24 to 22 features by removing Month and Season (insufficient data range: 1.45 months)
- Removed environmental features (Temperature, Humidity, Wind_Speed) due to data quality concerns
- Removed Appliance_kWh_Active to prevent data leakage

**2. Model Selection**
- Random Forest chosen as primary model (92.03% accuracy, balanced performance)
- XGBoost available as alternative (91.23% accuracy, higher recall: 94%)
- SVM included as baseline (80.38% accuracy)

**3. Deployment Strategy**
- Cloud-based deployment on Render for accessibility and scalability
- Automatic deployment from GitHub for continuous integration
- File-based storage for pilot study, with database-ready architecture for future scaling

**4. User Experience**
- Simplified input form (removed month field after feature optimization)
- Real-time feedback and visual indicators
- Prediction history for user reference
- Mobile-responsive design for accessibility

This implementation demonstrates a complete end-to-end machine learning system suitable for real-world deployment in dormitory electricity management scenarios.
