# 🏢 Dorm & Room Electricity Monitoring and Forecasting System

**ZAMCELCO Smart Meter Data Analysis**  
*Zamboanga City Electric Cooperative*

---

## 📊 Project Overview

A machine learning-powered web application for predicting electricity consumption in dormitory rooms using smart meter data from ZAMCELCO (Zamboanga City Electric Cooperative).

**Dataset Period:** March 1 - April 14, 2024 (1.45 months)  
**Location:** Zamboanga City, Philippines  
**Model Accuracy:** 97.32% (R² Score)

---

## ✨ Features

- 🔮 **Real-time Consumption Prediction** - Forecast electricity usage for 30-minute intervals
- 📊 **Interactive Dashboard** - Modern web interface with real-time visualizations
- ⚡ **Appliance Monitoring** - Track 9 different appliance types with custom wattages
- 🌡️ **Environmental Factors** - Considers temperature, humidity, and wind speed
- ⚠️ **Anomaly Detection** - Identifies high consumption patterns
- 💰 **Cost Estimation** - Calculates costs in Philippine Pesos (₱10.50/kWh)
- 📈 **Performance Metrics** - Real-time model performance dashboard
- 🏠 **Multi-Room Support** - 8 rooms across 3 dormitories

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd MachineLearning
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Open in browser**
```
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
MachineLearning/
├── app.py                          # Flask web application
├── train_model.py                  # Model training script
├── precompute_stats.py             # Statistics pre-computation
├── test_simulation.py              # Testing & simulation script
├── requirements.txt                # Python dependencies
├── Procfile                        # Render deployment config
├── render.yaml                     # Render service config
├── smart_meter_data.csv            # ZAMCELCO dataset (March-April 2024)
├── room_config.json                # Room configurations
├── electricity_model.pkl           # Trained Random Forest model
├── stats_cache.json                # Pre-computed statistics
├── prediction_history.json         # Recent predictions
├── actual_vs_predicted.png         # Model performance chart
├── ZAMCELCO_DATA_SUMMARY.md        # Complete documentation
├── README.md                       # This file
└── templates/
    └── index.html                  # Web interface template
```

---

## 🎯 Usage

### Making Predictions

1. **Select Location**
   - Choose dorm (A, B, or C)
   - Select room (1-8)

2. **Enter Environmental Conditions**
   - Temperature (normalized 0-1)
   - Humidity (normalized 0-1)
   - Wind Speed (normalized 0-1)
   - Average Past Consumption

3. **Set Time**
   - Hour (0-23)
   - Day (1-31)
   - Month (1-12)

4. **Select Active Appliances**
   - Electric Fan (35W)
   - Air Conditioner (1200W)
   - Laptop/PC (65W)
   - Refrigerator (120W)
   - TV/Monitor (40W)
   - Phone Charger (20W)
   - Electric Kettle (1500W)
   - Rice Cooker (400W)
   - Study Lamp (9W)

5. **Get Prediction**
   - Consumption in kWh
   - Cost in Philippine Pesos
   - Status (Normal/High)

---

## 🔧 Model Training

To retrain the model with new data:

```bash
# 1. Update smart_meter_data.csv with new data

# 2. Train the model
python train_model.py

# 3. Pre-compute statistics
python precompute_stats.py

# 4. Test the model
python test_simulation.py
```

---

## 📊 Dataset Information

**Source:** ZAMCELCO Smart Meters  
**Location:** Zamboanga City, Philippines  
**Period:** March 1 - April 14, 2024  
**Records:** 2,089 readings  
**Sampling:** ~30 minutes  

**Features:**
- Environmental: Temperature, Humidity, Wind Speed
- Temporal: Hour, Day, Month, Weekend, Season
- Room: Dorm ID, Room ID, Size, Occupants
- Appliances: 9 appliance types + total kWh
- Historical: Average past consumption

**Characteristics:**
- Tropical climate (24-33°C)
- High humidity (70-90%)
- Power grid brownouts included
- Realistic sensor imperfections
- Missing values handled

---

## 🎓 Model Performance

| Metric | Value |
|--------|-------|
| **R² Score** | 0.9732 (97.32%) |
| **MAE** | 0.0199 |
| **RMSE** | 0.0402 |
| **Cross-Validation R²** | 0.9697 |
| **Model Type** | Random Forest Regressor |

---

## 🌐 Deployment

### Deploy to Render

1. **Push to GitHub**
```bash
git add .
git commit -m "Deploy to Render"
git push origin main
```

2. **Connect to Render**
   - Go to [render.com](https://render.com)
   - Create new Web Service
   - Connect your GitHub repository
   - Render will auto-detect `render.yaml`

3. **Deploy**
   - Render will automatically deploy
   - Your app will be live at: `https://your-app.onrender.com`

### Environment Variables
No environment variables required for basic deployment.

---

## 🧪 Testing

Run the simulation test:

```bash
python test_simulation.py
```

This will test 11 realistic scenarios including:
- Early morning (minimal usage)
- Morning rush (getting ready)
- Midday with AC
- Study sessions
- Cooking dinner
- Entertainment
- Late night studying
- Maximum load test
- Cool weather
- Room size comparisons

---

## 📈 API Endpoints

### Health Check
```
GET /health
```
Returns model status and statistics.

### Home / Prediction
```
GET /
POST /
```
Main interface for making predictions.

---

## 🔍 Data Characteristics

**Real-World Features:**
- ✅ Timestamp irregularities (346 unique intervals)
- ✅ Missing values (291 sensor gaps)
- ✅ Sensor noise (±1-3% error)
- ✅ Power fluctuations (16 brownout events)
- ✅ Data loss (31 records removed)
- ✅ Measurement outliers (10 sensor spikes)

**Philippines-Specific:**
- ✅ Zamboanga tropical climate
- ✅ ZAMCELCO power grid characteristics
- ✅ Brownout handling
- ✅ Coastal humidity patterns

---

## 📝 License

This project is for academic and educational purposes.

---

## 👥 Contributors

- Your Name/Team Name
- Institution: [Your University]
- Course: [Your Course]
- Year: 2024

---

## 📞 Support

For questions or issues:
- Check `ZAMCELCO_DATA_SUMMARY.md` for detailed documentation
- Review model performance in `stats_cache.json`
- Test with `test_simulation.py`

---

## 🎯 Future Enhancements

- [ ] User authentication system
- [ ] Historical analytics dashboard
- [ ] Email alerts for high consumption
- [ ] Mobile application
- [ ] REST API for external integrations
- [ ] Database integration
- [ ] Real-time smart meter connection
- [ ] 12+ months seasonal data

---

## ✅ Project Status

**Status:** ✅ Production Ready  
**Last Updated:** April 29, 2026  
**Model Version:** 1.0  
**Data Version:** ZAMCELCO March-April 2024

---

**Built with ❤️ for ZAMCELCO and Zamboanga City**
