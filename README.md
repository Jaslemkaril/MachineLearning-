# Zamboanga City Dorm Electricity Monitoring System

## 🎯 Project Overview

A machine learning-powered electricity consumption forecasting system for dormitory management in Zamboanga City, using ZAMCELCO electricity data.

**Best Model**: Random Forest (92.50% accuracy)  
**Features**: 21 reliable, measurable features  
**Data Source**: ZAMCELCO Smart Meter Data  

---

## 📊 Final Results

### Model Performance

| Model | Accuracy | Recall | Precision | F1-Score | Status |
|-------|----------|--------|-----------|----------|--------|
| **Random Forest** ⭐ | **92.50%** | 0.90 | 0.83 | 0.86 | Best Model |
| XGBoost | 91.23% | 0.94 | 0.77 | 0.85 | Excellent |
| SVM | 80.38% | 0.74 | 0.60 | 0.66 | Good Baseline |

### Why These Results Are Realistic

✅ **Removed data leakage** (Appliance_kWh_Active)  
✅ **Removed synthetic environmental data** (Temperature, Humidity, Wind_Speed)  
✅ **21 reliable features** (appliances, time, room characteristics)  
✅ **Professor approved** (followed expert feedback)  
✅ **Validated with 3 algorithms** (consistent results)  
✅ **Cross-validation R² = 0.96** (robust performance)  

---

## 🔧 Features Used (21 Total)

### 1. Historical Consumption (1)
- Avg_Past_Consumption

### 2. Temporal Features (6)
- Hour, Day, Month, IsWeekend, Season, TimeOfDay

### 3. Anomaly Detection (1)
- Is_Anomaly

### 4. Room Characteristics (4)
- Dorm_Enc, Room_Enc, RoomSize_Enc, Num_Occupants

### 5. Appliance States (9)
- App_Electric_Fan
- App_Air_Conditioner ⭐ (56% importance)
- App_Laptop_PC
- App_Refrigerator
- App_TV_Monitor
- App_Phone_Charger
- App_Electric_Kettle ⭐ (35% importance)
- App_Rice_Cooker ⭐ (4% importance)
- App_Study_Lamp

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Jaslemkaril/MachineLearning-.git
cd MachineLearning-

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
# Start Flask app
python app.py

# Open browser
# Visit: http://127.0.0.1:5000
```

### Train Model

```bash
# Train all models and generate results
python train_model.py

# Outputs:
# - electricity_model.pkl (regression model)
# - electricity_classifier.pkl (classification model - 92.50%)
# - confusion_matrices.png (model comparison)
# - actual_vs_predicted.png (regression visualization)
```

### Generate Presentation Charts

```bash
# Create all presentation charts
python generate_presentation_charts.py

# Outputs: presentation_charts/ folder with 12 charts
```

---

## 📁 Project Structure

```
MachineLearning-/
├── app.py                          # Flask web application
├── train_model.py                  # Model training script
├── generate_presentation_charts.py # Chart generation
├── precompute_stats.py            # Statistics computation
├── requirements.txt               # Python dependencies
├── Procfile                       # Render deployment config
├── render.yaml                    # Render configuration
│
├── templates/
│   └── index.html                 # Web interface
│
├── presentation_charts/           # Generated charts (12 files)
│   ├── 1_model_accuracy_comparison.png
│   ├── 2_model_performance_metrics.png
│   └── ... (10 more charts)
│
├── Models/
│   ├── electricity_model.pkl      # Regression model
│   └── electricity_classifier.pkl # Classification model (92.50%)
│
├── Data/
│   ├── smart_meter_data.csv       # ZAMCELCO data
│   ├── room_config.json           # Room configurations
│   ├── stats_cache.json           # Precomputed statistics
│   └── prediction_history.json    # Prediction logs
│
└── Documentation/
    ├── START_HERE.md              # Quick start guide
    ├── FINAL_IMPLEMENTATION_STATUS.md
    ├── PROFESSOR_FEEDBACK_ANALYSIS.md
    ├── DEFENSE_CHEAT_SHEET.md
    ├── PAPER_COMPARISON.md
    ├── PAPER_IMPLEMENTATION_GUIDE.md
    ├── MODEL_COMPARISON_METHODOLOGY.md
    ├── REALISTIC_RESULTS_EXPLANATION.md
    └── UPDATED_FINAL_SUMMARY.md
```

---

## 🎓 For Your Paper/Defense

### Key Documentation Files

1. **START_HERE.md** - Quick overview and getting started
2. **DEFENSE_CHEAT_SHEET.md** - Q&A for defense preparation
3. **PROFESSOR_FEEDBACK_ANALYSIS.md** - Why environmental data was removed
4. **PAPER_IMPLEMENTATION_GUIDE.md** - How to write Section 4.2
5. **UPDATED_FINAL_SUMMARY.md** - Complete implementation summary

### Quick Defense Answers

**Q: Why 92.50% accuracy?**
> "We achieved 92.50% using Random Forest, which is realistic for electricity consumption prediction. Following our professor's advice, we removed synthetic environmental data and focused on direct consumption drivers: appliance usage, temporal patterns, and room characteristics. This actually improved our accuracy from 91.39% to 92.50%."

**Q: Why better than classmates (74%)?**
> "Different domains have different predictability. Student dropout (74%) involves complex human behavior. Electricity consumption (92.50%) is a physical system with measurable causes. Our accuracy aligns with energy forecasting literature (85-95% typical)."

**Q: What's your most important metric?**
> "Recall (90%) - we catch 90% of high consumption events. Critical for early warning systems to prevent unexpected costs."

---

## 🔄 Model Evolution

### Version 1: Original (Unrealistic)
- Accuracy: 96.33%
- Problem: Data leakage (Appliance_kWh_Active)
- Status: ❌ Not defensible

### Version 2: After Removing Data Leakage
- Accuracy: 91.55% (XGBoost)
- Fixed: Removed Appliance_kWh_Active
- Issue: ⚠️ Still had synthetic environmental data

### Version 3: After Professor's Feedback (FINAL)
- Accuracy: 92.50% (Random Forest)
- Fixed: Removed synthetic environmental features
- Status: ✅ Fully defensible and improved!

---

## 📊 Methodology

### Section 4.2: Model Comparison and Performance Evaluation

Following established machine learning evaluation methodology:

1. **Three Algorithms Compared**
   - Random Forest (ensemble learning)
   - Support Vector Machine (kernel-based)
   - XGBoost (gradient boosting)

2. **Confusion Matrix Evaluation**
   - True Positives (TP), True Negatives (TN)
   - False Positives (FP), False Negatives (FN)

3. **Four Standard Metrics**
   - **Precision**: Accuracy of positive predictions
   - **Recall**: Sensitivity to positive cases (most critical)
   - **F1-Score**: Harmonic mean of precision and recall
   - **Accuracy**: Overall correctness

4. **Cross-Validation**
   - 5-fold cross-validation
   - R² = 0.96 (consistent performance)

---

## 🌐 Deployment

### Live Demo
Deployed on Render: [Your Render URL]

### Local Development
```bash
python app.py
# Visit: http://127.0.0.1:5000
```

### Production Deployment
```bash
# Automatic deployment via GitHub
git push origin main
# Render auto-deploys on push
```

---

## 📈 Key Achievements

1. ✅ **Realistic accuracy** (92.50%)
2. ✅ **Removed data leakage** (Appliance_kWh_Active)
3. ✅ **Removed synthetic data** (environmental features)
4. ✅ **Professor approved** (followed expert feedback)
5. ✅ **Improved performance** (91.39% → 92.50%)
6. ✅ **Validated with 3 algorithms** (consistent results)
7. ✅ **Cross-validation confirmed** (R² = 0.96)
8. ✅ **Comprehensive documentation** (9 reference files)
9. ✅ **Live deployment** (Render cloud platform)
10. ✅ **Publication ready** (complete methodology)

---

## 🛠️ Technologies Used

- **Backend**: Python, Flask
- **ML Libraries**: scikit-learn, XGBoost
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Deployment**: Render (cloud platform)
- **Version Control**: Git, GitHub

---

## 📝 Citation

If using this project, please cite:

```
Zamboanga City Dorm Electricity Monitoring System
Machine Learning-Powered Consumption Forecasting
Data Source: ZAMCELCO Smart Meter Data
Model: Random Forest (92.50% accuracy)
```

---

## 👥 Contributors

- **Developer**: [Your Name]
- **Institution**: [Your University]
- **Course**: [Your Course]
- **Advisor**: [Professor Name]

---

## 📞 Support

For questions or issues:
- Check documentation in the repository
- Review DEFENSE_CHEAT_SHEET.md for common questions
- See PROFESSOR_FEEDBACK_ANALYSIS.md for methodology details

---

## ✅ Status

**Implementation**: ✅ Complete  
**Model Training**: ✅ Complete (92.50% accuracy)  
**Documentation**: ✅ Complete (9 files)  
**Deployment**: ✅ Live on Render  
**Paper Ready**: ✅ Yes  
**Defense Ready**: ✅ Yes  

---

**Last Updated**: April 30, 2026  
**Version**: 3.0 (Final - Professor Approved)  
**Status**: Production Ready 🚀
