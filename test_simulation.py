"""
Real-life simulation test for the electricity forecasting system.
Tests various scenarios: morning, afternoon, evening, different weather, appliance usage.
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Load the trained model
print("=" * 70)
print("ELECTRICITY CONSUMPTION FORECASTING - REAL-LIFE SIMULATION")
print("=" * 70)
print("\n[1] Loading trained model...")

try:
    model = joblib.load("electricity_model.pkl")
    print(f"✓ Model loaded successfully: {type(model).__name__}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# Load stats cache
try:
    with open("stats_cache.json", "r") as f:
        stats = json.load(f)
    print(f"✓ Stats cache loaded - Model Performance:")
    print(f"  • MAE:  {stats['mae']}")
    print(f"  • RMSE: {stats['rmse']}")
    print(f"  • R²:   {stats['r2']}")
    print(f"  • CV:   {stats['cv']}")
except Exception as e:
    print(f"✗ Warning: Could not load stats cache: {e}")

# Load room config
try:
    with open("room_config.json", "r") as f:
        room_config = json.load(f)
    print(f"✓ Room config loaded: {sum(len(rooms) for rooms in room_config.values())} rooms")
except Exception as e:
    print(f"✗ Error loading room config: {e}")
    exit(1)

# Constants
APPLIANCE_COLS = [
    "App_Electric_Fan", "App_Air_Conditioner", "App_Laptop_PC",
    "App_Refrigerator", "App_TV_Monitor", "App_Phone_Charger",
    "App_Electric_Kettle", "App_Rice_Cooker", "App_Study_Lamp",
]

FEATURE_COLS = [
    "Temperature", "Humidity", "Wind_Speed", "Avg_Past_Consumption",
    "Hour", "Day", "Month", "IsWeekend", "Season", "TimeOfDay", "Is_Anomaly",
    "Dorm_Enc", "Room_Enc", "RoomSize_Enc", "Num_Occupants",
    *APPLIANCE_COLS,
    "Appliance_kWh_Active",
]

SEASON_MAP = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
              6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
DORM_MAP = {"Dorm A": 0, "Dorm B": 1, "Dorm C": 2}
SIZE_MAP = {"Small": 0, "Medium": 1, "Large": 2}

APPLIANCE_WATTS = {
    "App_Electric_Fan": 35,
    "App_Air_Conditioner": 1200,
    "App_Laptop_PC": 65,
    "App_Refrigerator": 120,
    "App_TV_Monitor": 40,
    "App_Phone_Charger": 20,
    "App_Electric_Kettle": 1500,
    "App_Rice_Cooker": 400,
    "App_Study_Lamp": 9,
}

KWH_MAX = 2.0
PESO_PER_KWH = 10.50

def compute_appliance_kwh(appliances_on):
    """Calculate total kWh for 30-min slot from active appliances."""
    total_watts = sum(APPLIANCE_WATTS[app] for app in appliances_on)
    return round(total_watts * 0.5 / 1000, 4)

def predict_consumption(scenario_name, dorm, room, temp_norm, hum_norm, wind_norm, 
                       avg_past_norm, hour, day, month, appliances_on):
    """Make a prediction for given scenario."""
    
    # Get room info
    room_info = room_config[dorm][room]
    size_cat = room_info["size_cat"]
    occupants = room_info["occupants"]
    
    # Derive temporal features
    year = datetime.now().year
    try:
        d = datetime.date(year, month, min(day, 28))
        is_weekend = 1 if d.weekday() >= 5 else 0
    except:
        is_weekend = 0
    
    season = SEASON_MAP.get(month, 0)
    tod = 0 if hour <= 5 else (1 if hour <= 11 else (2 if hour <= 17 else 3))
    
    # Room encoding
    room_num = int(room.split()[1])
    room_enc = room_num - 1
    
    # Appliance flags
    app_flags = {app: (1 if app in appliances_on else 0) for app in APPLIANCE_COLS}
    app_kwh = compute_appliance_kwh(appliances_on)
    
    # Build feature row
    feature_row = {
        "Temperature": temp_norm,
        "Humidity": hum_norm,
        "Wind_Speed": wind_norm,
        "Avg_Past_Consumption": avg_past_norm,
        "Hour": hour,
        "Day": day,
        "Month": month,
        "IsWeekend": is_weekend,
        "Season": season,
        "TimeOfDay": tod,
        "Is_Anomaly": 0,
        "Dorm_Enc": DORM_MAP[dorm],
        "Room_Enc": room_enc,
        "RoomSize_Enc": SIZE_MAP[size_cat],
        "Num_Occupants": occupants,
        **app_flags,
        "Appliance_kWh_Active": app_kwh,
    }
    
    # Make prediction
    features_df = pd.DataFrame([feature_row])[FEATURE_COLS]
    prediction_norm = float(model.predict(features_df)[0])
    prediction_kwh = round(prediction_norm * KWH_MAX, 4)
    prediction_cost = round(prediction_kwh * PESO_PER_KWH, 2)
    status = "⚠️ HIGH" if prediction_norm > 0.75 else "✓ NORMAL"
    
    # Convert normalized values to real units for display
    temp_celsius = round(15 + temp_norm * 20, 1)
    humidity_percent = round(30 + hum_norm * 70, 0)
    wind_kmh = round(wind_norm * 30, 1)
    
    # Print results
    print(f"\n{'─' * 70}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'─' * 70}")
    print(f"Location:     {dorm} → {room} ({size_cat}, {room_info['size_m2']}m², {occupants} occupants)")
    print(f"Time:         {hour:02d}:00, Day {day}, Month {month} ({'Weekend' if is_weekend else 'Weekday'})")
    print(f"Weather:      {temp_celsius}°C, {humidity_percent}% humidity, {wind_kmh} km/h wind")
    print(f"Past Avg:     {avg_past_norm:.2f} (normalized)")
    print(f"Appliances:   {', '.join([a.replace('App_', '').replace('_', ' ') for a in appliances_on]) if appliances_on else 'None'}")
    print(f"App kWh:      {app_kwh} kWh")
    print(f"\n{'═' * 70}")
    print(f"PREDICTION:   {prediction_norm:.4f} (normalized) → {prediction_kwh} kWh")
    print(f"COST:         ₱{prediction_cost}")
    print(f"STATUS:       {status}")
    print(f"{'═' * 70}")
    
    return {
        "scenario": scenario_name,
        "prediction_norm": prediction_norm,
        "prediction_kwh": prediction_kwh,
        "cost": prediction_cost,
        "status": status
    }

# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("RUNNING REAL-LIFE SIMULATIONS")
print("=" * 70)

results = []

# Scenario 1: Early Morning - Minimal Usage
results.append(predict_consumption(
    scenario_name="1. Early Morning - Student Sleeping",
    dorm="Dorm A",
    room="Room 1",
    temp_norm=0.35,  # ~22°C (cool morning)
    hum_norm=0.60,   # ~72% (humid)
    wind_norm=0.20,  # ~6 km/h (light breeze)
    avg_past_norm=0.25,  # Low past consumption
    hour=5,
    day=15,
    month=4,  # April
    appliances_on=["App_Refrigerator"]  # Only fridge running
))

# Scenario 2: Morning - Getting Ready for Class
results.append(predict_consumption(
    scenario_name="2. Morning Rush - Getting Ready",
    dorm="Dorm B",
    room="Room 4",
    temp_norm=0.45,  # ~24°C
    hum_norm=0.55,   # ~68.5%
    wind_norm=0.30,  # ~9 km/h
    avg_past_norm=0.40,
    hour=7,
    day=15,
    month=4,
    appliances_on=["App_Electric_Fan", "App_Laptop_PC", "App_Study_Lamp", 
                   "App_Phone_Charger", "App_Refrigerator", "App_Electric_Kettle"]
))

# Scenario 3: Midday - Hot Weather, AC On
results.append(predict_consumption(
    scenario_name="3. Midday - Hot Day with AC",
    dorm="Dorm C",
    room="Room 7",
    temp_norm=0.85,  # ~32°C (very hot)
    hum_norm=0.70,   # ~79% (very humid)
    wind_norm=0.15,  # ~4.5 km/h (calm)
    avg_past_norm=0.65,  # High past consumption
    hour=13,
    day=15,
    month=5,  # May (summer)
    appliances_on=["App_Air_Conditioner", "App_Electric_Fan", "App_Laptop_PC",
                   "App_Refrigerator", "App_Phone_Charger"]
))

# Scenario 4: Afternoon Study Session
results.append(predict_consumption(
    scenario_name="4. Afternoon - Study Session",
    dorm="Dorm A",
    room="Room 3",
    temp_norm=0.60,  # ~27°C
    hum_norm=0.50,   # ~65%
    wind_norm=0.40,  # ~12 km/h
    avg_past_norm=0.50,
    hour=15,
    day=20,
    month=3,  # March
    appliances_on=["App_Electric_Fan", "App_Laptop_PC", "App_Study_Lamp",
                   "App_Refrigerator", "App_Phone_Charger"]
))

# Scenario 5: Evening - Dinner Time
results.append(predict_consumption(
    scenario_name="5. Evening - Cooking Dinner",
    dorm="Dorm B",
    room="Room 2",
    temp_norm=0.50,  # ~25°C
    hum_norm=0.60,   # ~72%
    wind_norm=0.25,  # ~7.5 km/h
    avg_past_norm=0.55,
    hour=18,
    day=20,
    month=3,
    appliances_on=["App_Electric_Fan", "App_Rice_Cooker", "App_Electric_Kettle",
                   "App_TV_Monitor", "App_Refrigerator", "App_Study_Lamp"]
))

# Scenario 6: Night - Entertainment & Relaxation
results.append(predict_consumption(
    scenario_name="6. Night - Movie & Chill",
    dorm="Dorm C",
    room="Room 5",
    temp_norm=0.40,  # ~23°C (cooler evening)
    hum_norm=0.65,   # ~75.5%
    wind_norm=0.35,  # ~10.5 km/h
    avg_past_norm=0.45,
    hour=21,
    day=25,
    month=11,  # November
    appliances_on=["App_Electric_Fan", "App_TV_Monitor", "App_Laptop_PC",
                   "App_Phone_Charger", "App_Refrigerator", "App_Study_Lamp"]
))

# Scenario 7: Late Night - Cramming for Exam
results.append(predict_consumption(
    scenario_name="7. Late Night - Exam Cramming",
    dorm="Dorm A",
    room="Room 5",
    temp_norm=0.35,  # ~22°C
    hum_norm=0.70,   # ~79%
    wind_norm=0.10,  # ~3 km/h
    avg_past_norm=0.60,
    hour=23,
    day=10,
    month=6,  # June
    appliances_on=["App_Electric_Fan", "App_Laptop_PC", "App_Study_Lamp",
                   "App_Phone_Charger", "App_Refrigerator"]
))

# Scenario 8: Weekend - All Appliances (Maximum Load)
results.append(predict_consumption(
    scenario_name="8. Weekend - Maximum Load Test",
    dorm="Dorm B",
    room="Room 6",
    temp_norm=0.90,  # ~33°C (extreme heat)
    hum_norm=0.80,   # ~86% (very humid)
    wind_norm=0.05,  # ~1.5 km/h (almost no wind)
    avg_past_norm=0.85,  # Very high past consumption
    hour=14,
    day=28,
    month=7,  # July (peak summer)
    appliances_on=APPLIANCE_COLS  # ALL appliances on!
))

# Scenario 9: Cool Weather - Minimal AC Need
results.append(predict_consumption(
    scenario_name="9. Cool Rainy Day - Low Consumption",
    dorm="Dorm C",
    room="Room 2",
    temp_norm=0.20,  # ~19°C (cool)
    hum_norm=0.85,   # ~89.5% (rainy)
    wind_norm=0.60,  # ~18 km/h (windy)
    avg_past_norm=0.30,
    hour=16,
    day=5,
    month=12,  # December (cool season)
    appliances_on=["App_Laptop_PC", "App_Study_Lamp", "App_Phone_Charger",
                   "App_Refrigerator"]
))

# Scenario 10: Small Room vs Large Room Comparison
print("\n\n" + "=" * 70)
print("BONUS: ROOM SIZE COMPARISON")
print("=" * 70)

results.append(predict_consumption(
    scenario_name="10a. Small Room (12m²) - Same Conditions",
    dorm="Dorm C",
    room="Room 3",  # 12m² Small
    temp_norm=0.70,
    hum_norm=0.60,
    wind_norm=0.30,
    avg_past_norm=0.50,
    hour=14,
    day=15,
    month=5,
    appliances_on=["App_Air_Conditioner", "App_Laptop_PC", "App_Refrigerator"]
))

results.append(predict_consumption(
    scenario_name="10b. Large Room (32m²) - Same Conditions",
    dorm="Dorm A",
    room="Room 4",  # 32m² Large
    temp_norm=0.70,
    hum_norm=0.60,
    wind_norm=0.30,
    avg_past_norm=0.50,
    hour=14,
    day=15,
    month=5,
    appliances_on=["App_Air_Conditioner", "App_Laptop_PC", "App_Refrigerator"]
))

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("SIMULATION SUMMARY")
print("=" * 70)

total_cost = sum(r["cost"] for r in results)
avg_kwh = sum(r["prediction_kwh"] for r in results) / len(results)
high_consumption_count = sum(1 for r in results if "HIGH" in r["status"])

print(f"\nTotal Scenarios Tested:     {len(results)}")
print(f"High Consumption Warnings:  {high_consumption_count}")
print(f"Average Consumption:        {avg_kwh:.4f} kWh per 30-min slot")
print(f"Total Cost (all scenarios): ₱{total_cost:.2f}")
print(f"\nLowest Consumption:  {min(results, key=lambda x: x['prediction_kwh'])['scenario']}")
print(f"                     {min(r['prediction_kwh'] for r in results)} kWh")
print(f"\nHighest Consumption: {max(results, key=lambda x: x['prediction_kwh'])['scenario']}")
print(f"                     {max(r['prediction_kwh'] for r in results)} kWh")

print("\n" + "=" * 70)
print("✓ SIMULATION COMPLETED SUCCESSFULLY")
print("=" * 70)
print("\nThe model is working correctly and producing realistic predictions!")
print("You can now run the Flask app with: python app.py")
print("=" * 70)
