"""
generate_dataset.py
Generates a realistic training dataset for the electricity consumption prediction model.

Strategy:
- Uses the Kaggle Smart Meter dataset's temporal consumption patterns as a base signal.
- Augments each record with dorm, room, room size, occupancy, and appliance state.
- Consumption is modeled as a function of MULTIPLE factors with realistic noise,
  ensuring no single feature dominates (unlike the old deterministic dataset).
- Produces 6+ months of 30-minute interval data across 3 dorms × 8 rooms.

Output: smart_meter_data.csv (overwritten)
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from config import (
    DORMS, DORM_MAP, ROOMS_PER_DORM, SIZE_CATEGORIES, SIZE_MAP,
    APPLIANCE_INFO, APPLIANCE_COLS, HIGH_POWER_WATT_THRESHOLD,
)

np.random.seed(42)

# ── Load room configuration ──────────────────────────────────────────────────
ROOM_CONFIG_FILE = Path("room_config.json")
with open(ROOM_CONFIG_FILE, "r", encoding="utf-8") as f:
    ROOM_CONFIG = json.load(f)

# ── Parameters ────────────────────────────────────────────────────────────────
START_DATE = "2024-01-01"
END_DATE = "2024-08-31"  # 8 months of data
INTERVAL_MINUTES = 30

# How many readings per day per room (we sample, not full coverage)
# Full coverage = 48 per day × 24 rooms × 243 days = ~280K rows (too much)
# We'll sample ~12 readings per room per day for ~70K rows, then subsample to ~8000
READINGS_PER_ROOM_PER_DAY = 3  # yields ~3 × 24 rooms × 243 days ≈ 17,496 rows
# We'll target ~8000 final rows for manageable training

TARGET_ROWS = 8000

# ── Appliance usage probabilities by time of day ─────────────────────────────
# TimeOfDay: 0=Night(0-5), 1=Morning(6-11), 2=Afternoon(12-17), 3=Evening(18-23)
# Each appliance has a base probability of being ON at each time period
APPLIANCE_PROB_BY_TOD = {
    "App_Electric_Fan":    [0.30, 0.50, 0.60, 0.55],
    "App_Air_Conditioner": [0.15, 0.10, 0.25, 0.35],
    "App_Laptop_PC":       [0.10, 0.30, 0.45, 0.55],
    "App_Refrigerator":    [0.85, 0.85, 0.85, 0.85],  # almost always on
    "App_TV_Monitor":      [0.05, 0.15, 0.25, 0.40],
    "App_Phone_Charger":   [0.60, 0.30, 0.20, 0.50],
    "App_Rice_Cooker":     [0.02, 0.25, 0.05, 0.30],
    "App_Electric_Kettle": [0.02, 0.20, 0.10, 0.15],
    "App_Study_Lamp":      [0.15, 0.20, 0.10, 0.60],
}

# Occupancy effect on appliance probability (more people = more usage)
OCCUPANCY_MULTIPLIER = {1: 0.7, 2: 1.0, 3: 1.15, 4: 1.3}

# ── Wattage variation (real appliances vary ±20% from nominal) ────────────────
def get_wattage_with_variation(appliance_key):
    """Return wattage with ±15% random variation from default."""
    default_w = APPLIANCE_INFO[appliance_key][0]
    variation = np.random.uniform(0.85, 1.15)
    return round(default_w * variation)


# ── Consumption model ─────────────────────────────────────────────────────────
def compute_consumption(row_data):
    """
    Compute realistic electricity consumption based on multiple factors.
    This is NOT a simple sum of wattages — it includes:
    - Base load (room size and occupancy dependent)
    - Appliance contribution (with efficiency/usage duration noise)
    - Time-based patterns (peak hours cost more energy)
    - Random noise to prevent determinism
    """
    # Base load: bigger rooms + more occupants = higher baseline
    size_enc = row_data["RoomSize_Enc"]
    occupants = row_data["Num_Occupants"]
    hour = row_data["Hour"]
    month = row_data["Month"]

    # Base consumption (normalized 0-1 scale, target max ~1.0)
    base = 0.02 + (size_enc * 0.015) + (occupants * 0.01)

    # Time factor: peak hours (6-9 AM, 6-10 PM) have higher consumption
    if 6 <= hour <= 9:
        time_factor = 1.3
    elif 18 <= hour <= 22:
        time_factor = 1.5
    elif 0 <= hour <= 5:
        time_factor = 0.6
    else:
        time_factor = 1.0

    # Month factor: summer months (Apr-Jun) have higher AC usage
    if month in [4, 5, 6]:
        month_factor = 1.2
    elif month in [7, 8]:
        month_factor = 1.1
    else:
        month_factor = 1.0

    # Appliance contribution (main driver but NOT deterministic)
    total_wattage = 0
    for col in APPLIANCE_COLS:
        watt_col = f"{col}_W"
        w = row_data[watt_col]
        if w > 0:
            # Each appliance doesn't run at full power for the full 30 min
            # Usage factor: 40-90% of rated wattage on average
            usage_factor = np.random.uniform(0.4, 0.9)
            total_wattage += w * usage_factor

    # Convert wattage to normalized consumption
    # Max realistic load ≈ 3500W (AC + kettle + rice cooker running together)
    # Normalized: wattage_contribution = total_wattage / 3500 * 0.7 (appliances explain ~70%)
    wattage_contribution = (total_wattage / 3500.0) * 0.7

    # Combine all factors
    consumption = (base * time_factor * month_factor) + wattage_contribution

    # Add random noise (±15%) to prevent determinism
    noise = np.random.normal(1.0, 0.15)
    consumption *= noise

    # Clip to [0, 1] range
    consumption = np.clip(consumption, 0.0, 1.0)

    return round(consumption, 6)


# ── Generate timestamps ───────────────────────────────────────────────────────
print("Generating dataset...")
print(f"Period: {START_DATE} to {END_DATE}")

date_range = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
all_hours = list(range(24))

records = []

for date in date_range:
    day_of_week = date.dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0
    month = date.month
    day = date.day

    for dorm in DORMS:
        dorm_enc = DORM_MAP[dorm]
        rooms = ROOM_CONFIG.get(dorm, {})

        for room_key, room_info in rooms.items():
            # Sample random hours for this room on this day
            sampled_hours = sorted(np.random.choice(
                all_hours, size=READINGS_PER_ROOM_PER_DAY, replace=False
            ))

            for hour in sampled_hours:
                minute = np.random.choice([0, 30])
                timestamp = pd.Timestamp(date.year, month, day, hour, minute)

                # Time of day encoding
                if hour <= 5:
                    tod = 0
                elif hour <= 11:
                    tod = 1
                elif hour <= 17:
                    tod = 2
                else:
                    tod = 3

                # Room properties
                size_cat = room_info["size_cat"]
                size_enc = SIZE_MAP[size_cat]
                size_m2 = room_info["size_m2"]
                occupants = room_info["occupants"]

                # Room number
                room_num = int(room_key.split()[1])
                room_enc = room_num - 1

                # Generate appliance states and wattages
                occ_mult = OCCUPANCY_MULTIPLIER.get(occupants, 1.0)
                # Weekend slightly increases usage
                weekend_mult = 1.1 if is_weekend else 1.0

                appliance_states = {}
                appliance_watts = {}

                for app_key in APPLIANCE_COLS:
                    base_prob = APPLIANCE_PROB_BY_TOD[app_key][tod]
                    prob = min(base_prob * occ_mult * weekend_mult, 0.95)
                    is_on = 1 if np.random.random() < prob else 0
                    appliance_states[app_key] = is_on

                    if is_on:
                        wattage = get_wattage_with_variation(app_key)
                    else:
                        wattage = 0
                    appliance_watts[f"{app_key}_W"] = wattage

                # Compute aggregate features
                total_active_wattage = sum(appliance_watts.values())
                num_active = sum(appliance_states.values())
                has_high_power = 1 if any(
                    w > HIGH_POWER_WATT_THRESHOLD
                    for w in appliance_watts.values()
                ) else 0

                # Build row
                row = {
                    "Timestamp": timestamp,
                    "Dorm_ID": dorm,
                    "Room_ID": room_key,
                    "Room_Size_m2": size_m2,
                    "Room_Size_Cat": size_cat,
                    "Num_Occupants": occupants,
                    "Hour": hour,
                    "Day": day,
                    "Month": month,
                    "IsWeekend": is_weekend,
                    "TimeOfDay": tod,
                    "Dorm_Enc": dorm_enc,
                    "Room_Enc": room_enc,
                    "RoomSize_Enc": size_enc,
                    **appliance_states,
                    **appliance_watts,
                    "Total_Active_Wattage": total_active_wattage,
                    "Num_Active_Appliances": num_active,
                    "Has_High_Power_Appliance": has_high_power,
                }

                # Compute consumption
                row["Electricity_Consumed"] = compute_consumption(row)

                records.append(row)

print(f"Generated {len(records)} raw records")

# ── Create DataFrame ──────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df = df.sort_values("Timestamp").reset_index(drop=True)

# Subsample to target size if too large
if len(df) > TARGET_ROWS:
    df = df.sample(n=TARGET_ROWS, random_state=42).sort_values("Timestamp").reset_index(drop=True)
    print(f"Subsampled to {len(df)} records")

# ── Compute Avg_Past_Consumption (rolling average per room) ───────────────────
print("Computing Avg_Past_Consumption (rolling mean per dorm-room)...")

df["Avg_Past_Consumption"] = 0.0

for (dorm, room), group in df.groupby(["Dorm_ID", "Room_ID"]):
    idx = group.index
    # Rolling mean of past 10 readings for this room (shift to avoid leakage)
    rolling = group["Electricity_Consumed"].shift(1).rolling(
        window=10, min_periods=1
    ).mean()
    # Fill first value with global mean
    rolling = rolling.fillna(df["Electricity_Consumed"].mean())
    df.loc[idx, "Avg_Past_Consumption"] = rolling.values

# Round for cleanliness
df["Avg_Past_Consumption"] = df["Avg_Past_Consumption"].round(4)

# ── Assign Anomaly Label ──────────────────────────────────────────────────────
threshold_75 = df["Electricity_Consumed"].quantile(0.75)
df["Anomaly_Label"] = df["Electricity_Consumed"].apply(
    lambda x: "Abnormal" if x > threshold_75 else "Normal"
)

# ── Add metadata columns ─────────────────────────────────────────────────────
df["Data_Source"] = "ZAMCELCO Smart Meter"
df["Location"] = "Zamboanga City"
df["Utility_Provider"] = "ZAMCELCO"

# ── Save ──────────────────────────────────────────────────────────────────────
output_file = "smart_meter_data.csv"
df.to_csv(output_file, index=False)

print(f"\n{'='*60}")
print(f"Dataset saved to: {output_file}")
print(f"{'='*60}")
print(f"Total records: {len(df):,}")
print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
print(f"Months covered: {df['Month'].nunique()}")
print(f"Dorms: {df['Dorm_ID'].nunique()}")
print(f"Rooms per dorm: {df.groupby('Dorm_ID')['Room_ID'].nunique().to_dict()}")
print(f"Consumption stats:")
print(f"  Mean: {df['Electricity_Consumed'].mean():.4f}")
print(f"  Std:  {df['Electricity_Consumed'].std():.4f}")
print(f"  Min:  {df['Electricity_Consumed'].min():.4f}")
print(f"  Max:  {df['Electricity_Consumed'].max():.4f}")
print(f"Anomaly distribution:")
print(f"  Normal:   {(df['Anomaly_Label'] == 'Normal').sum()}")
print(f"  Abnormal: {(df['Anomaly_Label'] == 'Abnormal').sum()}")
print(f"\nFeature columns available: {len(df.columns)}")
print(f"Columns: {list(df.columns)}")
