"""
config.py
Single source of truth for all shared constants, feature lists, and mappings.
Imported by app.py, train_model.py, precompute_stats.py, and generate_dataset.py.
"""

# ── Dorm & Room Structure ─────────────────────────────────────────────────────
DORMS = ["Dorm A", "Dorm B", "Dorm C"]
DORM_MAP = {d: i for i, d in enumerate(DORMS)}

ROOMS_PER_DORM = 8  # default rooms per dorm
SIZE_CATEGORIES = ["Small", "Medium", "Large"]
SIZE_MAP = {"Small": 0, "Medium": 1, "Large": 2}

# Size category → typical m² range
SIZE_M2_RANGE = {
    "Small": (10, 15),
    "Medium": (16, 25),
    "Large": (26, 35),
}

# ── Appliance Catalogue ───────────────────────────────────────────────────────
# key → (default_watts, display_label)
APPLIANCE_INFO = {
    "App_Electric_Fan":    (35,   "Electric Fan"),
    "App_Air_Conditioner": (1200, "Air Conditioner"),
    "App_Laptop_PC":       (65,   "Laptop / PC"),
    "App_Refrigerator":    (120,  "Refrigerator"),
    "App_TV_Monitor":      (40,   "TV / Monitor"),
    "App_Phone_Charger":   (20,   "Phone Charger"),
    "App_Electric_Kettle": (1500, "Electric Kettle"),
    "App_Rice_Cooker":     (400,  "Rice Cooker"),
    "App_Study_Lamp":      (9,    "Study Lamp"),
}

APPLIANCE_COLS = list(APPLIANCE_INFO.keys())

# Wattage feature columns (0 if off, actual watts if on)
APPLIANCE_WATT_COLS = [f"{col}_W" for col in APPLIANCE_COLS]

# ── Feature Columns (model input) ────────────────────────────────────────────
# These are the final features fed to the model in exact order.
FEATURE_COLS = [
    # Historical
    "Avg_Past_Consumption",
    # Time
    "Hour", "Day", "Month", "IsWeekend", "TimeOfDay",
    # Room context
    "Dorm_Enc", "Room_Enc", "RoomSize_Enc", "Num_Occupants",
    # Appliance aggregates
    "Total_Active_Wattage",
    "Num_Active_Appliances",
    "Has_High_Power_Appliance",
    # Per-appliance wattage (0 if off, W if on)
    *APPLIANCE_WATT_COLS,
]

# ── Thresholds & Constants ────────────────────────────────────────────────────
ANOMALY_THRESHOLD = 0.75       # normalized consumption above this = "High"
KWH_MAX = 2.0                  # normalized 1.0 maps to ~2.0 kWh per 30-min slot
PESO_PER_KWH = 10.50           # ZAMCELCO approximate rate

HIGH_POWER_WATT_THRESHOLD = 500  # appliances above this are "high power"

# ── History ───────────────────────────────────────────────────────────────────
MAX_HISTORY = 50  # keep more history for Avg_Past_Consumption derivation

# ── Season Map (unused in features but kept for reference) ────────────────────
SEASON_MAP = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
              6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
