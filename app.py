import io
import json
import base64
import datetime
import threading
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, render_template

app = Flask(__name__)

# ── Model ────────────────────────────────────────────────────────────────────
try:
    model = joblib.load("electricity_model.pkl")
    print(f"Model loaded: {type(model).__name__}")
except Exception as e:
    print(f"FATAL: Could not load electricity_model.pkl: {e}")
    raise RuntimeError(f"Could not load electricity_model.pkl: {e}")

APPLIANCE_COLS = [
    "App_Electric_Fan", "App_Air_Conditioner", "App_Laptop_PC",
    "App_Refrigerator", "App_TV_Monitor", "App_Phone_Charger",
    "App_Electric_Kettle", "App_Rice_Cooker", "App_Study_Lamp",
]

FEATURE_COLS = [
    # Removed environmental features per professor's feedback:
    # "Temperature", "Humidity", "Wind_Speed" - were synthetic/normalized data
    "Avg_Past_Consumption",
    "Hour", "Day", "IsWeekend", "TimeOfDay", "Is_Anomaly",
    # Month and Season removed: only 1.45 months of data, insufficient for seasonal patterns
    "Dorm_Enc", "Room_Enc", "RoomSize_Enc", "Num_Occupants",
    *APPLIANCE_COLS,
    # Removed "Appliance_kWh_Active" - data leakage
]

# Appliance catalogue: key → (default_watts, display_label)
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

SEASON_MAP = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
              6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}

DORMS      = ["Dorm A", "Dorm B", "Dorm C"]
DORM_MAP   = {d: i for i, d in enumerate(DORMS)}
ROOMS      = [f"Room {r}" for r in range(1, 9)]  # 8 rooms per dorm
SIZE_MAP   = {"Small": 0, "Medium": 1, "Large": 2}

# Load room configuration (size, occupants per room)
ROOM_CONFIG_FILE = Path("room_config.json")
try:
    with open(ROOM_CONFIG_FILE, "r", encoding="utf-8") as f:
        ROOM_CONFIG = json.load(f)
    print(f"Room config loaded: {sum(len(rooms) for rooms in ROOM_CONFIG.values())} rooms")
except Exception as e:
    print(f"Warning: Could not load room_config.json: {e}")
    ROOM_CONFIG = {}

ANOMALY_THRESHOLD = 0.75
KWH_MAX           = 2.0      # normalized 1.0 → 2.0 kWh per 30-min slot
PESO_PER_KWH      = 10.50

# ── Prediction history persistence ───────────────────────────────────────────
HISTORY_FILE = Path("prediction_history.json")
MAX_HISTORY  = 5

def _load_history() -> list:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []

def _save_history(history: list) -> None:
    try:
        HISTORY_FILE.write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass

prediction_history: list = _load_history()
_history_lock = threading.Lock()

# ── Stats (load from pre-computed cache) ─────────────────────────────────────
STATS_CACHE_FILE = Path("stats_cache.json")

def _load_stats() -> dict:
    """Load pre-computed stats from stats_cache.json (instant startup)."""
    if STATS_CACHE_FILE.exists():
        try:
            with open(STATS_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not load stats cache: {e}")
    # Fallback if cache missing
    return {
        "mae": "N/A", "rmse": "N/A", "r2": "N/A", "cv": "N/A",
        "importances": [], "chart": None,
        "model_type": type(model).__name__,
        "top_rooms": [], "error": "Stats cache not found",
    }

try:
    stats: dict = _load_stats()
    # Add hourly consumption data for interactive chart
    if 'hourly_data' not in stats:
        stats['hourly_data'] = [0.2,0.15,0.12,0.1,0.11,0.15,0.25,0.35,0.4,0.38,0.35,0.33,0.36,0.4,0.42,0.45,0.48,0.5,0.45,0.4,0.35,0.3,0.28,0.25]
    print(f"Stats loaded: MAE={stats.get('mae')}, R²={stats.get('r2')}")
except Exception as e:
    print(f"ERROR loading stats: {e}")
    stats = {
        "mae": "Error", "rmse": "Error", "r2": "Error", "cv": "Error",
        "importances": [], "chart": None,
        "model_type": type(model).__name__,
        "top_rooms": [], "error": str(e),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_room_info(dorm: str, room: str) -> dict:
    """Get room characteristics from config."""
    if dorm in ROOM_CONFIG and room in ROOM_CONFIG[dorm]:
        return ROOM_CONFIG[dorm][room]
    # Fallback defaults
    return {"size_m2": 20, "size_cat": "Medium", "occupants": 2}


def derive_extra(hour: int, day: int) -> tuple:
    """Derive IsWeekend and TimeOfDay from hour and day."""
    year = datetime.date.today().year
    month = 3  # Default to March (dataset month)
    try:
        d = datetime.date(year, month, min(day, 28))
        is_weekend = 1 if d.weekday() >= 5 else 0
    except Exception:
        is_weekend = 0
    tod = 0 if hour <= 5 else (1 if hour <= 11 else (2 if hour <= 17 else 3))
    return is_weekend, tod


def compute_appliance_kwh(active_flags: dict, custom_watts: dict) -> float:
    """Compute total kWh consumed in a 30-min slot from active appliance flags and custom wattages."""
    total_watts = sum(
        custom_watts.get(key, APPLIANCE_INFO[key][0])
        for key in APPLIANCE_COLS
        if active_flags.get(key, 0)
    )
    return round(total_watts * 0.5 / 1000, 4)   # 30 min = 0.5 h


def validate_form(form) -> tuple:
    errors = []

    def _float(key, lo, hi, label):
        raw = form.get(key, "").strip()
        if not raw:
            errors.append(f"{label} is required.")
            return None
        try:
            v = float(raw)
        except ValueError:
            errors.append(f"{label} must be a number.")
            return None
        if not (lo <= v <= hi):
            errors.append(f"{label} must be between {lo} and {hi}.")
            return None
        return v

    def _int(key, lo, hi, label):
        raw = form.get(key, "").strip()
        if not raw:
            errors.append(f"{label} is required.")
            return None
        try:
            v = int(raw)
        except ValueError:
            errors.append(f"{label} must be a whole number.")
            return None
        if not (lo <= v <= hi):
            errors.append(f"{label} must be between {lo} and {hi}.")
            return None
        return v

    # Environmental features removed per professor's feedback
    # temp  = _float("temperature",          0.0, 1.0, "Temperature")
    # hum   = _float("humidity",             0.0, 1.0, "Humidity")
    # wind  = _float("wind_speed",           0.0, 1.0, "Wind Speed")
    
    apc   = _float("avg_past_consumption", 0.0, 1.0, "Avg Past Consumption")
    hour  = _int("hour",  0, 23, "Hour")
    day   = _int("day",   1, 31, "Day")
    # Month removed: only 1.45 months in dataset, not meaningful for prediction

    dorm     = form.get("dorm_id", "Dorm A")
    room     = form.get("room_id", "Room 1")

    if dorm not in DORM_MAP:
        errors.append(f"Invalid dorm: {dorm!r}")
    if room not in ROOMS:
        errors.append(f"Invalid room: {room!r}")

    # Get room characteristics from config (size is auto-filled)
    room_info = get_room_info(dorm, room)
    size_cat = room_info["size_cat"]
    
    # Get occupancy from user input (allows override of default)
    occ_input = form.get("num_occupants", "").strip()
    if occ_input:
        try:
            occ = int(occ_input)
            if not (1 <= occ <= 4):
                errors.append("Number of occupants must be between 1 and 4.")
                occ = room_info["occupants"]  # Fallback to default
        except ValueError:
            errors.append("Number of occupants must be a number.")
            occ = room_info["occupants"]  # Fallback to default
    else:
        occ = room_info["occupants"]  # Use default from config

    # Appliance flags (checkboxes — absent means 0)
    app_flags = {key: 1 if form.get(key) else 0 for key in APPLIANCE_COLS}
    
    # Custom wattage values (optional — defaults to APPLIANCE_INFO if not provided)
    custom_watts = {}
    for key in APPLIANCE_COLS:
        watts_key = f"{key}_watts"
        if watts_key in form:
            try:
                w = int(form.get(watts_key, "0"))
                if 1 <= w <= 5000:
                    custom_watts[key] = w
            except ValueError:
                pass  # Use default if invalid

    if errors:
        return None, " ".join(errors)

    return {
        # Environmental features removed
        # "temp": temp, "hum": hum, "wind": wind,
        "apc": apc,
        "hour": hour, "day": day,
        # Month removed: only 1.45 months in dataset
        "dorm": dorm, "room": room,
        "size_cat": size_cat, "occ": occ,
        "app_flags": app_flags,
        "custom_watts": custom_watts,
    }, None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    """Health check endpoint for Render"""
    return {"status": "ok", "model": type(model).__name__, "stats_loaded": "mae" in stats}


@app.route("/", methods=["GET", "POST"])
def index():
    global prediction_history

    prediction    = None
    pred_status   = None
    selected_dorm = None
    selected_room = None
    pred_kwh      = None
    pred_cost     = None
    form_error    = None
    app_kwh_breakdown = []
    room_info     = None
    occ           = None

    if request.method == "POST":
        parsed, form_error = validate_form(request.form)

        if parsed:
            # Environmental features removed
            # temp     = parsed["temp"]
            # hum      = parsed["hum"]
            # wind     = parsed["wind"]
            
            apc      = parsed["apc"]
            hour     = parsed["hour"]
            day      = parsed["day"]
            # Month removed: only 1.45 months in dataset
            dorm     = parsed["dorm"]
            room     = parsed["room"]
            size_cat = parsed["size_cat"]
            occ      = parsed["occ"]
            app_flags = parsed["app_flags"]
            custom_watts = parsed["custom_watts"]

            selected_dorm = dorm
            selected_room = room

            is_weekend, tod = derive_extra(hour, day)
            app_kwh = compute_appliance_kwh(app_flags, custom_watts)

            # Room encoding: Room 1-8 → 0-7
            room_num = int(room.split()[1])
            room_enc = room_num - 1
            
            # Get room info for display
            room_info = get_room_info(dorm, room)

            feature_row = {
                # Environmental features removed per professor's feedback
                # "Temperature":          temp,
                # "Humidity":             hum,
                # "Wind_Speed":           wind,
                "Avg_Past_Consumption": apc,
                "Hour":                 hour,
                "Day":                  day,
                # Month and Season removed: only 1.45 months of data
                "IsWeekend":            is_weekend,
                "TimeOfDay":            tod,
                "Is_Anomaly":           0,
                "Dorm_Enc":             DORM_MAP[dorm],
                "Room_Enc":             room_enc,
                "RoomSize_Enc":         SIZE_MAP[size_cat],
                "Num_Occupants":        occ,
                **app_flags,
                # "Appliance_kWh_Active": app_kwh,  # Removed - data leakage
            }

            features_df = pd.DataFrame([feature_row])[FEATURE_COLS]
            prediction  = round(float(model.predict(features_df)[0]), 4)
            pred_status = "High Consumption" if prediction > ANOMALY_THRESHOLD else "Normal"
            pred_kwh    = round(prediction * KWH_MAX, 4)
            pred_cost   = round(pred_kwh * PESO_PER_KWH, 2)

            # Per-appliance kWh breakdown for the result panel
            app_kwh_breakdown = [
                {
                    "label":  APPLIANCE_INFO[key][1],
                    "watts":  custom_watts.get(key, APPLIANCE_INFO[key][0]),
                    "active": app_flags[key],
                    "kwh":    round(custom_watts.get(key, APPLIANCE_INFO[key][0]) * 0.5 / 1000, 4),
                }
                for key in APPLIANCE_COLS
            ]

            months_short = ["Jan","Feb","Mar","Apr","May","Jun",
                            "Jul","Aug","Sep","Oct","Nov","Dec"]
            # Use March as default month for display (dataset is March-April)
            display_month = "Mar"
            active_apps = [k.replace("App_","").replace("_"," ")
                           for k, v in app_flags.items() if v]
            entry = {
                "dorm": dorm, "room": room,
                # Environmental features removed
                # "temp": temp, "hum": hum, "wind": wind,
                "apc": apc,
                "time": f"{hour:02d}:00 · Day {day} · {display_month}",
                "result": prediction, "status": pred_status,
                "kwh": pred_kwh, "cost": pred_cost,
                "occ": occ, "size": size_cat,
                "appliances": ", ".join(active_apps) if active_apps else "None",
                "app_kwh": app_kwh,
            }

            with _history_lock:
                prediction_history = [entry] + prediction_history
                prediction_history = prediction_history[:MAX_HISTORY]
                _save_history(prediction_history)

    current_stats = dict(stats)

    with _history_lock:
        history_snapshot = list(prediction_history)

    return render_template(
        "index.html",
        prediction=prediction,
        pred_status=pred_status,
        selected_dorm=selected_dorm,
        selected_room=selected_room,
        pred_kwh=pred_kwh,
        pred_cost=pred_cost,
        stats=current_stats,
        history=history_snapshot,
        top_rooms=current_stats.get("top_rooms", []),
        form_error=form_error,
        appliance_info=APPLIANCE_INFO,
        appliance_cols=APPLIANCE_COLS,
        app_kwh_breakdown=app_kwh_breakdown,
        room_config=ROOM_CONFIG,
        room_info=room_info,
        occ=occ,
    )


if __name__ == "__main__":
    app.run(debug=True)
