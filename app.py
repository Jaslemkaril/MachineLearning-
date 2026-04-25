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
    "Temperature", "Humidity", "Wind_Speed", "Avg_Past_Consumption",
    "Hour", "Day", "Month", "IsWeekend", "Season", "TimeOfDay", "Is_Anomaly",
    "Dorm_Enc", "Room_Enc", "RoomSize_Enc", "Num_Occupants",
    *APPLIANCE_COLS,
    "Appliance_kWh_Active",
]

# Appliance catalogue: key → (brand, model, rated_watts)
APPLIANCE_INFO = {
    "App_Electric_Fan":    ("Panasonic",  "F-M14D5",           35),
    "App_Air_Conditioner": ("Carrier",    "53KHCT012-703",    1200),
    "App_Laptop_PC":       ("ASUS",       "VivoBook 15 X1502", 65),
    "App_Refrigerator":    ("Condura",    "CTD-510MNi",       120),
    "App_TV_Monitor":      ("Samsung",    "UA32T4500",         40),
    "App_Phone_Charger":   ("Anker",      "PowerPort III 20W", 20),
    "App_Electric_Kettle": ("Kyowa",      "KW-1270",         1500),
    "App_Rice_Cooker":     ("Hanabishi",  "HRC-508",          400),
    "App_Study_Lamp":      ("Firefly",    "FEL-500",            9),
}

SEASON_MAP = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
              6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}

DORMS      = ["Dorm A", "Dorm B", "Dorm C"]
DORM_MAP   = {d: i for i, d in enumerate(DORMS)}
ROOMS      = [f"Room {r}" for r in range(101, 109)]
ROOM_MAP   = {r: i for i, r in enumerate(ROOMS)}
SIZE_MAP   = {"Small": 0, "Medium": 1, "Large": 2}

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
def derive_extra(hour: int, day: int, month: int) -> tuple:
    year = datetime.date.today().year
    try:
        d = datetime.date(year, month, min(day, 28))
        is_weekend = 1 if d.weekday() >= 5 else 0
    except Exception:
        is_weekend = 0
    season = SEASON_MAP.get(month, 0)
    tod = 0 if hour <= 5 else (1 if hour <= 11 else (2 if hour <= 17 else 3))
    return is_weekend, season, tod


def compute_appliance_kwh(active_flags: dict) -> float:
    """Compute total kWh consumed in a 30-min slot from active appliance flags."""
    total_watts = sum(
        APPLIANCE_INFO[key][2]
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

    temp  = _float("temperature",          0.0, 1.0, "Temperature")
    hum   = _float("humidity",             0.0, 1.0, "Humidity")
    wind  = _float("wind_speed",           0.0, 1.0, "Wind Speed")
    apc   = _float("avg_past_consumption", 0.0, 1.0, "Avg Past Consumption")
    hour  = _int("hour",  0, 23, "Hour")
    day   = _int("day",   1, 31, "Day")
    month = _int("month", 1, 12, "Month")
    occ   = _int("num_occupants", 1, 4, "Number of Occupants")

    dorm     = form.get("dorm_id", "Dorm A")
    room     = form.get("room_id", "Room 101")
    size_cat = form.get("room_size_cat", "Medium")

    if dorm not in DORM_MAP:
        errors.append(f"Invalid dorm: {dorm!r}")
    if room not in ROOM_MAP:
        errors.append(f"Invalid room: {room!r}")
    if size_cat not in SIZE_MAP:
        errors.append(f"Invalid room size: {size_cat!r}")

    # Appliance flags (checkboxes — absent means 0)
    app_flags = {key: 1 if form.get(key) else 0 for key in APPLIANCE_COLS}

    if errors:
        return None, " ".join(errors)

    return {
        "temp": temp, "hum": hum, "wind": wind, "apc": apc,
        "hour": hour, "day": day, "month": month,
        "dorm": dorm, "room": room,
        "size_cat": size_cat, "occ": occ,
        "app_flags": app_flags,
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

    if request.method == "POST":
        parsed, form_error = validate_form(request.form)

        if parsed:
            temp     = parsed["temp"]
            hum      = parsed["hum"]
            wind     = parsed["wind"]
            apc      = parsed["apc"]
            hour     = parsed["hour"]
            day      = parsed["day"]
            month    = parsed["month"]
            dorm     = parsed["dorm"]
            room     = parsed["room"]
            size_cat = parsed["size_cat"]
            occ      = parsed["occ"]
            app_flags = parsed["app_flags"]

            selected_dorm = dorm
            selected_room = room

            is_weekend, season, tod = derive_extra(hour, day, month)
            app_kwh = compute_appliance_kwh(app_flags)

            feature_row = {
                "Temperature":          temp,
                "Humidity":             hum,
                "Wind_Speed":           wind,
                "Avg_Past_Consumption": apc,
                "Hour":                 hour,
                "Day":                  day,
                "Month":                month,
                "IsWeekend":            is_weekend,
                "Season":               season,
                "TimeOfDay":            tod,
                "Is_Anomaly":           0,
                "Dorm_Enc":             DORM_MAP[dorm],
                "Room_Enc":             ROOM_MAP[room],
                "RoomSize_Enc":         SIZE_MAP[size_cat],
                "Num_Occupants":        occ,
                **app_flags,
                "Appliance_kWh_Active": app_kwh,
            }

            features_df = pd.DataFrame([feature_row])[FEATURE_COLS]
            prediction  = round(float(model.predict(features_df)[0]), 4)
            pred_status = "High Consumption" if prediction > ANOMALY_THRESHOLD else "Normal"
            pred_kwh    = round(prediction * KWH_MAX, 4)
            pred_cost   = round(pred_kwh * PESO_PER_KWH, 2)

            # Per-appliance kWh breakdown for the result panel
            app_kwh_breakdown = [
                {
                    "label":  key.replace("App_", "").replace("_", " "),
                    "brand":  APPLIANCE_INFO[key][0],
                    "model":  APPLIANCE_INFO[key][1],
                    "watts":  APPLIANCE_INFO[key][2],
                    "active": app_flags[key],
                    "kwh":    round(APPLIANCE_INFO[key][2] * 0.5 / 1000, 4),
                }
                for key in APPLIANCE_COLS
            ]

            months_short = ["Jan","Feb","Mar","Apr","May","Jun",
                            "Jul","Aug","Sep","Oct","Nov","Dec"]
            active_apps = [k.replace("App_","").replace("_"," ")
                           for k, v in app_flags.items() if v]
            entry = {
                "dorm": dorm, "room": room,
                "temp": temp, "hum": hum, "wind": wind, "apc": apc,
                "time": f"{hour:02d}:00 · Day {day} · {months_short[month-1]}",
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
    )


if __name__ == "__main__":
    app.run(debug=True)
