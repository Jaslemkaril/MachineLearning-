import json
import datetime
import threading
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, render_template, jsonify

from config import (
    FEATURE_COLS, APPLIANCE_COLS, APPLIANCE_WATT_COLS, APPLIANCE_INFO,
    DORMS, DORM_MAP, SIZE_MAP, SIZE_CATEGORIES, SIZE_M2_RANGE,
    ANOMALY_THRESHOLD, KWH_MAX, PESO_PER_KWH, HIGH_POWER_WATT_THRESHOLD,
    MAX_HISTORY, ROOMS_PER_DORM,
)

app = Flask(__name__)

# ── Models ────────────────────────────────────────────────────────────────────
try:
    model = joblib.load("electricity_model.pkl")
    print(f"Model loaded: {type(model).__name__}")
except Exception as e:
    print(f"FATAL: Could not load electricity_model.pkl: {e}")
    raise RuntimeError(f"Could not load electricity_model.pkl: {e}")

try:
    classifier = joblib.load("electricity_classifier.pkl")
    print(f"Classifier loaded: {type(classifier).__name__}")
except Exception:
    classifier = None
    print("Warning: No classifier loaded, using threshold-based classification")

# ── Room configuration ────────────────────────────────────────────────────────
ROOM_CONFIG_FILE = Path("room_config.json")

def _load_room_config() -> dict:
    try:
        with open(ROOM_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load room_config.json: {e}")
        return {}

def _save_room_config(config: dict) -> None:
    with open(ROOM_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

ROOM_CONFIG = _load_room_config()
print(f"Room config loaded: {sum(len(rooms) for rooms in ROOM_CONFIG.values())} rooms")

# ── Custom Appliance configuration ─────────────────────────────────────────────
CUSTOM_APPLIANCE_FILE = Path("custom_appliances.json")

def _load_custom_appliances() -> dict:
    try:
        if CUSTOM_APPLIANCE_FILE.exists():
            with open(CUSTOM_APPLIANCE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load custom_appliances.json: {e}")
    return {}

def _save_custom_appliances(config: dict) -> None:
    with open(CUSTOM_APPLIANCE_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

CUSTOM_APPLIANCES = _load_custom_appliances()
print(f"Custom appliances loaded: {len(CUSTOM_APPLIANCES)}")

# Merge default and custom appliances
def get_merged_appliance_info() -> dict:
    merged = dict(APPLIANCE_INFO)
    for key, (watts, label) in CUSTOM_APPLIANCES.items():
        merged[key] = (watts, label)
    return merged

def get_merged_appliance_cols() -> list:
    return list(APPLIANCE_COLS) + list(CUSTOM_APPLIANCES.keys())

# ── Prediction history persistence ───────────────────────────────────────────
HISTORY_FILE = Path("prediction_history.json")

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
    return {
        "mae": "N/A", "rmse": "N/A", "r2": "N/A", "cv": "N/A",
        "importances": [], "chart": None,
        "model_type": type(model).__name__,
        "top_rooms": [], "hourly_data": [0]*24,
        "monthly_chart": {"months": [], "means": [], "stds": [], "counts": []},
        "dorm_chart": {"dorms": [], "means": [], "stds": []},
    }

stats: dict = _load_stats()
print(f"Stats loaded: MAE={stats.get('mae')}, R²={stats.get('r2')}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_room_info(dorm: str, room: str) -> dict:
    """Get room characteristics from config."""
    if dorm in ROOM_CONFIG and room in ROOM_CONFIG[dorm]:
        return ROOM_CONFIG[dorm][room]
    # Fallback defaults
    return {"size_m2": 20, "size_cat": "Medium", "occupants": 2}


def derive_extra(hour: int, day: int, month: int) -> tuple:
    """Derive IsWeekend and TimeOfDay from hour, day, and month."""
    year = datetime.date.today().year
    try:
        d = datetime.date(year, month, min(day, 28))
        is_weekend = 1 if d.weekday() >= 5 else 0
    except Exception:
        is_weekend = 0
    tod = 0 if hour <= 5 else (1 if hour <= 11 else (2 if hour <= 17 else 3))
    return is_weekend, tod


def compute_appliance_kwh(active_flags: dict, custom_watts: dict, appliance_info: dict) -> float:
    """Compute total kWh consumed in a 30-min slot from active appliance flags and custom wattages."""
    total_watts = sum(
        custom_watts.get(key, appliance_info[key][0])
        for key in appliance_info
        if active_flags.get(key, 0)
    )
    return round(total_watts * 0.5 / 1000, 4)   # 30 min = 0.5 h


def get_avg_past_consumption(dorm: str, room: str) -> float:
    """Derive average past consumption from prediction history for this room."""
    with _history_lock:
        room_history = [
            h for h in prediction_history
            if h.get("dorm") == dorm and h.get("room") == room
        ]
    if room_history:
        values = [h.get("result", 0) for h in room_history[:10]]
        return round(sum(values) / len(values), 4)
    # Default: moderate consumption
    return 0.15


def validate_form(form) -> tuple:
    errors = []

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

    hour  = _int("hour",  0, 23, "Hour")
    day   = _int("day",   1, 31, "Day")
    month = _int("month", 1, 12, "Month")

    dorm = form.get("dorm_id", "Dorm A")
    room = form.get("room_id", "Room 1")

    if dorm not in DORM_MAP:
        errors.append(f"Invalid dorm: {dorm!r}")

    # Validate room exists in config
    available_rooms = list(ROOM_CONFIG.get(dorm, {}).keys())
    if room not in available_rooms:
        errors.append(f"Invalid room: {room!r}")

    # Get room characteristics from config
    room_info = get_room_info(dorm, room)
    size_cat = room_info["size_cat"]

    # Get occupancy from user input (allows override of default)
    occ_input = form.get("num_occupants", "").strip()
    if occ_input:
        try:
            occ = int(occ_input)
            if not (1 <= occ <= 4):
                errors.append("Number of occupants must be between 1 and 4.")
                occ = room_info["occupants"]
        except ValueError:
            errors.append("Number of occupants must be a number.")
            occ = room_info["occupants"]
    else:
        occ = room_info["occupants"]

    # Get merged appliance columns (default + custom)
    merged_appliance_cols = get_merged_appliance_cols()
    merged_appliance_info = get_merged_appliance_info()

    # Appliance flags (checkboxes — absent means 0)
    app_flags = {key: 1 if form.get(key) else 0 for key in merged_appliance_cols}

    # Custom wattage values (optional — defaults to APPLIANCE_INFO if not provided)
    custom_watts = {}
    for key in merged_appliance_cols:
        watts_key = f"{key}_watts"
        if watts_key in form:
            try:
                w = int(form.get(watts_key, "0"))
                if 1 <= w <= 5000:
                    custom_watts[key] = w
            except ValueError:
                pass  # Use default if invalid

    # Appliance quantities (1-10 per type)
    app_qty = {}
    for key in merged_appliance_cols:
        qty_key = f"{key}_qty"
        if qty_key in form:
            try:
                q = int(form.get(qty_key, "1"))
                app_qty[key] = max(1, min(10, q))
            except ValueError:
                app_qty[key] = 1
        else:
            app_qty[key] = 1

    if errors:
        return None, " ".join(errors)

    return {
        "hour": hour, "day": day, "month": month,
        "dorm": dorm, "room": room,
        "size_cat": size_cat, "occ": occ,
        "app_flags": app_flags,
        "custom_watts": custom_watts,
        "app_qty": app_qty,
        "merged_appliance_cols": merged_appliance_cols,
        "merged_appliance_info": merged_appliance_info,
    }, None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    """Health check endpoint for Render"""
    return {"status": "ok", "model": type(model).__name__, "stats_loaded": "mae" in stats}


@app.route("/api/rooms/<dorm>")
def api_rooms(dorm):
    """Return room list for a given dorm (for dynamic dropdown update)."""
    rooms = ROOM_CONFIG.get(dorm, {})
    return jsonify(rooms)


@app.route("/add_room", methods=["POST"])
def add_room():
    """Add a new room to a dorm using predefined size categories."""
    global ROOM_CONFIG
    dorm = request.form.get("add_dorm", "").strip()
    size_cat = request.form.get("add_size_cat", "").strip()

    if dorm not in DORMS:
        return render_template("index.html",
                               **_template_context(form_error=f"Invalid dorm: {dorm}"))

    if size_cat not in SIZE_CATEGORIES:
        return render_template("index.html",
                               **_template_context(form_error=f"Invalid size category: {size_cat}"))

    # Determine next room number
    existing_rooms = ROOM_CONFIG.get(dorm, {})
    existing_nums = [int(k.split()[1]) for k in existing_rooms.keys()]
    next_num = max(existing_nums, default=0) + 1
    new_room_key = f"Room {next_num}"

    # Assign m² from the size category range
    lo, hi = SIZE_M2_RANGE[size_cat]
    size_m2 = (lo + hi) // 2  # midpoint

    new_room = {"size_m2": size_m2, "size_cat": size_cat, "occupants": 2}

    if dorm not in ROOM_CONFIG:
        ROOM_CONFIG[dorm] = {}
    ROOM_CONFIG[dorm][new_room_key] = new_room
    _save_room_config(ROOM_CONFIG)

    return render_template("index.html",
                           **_template_context(add_success=f"{new_room_key} ({size_cat}, {size_m2} m²) added to {dorm}"))


@app.route("/add_appliance", methods=["POST"])
def add_appliance():
    """Add a new custom appliance to the system."""
    global CUSTOM_APPLIANCES
    name = request.form.get("appliance_name", "").strip()
    wattage = request.form.get("appliance_wattage", "").strip()

    if not name:
        return render_template("index.html",
                               **_template_context(form_error="Appliance name is required."))

    if not wattage:
        return render_template("index.html",
                               **_template_context(form_error="Wattage is required."))

    try:
        watts = int(wattage)
        if not (1 <= watts <= 10000):
            return render_template("index.html",
                                   **_template_context(form_error="Wattage must be between 1 and 10000."))
    except ValueError:
        return render_template("index.html",
                               **_template_context(form_error="Wattage must be a number."))

    # Generate a key for the appliance (sanitize name)
    key = "App_" + name.replace(" ", "_").replace("-", "_").title()

    # Add to custom appliances
    CUSTOM_APPLIANCES[key] = (watts, name)
    _save_custom_appliances(CUSTOM_APPLIANCES)

    return render_template("index.html",
                           **_template_context(appliance_success=f"{name} ({watts}W) added successfully"))


def _template_context(**extra):
    """Build common template context."""
    with _history_lock:
        history_snapshot = list(prediction_history)

    merged_appliance_info = get_merged_appliance_info()
    merged_appliance_cols = get_merged_appliance_cols()

    ctx = {
        "prediction": extra.get("prediction"),
        "pred_status": extra.get("pred_status"),
        "selected_dorm": extra.get("selected_dorm"),
        "selected_room": extra.get("selected_room"),
        "pred_kwh": extra.get("pred_kwh"),
        "pred_cost": extra.get("pred_cost"),
        "stats": stats,
        "history": history_snapshot,
        "top_rooms": stats.get("top_rooms", []),
        "form_error": extra.get("form_error"),
        "appliance_info": merged_appliance_info,
        "appliance_cols": merged_appliance_cols,
        "app_kwh_breakdown": extra.get("app_kwh_breakdown", []),
        "room_config": ROOM_CONFIG,
        "room_info": extra.get("room_info"),
        "occ": extra.get("occ"),
        "size_categories": SIZE_CATEGORIES,
        "dorms": DORMS,
    }
    ctx.update(extra)
    return ctx


@app.route("/", methods=["GET", "POST"])
def index():
    global prediction_history

    prediction    = None
    pred_status   = None
    selected_dorm = None
    selected_room = None
    pred_kwh      = None
    pred_cost     = None
    monthly_kwh   = None
    monthly_cost  = None
    form_error    = None
    app_kwh_breakdown = []
    room_info     = None
    occ           = None

    if request.method == "POST":
        parsed, form_error = validate_form(request.form)

        if parsed:
            hour     = parsed["hour"]
            day      = parsed["day"]
            month    = parsed["month"]
            dorm     = parsed["dorm"]
            room     = parsed["room"]
            size_cat = parsed["size_cat"]
            occ      = parsed["occ"]
            app_flags = parsed["app_flags"]
            custom_watts = parsed["custom_watts"]
            app_qty = parsed["app_qty"]
            merged_appliance_info = parsed["merged_appliance_info"]

            selected_dorm = dorm
            selected_room = room

            is_weekend, tod = derive_extra(hour, day, month)
            app_kwh = compute_appliance_kwh(app_flags, custom_watts, merged_appliance_info)

            # Auto-derive average past consumption from history
            apc = get_avg_past_consumption(dorm, room)

            # Room encoding
            room_num = int(room.split()[1])
            room_enc = room_num - 1

            # Get room info for display
            room_info = get_room_info(dorm, room)

            # Build per-appliance wattage features (0 if off, W×qty if on)
            appliance_watt_features = {}
            total_active_wattage = 0
            num_active = 0
            has_high_power = 0

            # Handle default appliances (have corresponding model features)
            for key in APPLIANCE_COLS:
                watt_col = f"{key}_W"
                qty = app_qty.get(key, 1)
                if app_flags[key]:
                    w = custom_watts.get(key, APPLIANCE_INFO[key][0]) * qty
                    appliance_watt_features[watt_col] = w
                    total_active_wattage += w
                    num_active += qty
                    if w > HIGH_POWER_WATT_THRESHOLD:
                        has_high_power = 1
                else:
                    appliance_watt_features[watt_col] = 0

            # Handle custom appliances (no corresponding model features, but contribute to totals)
            for key in CUSTOM_APPLIANCES.keys():
                if app_flags.get(key, 0):
                    w = custom_watts.get(key, CUSTOM_APPLIANCES[key][0]) * app_qty.get(key, 1)
                    total_active_wattage += w
                    num_active += app_qty.get(key, 1)
                    if w > HIGH_POWER_WATT_THRESHOLD:
                        has_high_power = 1

            # Build feature row matching FEATURE_COLS order
            feature_row = {
                "Avg_Past_Consumption": apc,
                "Hour":                 hour,
                "Day":                  day,
                "Month":                month,
                "IsWeekend":            is_weekend,
                "TimeOfDay":            tod,
                "Dorm_Enc":             DORM_MAP[dorm],
                "Room_Enc":             room_enc,
                "RoomSize_Enc":         SIZE_MAP[size_cat],
                "Num_Occupants":        occ,
                "Total_Active_Wattage": total_active_wattage,
                "Num_Active_Appliances": num_active,
                "Has_High_Power_Appliance": has_high_power,
                **appliance_watt_features,
            }

            features_df = pd.DataFrame([feature_row])[FEATURE_COLS]

            # Regression prediction
            prediction = round(float(model.predict(features_df)[0]), 4)
            prediction = max(0.0, min(1.0, prediction))  # clip

            # Classification (use classifier if available)
            if classifier is not None:
                clf_pred = classifier.predict(features_df)[0]
                pred_status = "High Consumption" if clf_pred == 1 else "Normal"
            else:
                pred_status = "High Consumption" if prediction > ANOMALY_THRESHOLD else "Normal"

            pred_kwh = round(prediction * KWH_MAX, 4)
            pred_cost = round(pred_kwh * PESO_PER_KWH, 2)

            # Per-appliance kWh breakdown for the result panel
            app_kwh_breakdown = [
                {
                    "label":  merged_appliance_info[key][1],
                    "watts":  custom_watts.get(key, merged_appliance_info[key][0]),
                    "qty":    app_qty.get(key, 1),
                    "active": app_flags[key],
                    "kwh":    round(custom_watts.get(key, merged_appliance_info[key][0]) * app_qty.get(key, 1) * 0.5 / 1000, 4) if app_flags[key] else 0,
                }
                for key in merged_appliance_info
            ]

            months_short = ["Jan","Feb","Mar","Apr","May","Jun",
                            "Jul","Aug","Sep","Oct","Nov","Dec"]
            display_month = months_short[month - 1]
            active_apps = [k.replace("App_","").replace("_"," ")
                           for k, v in app_flags.items() if v]
            entry = {
                "dorm": dorm, "room": room,
                "apc": apc,
                "time": f"{hour:02d}:00 · Day {day} · {display_month}",
                "result": prediction, "status": pred_status,
                "kwh": pred_kwh, "cost": pred_cost,
                "occ": occ, "size": size_cat,
                "appliances": ", ".join(active_apps) if active_apps else "None",
                "app_kwh": app_kwh,
            }

            # ── Monthly Prediction (run model across 24 hours) ────────────
            monthly_predictions = []
            for h in range(24):
                is_we_h, tod_h = derive_extra(h, day, month)
                row_h = {
                    "Avg_Past_Consumption": apc,
                    "Hour": h, "Day": day, "Month": month,
                    "IsWeekend": is_we_h, "TimeOfDay": tod_h,
                    "Dorm_Enc": DORM_MAP[dorm], "Room_Enc": room_enc,
                    "RoomSize_Enc": SIZE_MAP[size_cat], "Num_Occupants": occ,
                    "Total_Active_Wattage": total_active_wattage,
                    "Num_Active_Appliances": num_active,
                    "Has_High_Power_Appliance": has_high_power,
                    **appliance_watt_features,
                }
                df_h = pd.DataFrame([row_h])[FEATURE_COLS]
                pred_h = float(model.predict(df_h)[0])
                monthly_predictions.append(max(0.0, min(1.0, pred_h)))

            # Average across 24 hours × 2 slots/hr × 30 days
            avg_hourly_pred = sum(monthly_predictions) / 24.0
            monthly_kwh = round(avg_hourly_pred * KWH_MAX * 2 * 30, 2)  # 2 slots/hr, 30 days
            monthly_cost = round(monthly_kwh * PESO_PER_KWH, 2)

            with _history_lock:
                prediction_history = [entry] + prediction_history
                prediction_history = prediction_history[:MAX_HISTORY]
                _save_history(prediction_history)

    return render_template(
        "index.html",
        **_template_context(
            prediction=prediction,
            pred_status=pred_status,
            selected_dorm=selected_dorm,
            selected_room=selected_room,
            pred_kwh=pred_kwh,
            pred_cost=pred_cost,
            monthly_kwh=monthly_kwh if prediction is not None else None,
            monthly_cost=monthly_cost if prediction is not None else None,
            form_error=form_error,
            app_kwh_breakdown=app_kwh_breakdown,
            room_info=room_info,
            occ=occ,
        ),
    )


if __name__ == "__main__":
    app.run(debug=True)
