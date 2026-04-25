import io
import json
import base64
import datetime
import threading
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from flask import Flask, request, render_template

app = Flask(__name__)

# ── Model ────────────────────────────────────────────────────────────────────
try:
    model = joblib.load("electricity_model.pkl")
except Exception as e:
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

# ── Stats (background thread) ─────────────────────────────────────────────────
stats: dict = {}
_stats_ready = threading.Event()

def _build_stats() -> None:
    try:
        df = pd.read_csv("smart_meter_data.csv")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Hour"]      = df["Timestamp"].dt.hour
        df["Day"]       = df["Timestamp"].dt.day
        df["Month"]     = df["Timestamp"].dt.month
        df["IsWeekend"] = df["Timestamp"].dt.dayofweek.isin([5, 6]).astype(int)
        df["Season"]    = df["Month"].map(SEASON_MAP)
        df["TimeOfDay"] = pd.cut(df["Hour"], bins=[-1, 5, 11, 17, 23],
                                  labels=[0, 1, 2, 3]).astype(int)
        df["Is_Anomaly"]  = (df["Anomaly_Label"] != "Normal").astype(int)
        df["Dorm_Enc"]    = df["Dorm_ID"].map(DORM_MAP)
        df["Room_Enc"]    = df["Room_ID"].str.extract(r"(\d+)").astype(int) - 101
        df["RoomSize_Enc"] = df["Room_Size_Cat"].map(SIZE_MAP)
        df = df.sort_values("Timestamp")

        X = df[FEATURE_COLS]
        y = df["Electricity_Consumed"]

        train_size = int(len(df) * 0.7)
        X_test = X.iloc[train_size:]
        y_test = y.iloc[train_size:]

        y_pred = model.predict(X_test)

        mae  = round(mean_absolute_error(y_test, y_pred), 4)
        rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
        r2   = round(r2_score(y_test, y_pred), 4)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
        cv = round(float(cv_scores.mean()), 4)

        # Feature importances / coefficients
        if hasattr(model, "feature_importances_"):
            values = list(model.feature_importances_)
            max_val = max(values) or 1
            importances = [
                {
                    "feature": f,
                    "coef": round(v, 4),
                    "width": max(2, round(v / max_val * 100)),
                    "color_class": "coeff-bar-pos",
                }
                for f, v in zip(FEATURE_COLS, values)
            ]
        else:
            values = list(model.coef_)
            max_abs = max(abs(c) for c in values) or 1
            importances = [
                {
                    "feature": f,
                    "coef": round(c, 4),
                    "width": max(2, round(abs(c) / max_abs * 100)),
                    "color_class": "coeff-bar-pos" if c >= 0 else "coeff-bar-neg",
                }
                for f, c in zip(FEATURE_COLS, values)
            ]

        # Top consuming rooms from real data
        top_rooms = _compute_top_rooms(df)

        # Chart
        fig, ax = plt.subplots(figsize=(9, 3.5))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")
        ax.plot(y_test.values[:100], color="#38bdf8", linewidth=1.5, label="Actual")
        ax.plot(y_pred[:100], color="#818cf8", linewidth=1.5, label="Predicted", linestyle="--")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=9)
        ax.set_xlabel("Sample", color="#94a3b8", fontsize=9)
        ax.set_ylabel("Consumption", color="#94a3b8", fontsize=9)
        ax.tick_params(colors="#64748b")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e293b")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0f172a")
        plt.close(fig)
        buf.seek(0)
        chart_b64 = base64.b64encode(buf.read()).decode("utf-8")

        stats.update({
            "mae": mae, "rmse": rmse, "r2": r2, "cv": cv,
            "importances": importances, "chart": chart_b64,
            "model_type": type(model).__name__,
            "top_rooms": top_rooms,
            "error": None,
        })
    except Exception as exc:
        stats.update({"error": str(exc)})
    finally:
        _stats_ready.set()


def _compute_top_rooms(df: pd.DataFrame) -> list:
    """Average actual Electricity_Consumed per dorm/room from the dataset."""
    agg = (
        df.groupby(["Dorm_ID", "Room_ID"])["Electricity_Consumed"]
          .mean()
          .reset_index()
          .rename(columns={"Electricity_Consumed": "value"})
          .sort_values("value", ascending=False)
          .head(5)
    )
    return [
        {"dorm": row.Dorm_ID, "room": row.Room_ID, "value": round(row.value, 4)}
        for row in agg.itertuples()
    ]


threading.Thread(target=_build_stats, daemon=True).start()


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

    stats_ready = _stats_ready.wait(timeout=30)
    current_stats = dict(stats) if stats_ready else {
        "mae": "…", "rmse": "…", "r2": "…", "cv": "…",
        "importances": [], "chart": None,
        "model_type": type(model).__name__,
        "top_rooms": [], "error": None,
    }

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
