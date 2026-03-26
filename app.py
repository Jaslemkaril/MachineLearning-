import io
import base64
import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from flask import Flask, request, render_template

app = Flask(__name__)
model = joblib.load("electricity_model.pkl")

FEATURE_COLS = [
    "Temperature", "Humidity", "Wind_Speed", "Avg_Past_Consumption",
    "Hour", "Day", "Month", "IsWeekend", "Season", "TimeOfDay", "Is_Anomaly",
    "Dorm_ID", "Room_ID"
]

SEASON_MAP = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
              6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}

DORM_MAP = {"Dorm A": 0, "Dorm B": 1, "Dorm C": 2}
ROOM_MAP = {f"Room {r}": i for i, r in enumerate(range(101, 109))}

ANOMALY_THRESHOLD = 0.75

# Electricity cost configuration
# Normalized value 1.0 is mapped to KWH_MAX kWh per 30-min reading
KWH_MAX = 2.0
PESO_PER_KWH = 10.50

TOP_ROOMS = [
    {"dorm": "Dorm B", "room": "Room 104", "value": 0.91},
    {"dorm": "Dorm A", "room": "Room 107", "value": 0.88},
    {"dorm": "Dorm C", "room": "Room 102", "value": 0.85},
    {"dorm": "Dorm B", "room": "Room 106", "value": 0.82},
    {"dorm": "Dorm A", "room": "Room 103", "value": 0.79},
]


def derive_extra(hour, day, month):
    """Derive IsWeekend, Season, TimeOfDay from basic time inputs."""
    year = datetime.date.today().year
    try:
        d = datetime.date(year, month, min(day, 28))
        is_weekend = 1 if d.weekday() >= 5 else 0
    except Exception:
        is_weekend = 0
    season = SEASON_MAP.get(month, 0)
    if hour <= 5:
        tod = 0
    elif hour <= 11:
        tod = 1
    elif hour <= 17:
        tod = 2
    else:
        tod = 3
    return is_weekend, season, tod


def build_stats():
    df = pd.read_csv("smart_meter_data.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Hour"]  = df["Timestamp"].dt.hour
    df["Day"]   = df["Timestamp"].dt.day
    df["Month"] = df["Timestamp"].dt.month
    df["IsWeekend"]  = df["Timestamp"].dt.dayofweek.isin([5, 6]).astype(int)
    df["Season"]     = df["Month"].map(SEASON_MAP)
    df["TimeOfDay"]  = pd.cut(df["Hour"], bins=[-1, 5, 11, 17, 23],
                               labels=[0, 1, 2, 3]).astype(int)
    df["Is_Anomaly"] = (df["Anomaly_Label"] != "Normal").astype(int)
    # Assign synthetic Dorm_ID / Room_ID for stats computation (CSV has none)
    np.random.seed(42)
    df["Dorm_ID"] = np.random.randint(0, 3, size=len(df))
    df["Room_ID"] = np.random.randint(0, 10, size=len(df))
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
    cv   = round(float(cross_val_score(model, X, y, cv=5, scoring="r2").mean()), 4)

    # Feature importances (RF) or coefficients (LR)
    if hasattr(model, "feature_importances_"):
        values = list(model.feature_importances_)
        max_val = max(values) or 1
        importances = [
            {
                "feature": f,
                "coef": round(v, 4),
                "width": max(2, round(v / max_val * 100)),
                "color_class": "coeff-bar-pos"
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
                "color_class": "coeff-bar-pos" if c >= 0 else "coeff-bar-neg"
            }
            for f, c in zip(FEATURE_COLS, values)
        ]

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

    model_type = type(model).__name__
    return {"mae": mae, "rmse": rmse, "r2": r2, "cv": cv,
            "importances": importances, "chart": chart_b64,
            "model_type": model_type}


stats = build_stats()
prediction_history = []



@app.route("/", methods=["GET", "POST"])
def index():
    global prediction_history
    prediction = None
    pred_status = None
    selected_dorm = None
    selected_room = None
    pred_kwh = None
    pred_cost = None
    if request.method == "POST":
        try:
            temp  = float(request.form["temperature"])
            hum   = float(request.form["humidity"])
            wind  = float(request.form["wind_speed"])
            apc   = float(request.form["avg_past_consumption"])
            hour  = int(request.form["hour"])
            day   = int(request.form["day"])
            month = int(request.form["month"])
            selected_dorm = request.form.get("dorm_id", "Dorm A")
            selected_room = request.form.get("room_id", "Room 101")
            dorm_enc = DORM_MAP.get(selected_dorm, 0)
            room_enc = ROOM_MAP.get(selected_room, 0)
            is_weekend, season, tod = derive_extra(hour, day, month)
            features = [temp, hum, wind, apc, hour, day, month,
                        is_weekend, season, tod, 0, dorm_enc, room_enc]
            prediction = round(model.predict([features])[0], 4)
            pred_status = "High Consumption" if prediction > ANOMALY_THRESHOLD else "Normal"
            pred_kwh = round(prediction * KWH_MAX, 4)
            pred_cost = round(pred_kwh * PESO_PER_KWH, 2)
            months = ['Jan','Feb','Mar','Apr','May','Jun',
                      'Jul','Aug','Sep','Oct','Nov','Dec']
            prediction_history = [{
                "dorm": selected_dorm, "room": selected_room,
                "temp": temp, "hum": hum, "wind": wind, "apc": apc,
                "time": f"{hour:02d}:00 · Day {day} · {months[month-1]}",
                "result": prediction, "status": pred_status,
                "kwh": pred_kwh, "cost": pred_cost
            }] + prediction_history
            prediction_history = prediction_history[:5]
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("index.html", prediction=prediction,
                           pred_status=pred_status,
                           selected_dorm=selected_dorm,
                           selected_room=selected_room,
                           pred_kwh=pred_kwh, pred_cost=pred_cost,
                           stats=stats, history=prediction_history,
                           top_rooms=TOP_ROOMS)


if __name__ == "__main__":
    app.run(debug=True)
