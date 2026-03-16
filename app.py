import io
import base64
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

# Pre-compute metrics and chart once at startup
def build_stats():
    df = pd.read_csv("smart_meter_data.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Hour"]  = df["Timestamp"].dt.hour
    df["Day"]   = df["Timestamp"].dt.day
    df["Month"] = df["Timestamp"].dt.month
    df = df.sort_values("Timestamp")

    feature_cols = ["Temperature", "Humidity", "Wind_Speed",
                    "Avg_Past_Consumption", "Hour", "Day", "Month"]
    X = df[feature_cols]
    y = df["Electricity_Consumed"]

    train_size = int(len(df) * 0.7)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    y_pred = model.predict(X_test)

    mae  = round(mean_absolute_error(y_test, y_pred), 4)
    rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
    r2   = round(r2_score(y_test, y_pred), 4)
    cv   = round(float(cross_val_score(model, X, y, cv=5, scoring="r2").mean()), 4)

    max_abs = max(abs(c) for c in model.coef_) or 1
    coeffs = [
        {
            "feature": f,
            "coef": round(c, 4),
            "width": max(2, round(abs(c) / max_abs * 100)),
            "color_class": "coeff-bar-pos" if c >= 0 else "coeff-bar-neg"
        }
        for f, c in zip(feature_cols, model.coef_)
    ]

    # Chart
    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.plot(y_test.values[:100], color="#38bdf8", linewidth=1.5, label="Actual")
    ax.plot(y_pred[:100],        color="#818cf8", linewidth=1.5, label="Predicted", linestyle="--")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=9)
    ax.set_xlabel("Sample", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Consumption", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#64748b")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#0f172a")
    plt.close(fig)
    buf.seek(0)
    chart_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {"mae": mae, "rmse": rmse, "r2": r2, "cv": cv,
            "coeffs": coeffs, "chart": chart_b64}

stats = build_stats()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            features = [
                float(request.form["temperature"]),
                float(request.form["humidity"]),
                float(request.form["wind_speed"]),
                float(request.form["avg_past_consumption"]),
                int(request.form["hour"]),
                int(request.form["day"]),
                int(request.form["month"]),
            ]
            prediction = round(model.predict([features])[0], 4)
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("index.html", prediction=prediction, stats=stats)

if __name__ == "__main__":
    app.run(debug=True)
