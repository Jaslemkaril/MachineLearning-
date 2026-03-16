import joblib
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
model = joblib.load("electricity_model.pkl")

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
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
