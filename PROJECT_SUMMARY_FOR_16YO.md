# Project Summary — Explained For a 16‑Year‑Old

Hey — this file explains everything we made in this project in plain language. If you're 16 and curious about what all these files do, this is for you.

---

## What this project is

Imagine you want to guess how much electricity students use in dorm rooms. This project builds a small web app that can predict short-term electricity use, a model to make those predictions, and a simulator that can create a month of fake dorm electricity data so we can test ideas.

We also made a little appliance list (like a spreadsheet of gadget wattages) so the simulator can pretend which rooms have which gadgets.

---

## Quick list of the important files we created or changed

- `app.py` — The web app server. Open this in a browser to change sliders (like temperature or time of day) and get a predicted electricity value for a 30‑minute window.
- `train_model.py` — The script that trains the machine learning model (Random Forest + a simple linear model). It reads historical meter data and saves a trained model to `electricity_model.pkl`.
- `templates/index.html` — The web page UI: sliders, dropdowns, buttons, and a spot where the prediction shows up.
- `_simulate_dorm_data.py` — The simulator that makes one month of fake dorm data at 30‑minute steps. It writes `simulated_dorm_month.csv` and `room_appliance_ownership.json`.
- `appliance_wattage_catalog.csv` — A small table that lists appliances, typical watts, and how likely each dorm room is to have them.
- `simulated_dorm_month.csv` — The big fake dataset the simulator writes (34,560 rows for 30 days × 48 intervals/day × 24 rooms).
- `room_appliance_ownership.json` — Which room has which appliances (true/false mapping).
- `smart_meter_data.csv` — The real-ish dataset used for training (if you want to retrain models).
- `electricity_model.pkl` — The saved trained model the web app loads to make predictions.
- `Results_Section_Tasks*.docx` — Word documents we created to explain results in different tones and formats.
- `.venv/` — The Python virtual environment (it stores the packages used to run everything).
- `requirements.txt` — A list of Python packages needed to run the code.
- `Procfile`, `render.yaml` — Small deployment files used if you want to host the app on a service like Render.

---

## How the simulator works (simple version)

1. We list common dorm appliances (fan, lamp, mini-fridge, rice cooker, laptop, etc.) and give each one:
   - a typical wattage (how many watts it uses),
   - a chance that any given room “owns” it,
   - a duty cycle or usage pattern (how often it's on).

2. We simulate 3 dorm buildings with 8 rooms each (24 total rooms). For each room, we randomly decide which appliances it has using the "chance" numbers.

3. We step through time in 30‑minute chunks for 30 days (that creates 34,560 rows). For each room and time slot we:
   - check which appliances are on (based on time-of-day rules and randomness),
   - add up their energy for that 30‑minute window (watts × time fraction -> kWh),
   - add a little noise to make things look real,
   - normalize the result to a 0–1 scale (where 1.0 means a configured maximum, `KWH_MAX`, which is 2.0 kWh per 30 minutes in this project).

4. Results are written out as `simulated_dorm_month.csv` and the ownership mapping is saved in `room_appliance_ownership.json`.

This fake data is useful for testing the web UI and for trying to train models that understand appliance-level behavior.

---

## What the main columns in `simulated_dorm_month.csv` mean

- `Timestamp` — The date and time for the 30‑minute slot.
- `Electricity_Consumed` — A normalized number between 0 and 1. Think of it like a percent of the maximum. In this project, `1.0` means `KWH_MAX` (2.0 kWh for a 30‑minute period).
- `Total_kWh` — The actual energy used in that 30‑minute slot (in kilowatt‑hours). Use this to compute cost.
- `Temperature`, `Humidity`, `Wind_Speed` — Simple, normalized environmental features (0–1) used as inputs for the model.
- `Avg_Past_Consumption` — A short rolling average of past consumption for that room (helps the model know recent trends).
- `Anomaly_Label` — Either `Normal` or `Anomaly` (the simulator flags very high values as anomalies).
- `Dorm`, `Room` — The identifiers for the location.
- `Active_Appliances` — A semicolon-separated list of which appliances were on during that interval.
- `Ownership` — A JSON string showing which appliances are owned in that room.

Example: if `Total_kWh` is `0.5` for a 30-minute slot, that means the room used 0.5 kWh in that half hour. To get cost, multiply `Total_kWh` by your electricity price per kWh.

---

## How to run the simulator and app (Windows PowerShell example)

1. Open PowerShell in the project folder.
2. Activate the virtual environment:

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

3. Run the simulator (this creates the CSV and JSON):

```
.\.venv\Scripts\python .\_simulate_dorm_data.py
```

You should see a message like `Wrote simulated_dorm_month.csv rows= 34560`.

4. To run the web app locally (quick method):

```
.\.venv\Scripts\python .\app.py
```

Then open your browser at `http://127.0.0.1:5000` (or whichever address the app prints). Use the sliders and dropdowns to make predictions.

---

## Why we normalized consumption (the 0–1 thing)

Normalization just shrinks numbers so the model learns faster and behaves better. In this project we pick a `KWH_MAX` (2.0 kWh for a 30‑minute slot) and say: if a room used that much or more, call it `1.0`. Everything else is a fraction of that. The simulator also writes `Total_kWh` so you always have the real energy number.

---

## Things we already fixed / improved while building this

- `requirements.txt` encoding was fixed to UTF‑8 so installs don't break.
- `app.py` was made more robust: expensive summary stats are built only when needed (so the web app starts faster).
- Minor consistency fixes so room IDs and random seeds behave predictably.

---

## Ideas for next steps (if you want more)

- Hook `appliance_wattage_catalog.csv` into the web UI so users can pick appliance models and auto-fill wattage.
- Add a small panel on the site that shows `Total_kWh` and estimated cost (kWh × price per kWh).
- Retrain the model including appliance-level features (this would likely improve accuracy).
- Build visualizations: per-room daily usage charts, top appliance contributions, and anomaly dashboards.

---

## I can do any of these for you next

- Add the appliance UI to the page and make the app compute kWh and cost.
- Retrain the model using the simulated data and save a new `electricity_model.pkl`.
- Make quick plots (daily averages, histograms) and include them in a short report.

Tell me which one you want and I’ll do it.

---

File created by the assistant: `PROJECT_SUMMARY_FOR_16YO.md`
