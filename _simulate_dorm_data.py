import pandas as pd
import numpy as np
import random
import json
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# Config
start_date = datetime(2026, 4, 1)
days = 30
interval_minutes = 30
interval_hours = interval_minutes / 60.0
periods_per_day = int(24 * 60 / interval_minutes)
num_dorms = 3
rooms_per_dorm = 8
KWH_MAX = 2.0  # normalized 1.0 -> 2.0 kWh per 30-min interval

# Appliance catalog (brand and typical wattage)
appliance_catalog = {
    'Refrigerator': {'brand': 'Condura', 'watts': 150, 'dorm_prob': 0.10, 'duty_cycle': 0.5},
    'MiniFridge':  {'brand': 'Hanabishi', 'watts': 50,  'dorm_prob': 0.35, 'duty_cycle': 0.5},
    'TV32':        {'brand': 'Samsung',  'watts': 50,  'dorm_prob': 0.30, 'duty_cycle': 0.2},
    'RiceCooker':  {'brand': 'Hanabishi', 'watts': 600, 'dorm_prob': 0.65, 'duty_cycle': 0.1},
    'Fan':         {'brand': 'Asahi',    'watts': 60,  'dorm_prob': 0.98, 'duty_cycle': 1.0},
    'LEDLamp':     {'brand': 'Firefly',  'watts': 9,   'dorm_prob': 0.95, 'duty_cycle': 1.0},
    'Desktop':     {'brand': 'Dell',     'watts': 350, 'dorm_prob': 0.08, 'duty_cycle': 0.2},
    'Laptop':      {'brand': 'Dell',     'watts': 65,  'dorm_prob': 0.85, 'duty_cycle': 0.4},
    'PhoneCharger':{'brand': 'Anker/Generic','watts': 10,'dorm_prob': 0.99,'duty_cycle': 0.5},
    'Router':      {'brand': 'PLDT',     'watts': 10,  'dorm_prob': 0.25, 'duty_cycle': 1.0},
    'Kettle':      {'brand': 'Hanabishi', 'watts': 1500,'dorm_prob': 0.25, 'duty_cycle': 0.02},
    'HotPlate':    {'brand': 'Generic',   'watts': 1000,'dorm_prob': 0.15, 'duty_cycle': 0.02},
    'Iron':        {'brand': 'Philips',   'watts': 1000,'dorm_prob': 0.05, 'duty_cycle': 0.01},
    'AC1HP':       {'brand': 'Kolin',     'watts': 1000,'dorm_prob': 0.08, 'duty_cycle': 0.3},
    'WashingMachine':{'brand': 'LG',     'watts': 600, 'dorm_prob': 0.05, 'duty_cycle': 0.02},
    'WaterDispenser':{'brand': 'Eureka',  'watts': 600, 'dorm_prob': 0.02, 'duty_cycle': 0.05},
}

# Time-of-day usage probability modifiers (per appliance)
def usage_prob(appl, hour):
    # hour: 0-23
    if appl == 'Fan':
        return 0.9 if 6 <= hour <= 23 else 0.7
    if appl == 'LEDLamp':
        return 0.8 if (18 <= hour <= 23 or hour <= 6) else 0.05
    if appl == 'RiceCooker':
        return 0.3 if (6 <= hour <= 8 or 11 <= hour <= 13 or 17 <= hour <= 19) else 0.01
    if appl == 'Kettle':
        return 0.25 if (6 <= hour <= 9 or 17 <= hour <= 20) else 0.005
    if appl == 'HotPlate':
        return 0.12 if (11 <= hour <= 13 or 17 <= hour <= 20) else 0.005
    if appl == 'Laptop':
        return 0.6 if (8 <= hour <= 23) else 0.15
    if appl == 'PhoneCharger':
        return 0.7 if (22 <= hour or hour <= 7) else 0.2
    if appl == 'TV32':
        return 0.25 if (19 <= hour <= 23) else 0.03
    if appl == 'AC1HP':
        return 0.25 if (13 <= hour <= 17) else (0.15 if (22 <= hour <= 6) else 0.02)
    if appl == 'Desktop':
        return 0.15 if (9 <= hour <= 22) else 0.02
    if appl == 'MiniFridge' or appl == 'Refrigerator':
        return None  # handled by duty cycle
    if appl == 'Router':
        return 1.0
    if appl == 'WashingMachine':
        return 0.03 if (9 <= hour <= 20) else 0.005
    if appl == 'Iron':
        return 0.02 if (10 <= hour <= 20) else 0.001
    if appl == 'WaterDispenser':
        return 0.05 if (6 <= hour <= 22) else 0.01
    return 0.01

# Build rooms
dorms = [f'Dorm {c}' for c in ['A','B','C']]
rooms = []
for di, dorm in enumerate(dorms):
    for r in range(101, 101 + rooms_per_dorm):
        room = {
            'dorm': dorm,
            'room': f'Room {r}',
            'room_index': r,
            'appliances': {}
        }
        # assign ownership
        for aname, meta in appliance_catalog.items():
            owned = random.random() < meta['dorm_prob']
            # mini fridge and refrigerator are alternatives: prefer MiniFridge for dorm
            room['appliances'][aname] = {
                'owned': bool(owned),
                'watts': meta['watts'],
                'brand': meta['brand'],
                'duty_cycle': meta.get('duty_cycle', 1.0)
            }
        rooms.append(room)

# Some global building-level router/washing machine share adjustments (common in dorms)
# Ensure at least one washing machine per dorm
for di in range(num_dorms):
    dorm_rooms = rooms[di*rooms_per_dorm:(di+1)*rooms_per_dorm]
    if not any(r['appliances']['WashingMachine']['owned'] for r in dorm_rooms):
        # assign one
        random.choice(dorm_rooms)['appliances']['WashingMachine']['owned'] = True

# Simulation loop
rows = []
ownership_summary = {}

def normalize_val(x, vmin=0.0, vmax=1.0):
    return max(0.0, min(1.0, (x - vmin) / (vmax - vmin)))

# Precompute timestamps
total_periods = periods_per_day * days
for room in rooms:
    ownership_summary[f"{room['dorm']}-{room['room']}"] = {k: v['owned'] for k, v in room['appliances'].items()}

# Generate time series per room
for p in range(total_periods):
    ts = start_date + timedelta(minutes=interval_minutes * p)
    hour = ts.hour
    day = ts.day
    month = ts.month

    # environmental patterns (normalized 0-1)
    # simple diurnal sine for temperature and humidity
    t_base = 0.45 + 0.08 * np.sin(2 * np.pi * (hour / 24.0)) + np.random.normal(0, 0.03)
    h_base = 0.55 - 0.10 * np.sin(2 * np.pi * (hour / 24.0)) + np.random.normal(0, 0.04)
    w_base = max(0.0, min(0.3, 0.08 + 0.03 * np.cos(2 * np.pi * (hour / 24.0)) + np.random.normal(0, 0.02)))

    for room_idx, room in enumerate(rooms):
        # compute appliance on/off
        total_kwh = 0.0
        active_list = []
        for aname, adm in room['appliances'].items():
            if not adm['owned']:
                continue
            watts = adm['watts']
            if aname in ['MiniFridge', 'Refrigerator']:
                # duty-cycle based intermittent operation
                on_prob = adm.get('duty_cycle', 0.5)
                # simulate duty: fridge cycles based on random draw
                is_on = random.random() < on_prob
                if is_on:
                    # apply duty rate fraction of interval runtime
                    usage_frac = adm.get('duty_cycle', 0.5)
                    kwh = watts * usage_frac * interval_hours / 1000.0
                    total_kwh += kwh
                    active_list.append(aname)
            else:
                prob = usage_prob(aname, hour)
                is_on = random.random() < (prob)
                if is_on:
                    kwh = watts * interval_hours / 1000.0
                    # small randomization for usage intensity
                    kwh *= random.uniform(0.85, 1.15)
                    total_kwh += kwh
                    active_list.append(aname)

        # add small baseline for always-on tiny devices (phone chargers, router)
        # already included above via probability

        # Add noise and bounding
        total_kwh = max(0.0, total_kwh * random.uniform(0.95, 1.05))
        normalized = total_kwh / KWH_MAX
        # Clip to 1.0
        normalized = max(0.0, min(1.0, normalized))

        # We'll compute Avg_Past_Consumption per room using previous rows for that room
        # For efficiency, look back in rows list for last N entries for the same room
        N = 48  # 24 hours
        lookback = []
        if len(rows) >= 1:
            # find previous entries for same room
            # rows are appended in time-major order, so last occurrence with same room is likely earlier
            cnt = 0
            i = len(rows) - 1
            while i >= 0 and cnt < N:
                r = rows[i]
                if r['Dorm'] == room['dorm'] and r['Room'] == room['room']:
                    lookback.append(r['Electricity_Consumed'])
                    cnt += 1
                i -= 1
        if lookback:
            avg_past = float(np.mean(lookback))
        else:
            avg_past = normalized

        anomaly_label = 'Anomaly' if normalized > 0.75 else 'Normal'

        rows.append({
            'Timestamp': ts.isoformat(sep=' '),
            'Electricity_Consumed': round(float(normalized), 6),
            'Total_kWh': round(float(total_kwh), 6),
            'Temperature': round(float(max(0.0, min(1.0, t_base + np.random.normal(0, 0.01)))), 4),
            'Humidity': round(float(max(0.0, min(1.0, h_base + np.random.normal(0, 0.02)))), 4),
            'Wind_Speed': round(float(w_base), 4),
            'Avg_Past_Consumption': round(float(avg_past), 6),
            'Anomaly_Label': anomaly_label,
            'Dorm': room['dorm'],
            'Room': room['room'],
            'Active_Appliances': ';'.join(active_list),
            'Ownership': json.dumps({k: v['owned'] for k, v in room['appliances'].items()})
        })

# Save outputs
out_df = pd.DataFrame(rows)
out_csv = 'simulated_dorm_month.csv'
out_df.to_csv(out_csv, index=False)
with open('room_appliance_ownership.json', 'w') as f:
    json.dump(ownership_summary, f, indent=2)

print('Wrote', out_csv, 'rows=', len(out_df))
print('Wrote room_appliance_ownership.json with', len(ownership_summary), 'rooms')
