"""
EV Motor & Drivetrain Digital Twin — Day 11
AI-Based Condition Monitoring
Trains an unsupervised anomaly detector (Isolation Forest) on several NORMAL
drive-cycle runs from the Day 6 engine, then tests it against runs with real
physics-level faults injected (not cosmetic overlays -- actual parameter
changes to the engine, same principle as Day 10's scenario re-simulation).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import sys
sys.path.insert(0, "/home/claude/evdt")
from day6_engine import EVDrivetrainEngine

rng = np.random.default_rng(11)

def target_speed_kmh(ti, cruise=54.0):
    if ti < 10: return 0.0
    elif ti < 25: return cruise * (ti - 10) / 15
    elif ti < 55: return cruise
    elif ti < 70: return cruise * (1 - (ti - 55) / 15)
    else: return 0.0

def run_cycle(engine, dt=0.1, duration=80.0, cruise=54.0, kp=0.05, current_spike_at=None, twin=None):
    """Runs `engine` (the system under test) closed-loop. If `twin` is given, it receives
    the SAME throttle/brake commands each step (a nominal-physics reference), so residuals
    between engine and twin can be computed -- this is standard model-based fault detection."""
    rows = []
    steps = int(duration / dt)
    for i in range(steps):
        ti = i * dt
        target = target_speed_kmh(ti, cruise)
        error = target - engine.speed * 3.6
        th = float(np.clip(kp * error, 0, 1)) if error >= 0 else 0.0
        br = float(np.clip(-kp * error, 0, 1)) if error < 0 else 0.0
        s = engine.step(th, br, dt)
        # real sensor noise on the system-under-test readings (Day 5 model) -- without this,
        # engine and twin are identical deterministic physics and residuals are EXACTLY zero,
        # which Isolation Forest cannot learn a boundary from (found this the hard way below)
        s = dict(s)
        s["battery_current_a"] += rng.normal(0, 0.8)
        s["motor_temp_c"] += rng.normal(0, 0.3)
        if current_spike_at is not None and current_spike_at[0] <= ti <= current_spike_at[1]:
            s["battery_current_a"] += 120.0   # simulated current-sensor glitch, real fault-injection
        if twin is not None:
            st = twin.step(th, br, dt)         # twin gets IDENTICAL commands, nominal physics
            s = dict(s)
            s["twin_current_a"] = st["battery_current_a"]
            s["twin_temp_c"] = st["motor_temp_c"]
            s["resid_current"] = s["battery_current_a"] - st["battery_current_a"]
            s["resid_temp"] = s["motor_temp_c"] - st["motor_temp_c"]
        rows.append(s)
    return pd.DataFrame(rows)

FEATURES = ["resid_current", "resid_temp"]

# =====================================================================
# Build TRAINING set: multiple normal runs (engine == nominal twin -> residuals ~ 0 + noise)
# =====================================================================
train_frames = []
for cruise in [45, 50, 54, 58, 62, 48]:
    eng = EVDrivetrainEngine()
    twin = EVDrivetrainEngine()
    df = run_cycle(eng, cruise=cruise, twin=twin)
    train_frames.append(df)
train_df = pd.concat(train_frames, ignore_index=True)

clf = IsolationForest(n_estimators=200, contamination=0.02, random_state=0)
clf.fit(train_df[FEATURES])

# =====================================================================
# TEST 1: held-out NORMAL run (not seen in training) -> false positive check
# =====================================================================
eng_normal = EVDrivetrainEngine()
twin_normal = EVDrivetrainEngine()
df_normal = run_cycle(eng_normal, cruise=54.0, twin=twin_normal)
pred_normal = clf.predict(df_normal[FEATURES])   # -1 = anomaly, 1 = normal
fpr = np.mean(pred_normal == -1) * 100

# =====================================================================
# TEST 2: motor overtemperature fault (real physics change: worse cooling + higher loss)
# =====================================================================
eng_overtemp = EVDrivetrainEngine()
eng_overtemp.R_motor *= 4.0     # degraded cooling (real thermal-resistance change)
eng_overtemp.MOTOR_EFF -= 0.10  # extra internal loss driving heat up
twin_ot = EVDrivetrainEngine()  # twin stays NOMINAL -- doesn't know about the fault, same as reality
df_overtemp = run_cycle(eng_overtemp, cruise=54.0, twin=twin_ot)
pred_overtemp = clf.predict(df_overtemp[FEATURES])
fault_zone_ot = df_overtemp.time_s >= 20
detect_rate_ot = np.mean(pred_overtemp[fault_zone_ot.values] == -1) * 100

# =====================================================================
# TEST 3: reduced motor efficiency fault (real physics change)
# =====================================================================
eng_loweff = EVDrivetrainEngine()
eng_loweff.MOTOR_EFF = 0.78   # down from 0.93 -- real degradation
twin_le = EVDrivetrainEngine()
df_loweff = run_cycle(eng_loweff, cruise=54.0, twin=twin_le)
pred_loweff = clf.predict(df_loweff[FEATURES])
fault_zone_le = df_loweff.time_s >= 12
detect_rate_le = np.mean(pred_loweff[fault_zone_le.values] == -1) * 100

# =====================================================================
# TEST 4: current sensor glitch (transient fault, not a parameter change)
# =====================================================================
eng_spike = EVDrivetrainEngine()
twin_sp = EVDrivetrainEngine()
df_spike = run_cycle(eng_spike, cruise=54.0, current_spike_at=(30.0, 33.0), twin=twin_sp)
pred_spike = clf.predict(df_spike[FEATURES])
fault_zone_sp = (df_spike.time_s >= 30.0) & (df_spike.time_s <= 33.0)
detect_rate_sp = np.mean(pred_spike[fault_zone_sp.values] == -1) * 100

print(f"Training set: {len(train_df)} points across 6 normal runs (cruise 45-62 km/h)")
print()
print(f"TEST 1  Held-out normal run       -> false positive rate: {fpr:.2f}%")
print(f"TEST 2  Motor overtemperature     -> detection rate (t>=20s): {detect_rate_ot:.1f}%")
print(f"TEST 3  Reduced motor efficiency  -> detection rate (t>=12s): {detect_rate_le:.1f}%")
print(f"TEST 4  Current sensor glitch     -> detection rate (30-33s): {detect_rate_sp:.1f}%")

# ---------------- plot: anomaly score over time for each test ----------------
def score_series(clf, df):
    return -clf.decision_function(df[FEATURES])  # higher = more anomalous

fig, axs = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
tests = [
    (df_normal, pred_normal, None, "TEST 1: Held-out Normal Run", "#4C9A2A"),
    (df_overtemp, pred_overtemp, (20, 80), "TEST 2: Motor Overtemperature (fault from t=20s)", "#B8462E"),
    (df_loweff, pred_loweff, (12, 80), "TEST 3: Reduced Motor Efficiency (fault from t=12s)", "#B8462E"),
    (df_spike, pred_spike, (30, 33), "TEST 4: Current Sensor Glitch (fault 30-33s)", "#B8462E"),
]
for ax, (df, pred, zone, title, zcolor) in zip(axs, tests):
    scores = score_series(clf, df)
    ax.plot(df.time_s, scores, color="#1B3A5C", linewidth=1)
    anomalies = df.time_s[pred == -1]
    ax.scatter(anomalies, scores[pred == -1], color="#B8462E", s=8, zorder=3, label="Flagged anomaly")
    if zone:
        ax.axvspan(zone[0], zone[1], color=zcolor, alpha=0.12, label="True fault window")
    ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.set_ylabel("Anomaly score")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
axs[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("/home/claude/evdt/day11_condition_monitoring.png", dpi=150)
print("\nSaved day11_condition_monitoring.png")

summary = pd.DataFrame([
    ["Held-out normal run", "false positive rate", f"{fpr:.2f}%"],
    ["Motor overtemperature", "detection rate (t>=40s)", f"{detect_rate_ot:.1f}%"],
    ["Reduced motor efficiency", "detection rate (t>=12s)", f"{detect_rate_le:.1f}%"],
    ["Current sensor glitch", "detection rate (30-33s)", f"{detect_rate_sp:.1f}%"],
], columns=["Test", "Metric", "Result"])
summary.to_csv("/home/claude/evdt/day11_summary.csv", index=False)
print("\n" + summary.to_string(index=False))
