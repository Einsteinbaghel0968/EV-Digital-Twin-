"""
EV Motor & Drivetrain Digital Twin — Day 14
What-If / Predictive Simulation
Runs the SAME test protocol (0-100 km/h + steady-cruise efficiency + peak temp)
under several hypothetical operating conditions, so a decision can be made
BEFORE it happens -- exactly the Stage 1 goal: "what-if simulations to
evaluate changes in load, speed, operating conditions before applying them
to the model."
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/claude/evdt")
from day6_engine import EVDrivetrainEngine

def target_speed_kmh(ti, cruise):
    if ti < 10: return 0.0
    elif ti < 25: return cruise * (ti - 10) / 15
    elif ti < 55: return cruise
    elif ti < 70: return cruise * (1 - (ti - 55) / 15)
    else: return 0.0

def make_engine(mass_delta=0, r_motor=0.06, motor_eff=0.93, cold_penalty=0.0):
    """cold_penalty: extra battery voltage-sag coefficient (0 = normal, real EVs lose
    range in cold weather partly from higher internal resistance at low temp)."""
    eng = EVDrivetrainEngine()
    eng.MASS += mass_delta
    eng.R_motor = r_motor
    eng.MOTOR_EFF = motor_eff
    eng._cold_penalty = cold_penalty
    return eng

def step_with_cold(eng, throttle, brake, dt):
    """Wraps engine.step(). A colder battery has higher internal resistance -> more
    I^2R loss for the same useful power delivered, so this adds REAL extra energy draw
    to eng.energy_wh (not just a cosmetic display change -- that was the first, broken,
    version of this function: it edited the returned dict but never touched the actual
    energy state that efficiency/range are computed from, so it silently did nothing)."""
    s = eng.step(throttle, brake, dt)
    penalty = getattr(eng, "_cold_penalty", 0)
    if penalty > 0:
        extra_loss_w = penalty * abs(s["motor_power_kw"] * 1000)   # extra resistive loss, watts
        extra_wh = extra_loss_w * dt / 3600
        eng.energy_wh += extra_wh
        eng.soc = float(np.clip(100 - (eng.energy_wh / eng.BATT_CAPACITY_WH) * 100, 0, 100))
        s = dict(s)
        s["battery_soc_pct"] = round(eng.soc, 3)
        s["battery_current_a"] += extra_loss_w / max(s["battery_voltage_v"], 1)
    return s

def run_0_100(eng, dt=0.01, max_t=25.0):
    t, speed = 0.0, 0.0
    while t < max_t:
        s = step_with_cold(eng, 1.0, 0.0, dt)
        speed = s["vehicle_speed_kmh"]
        t = s["time_s"]
        if speed >= 100.0:
            return t
    return None

def run_cruise_efficiency(eng, cruise_kmh, dt=0.05, warm_s=40.0, measure_s=60.0, kp=0.05):
    def ctrl(err): return (float(np.clip(kp*err,0,1)),0.0) if err>=0 else (0.0,float(np.clip(-kp*err,0,1)))
    t = 0.0
    while t < warm_s:
        err = cruise_kmh - eng.speed*3.6
        th, br = ctrl(err)
        step_with_cold(eng, th, br, dt); t += dt
    e0 = eng.energy_wh; dist = 0.0; t = 0.0; peak_temp = eng.motor_temp
    while t < measure_s:
        err = cruise_kmh - eng.speed*3.6
        th, br = ctrl(err)
        s = step_with_cold(eng, th, br, dt)
        dist += (s["vehicle_speed_kmh"]/3600)*dt
        peak_temp = max(peak_temp, s["motor_temp_c"])
        t += dt
    kwh_100km = ((eng.energy_wh - e0)/1000)/dist*100 if dist>0 else float("nan")
    return kwh_100km, peak_temp

def evaluate_scenario(name, mass_delta=0, r_motor=0.06, motor_eff=0.93, cold_penalty=0.0, cruise_kmh=90):
    eng1 = make_engine(mass_delta=mass_delta, r_motor=r_motor, motor_eff=motor_eff, cold_penalty=cold_penalty)
    t_0_100 = run_0_100(eng1)
    eng2 = make_engine(mass_delta=mass_delta, r_motor=r_motor, motor_eff=motor_eff, cold_penalty=cold_penalty)
    kwh_100km, peak_temp = run_cruise_efficiency(eng2, cruise_kmh=cruise_kmh)
    capacity_wh = eng2.BATT_CAPACITY_WH
    range_km = (capacity_wh/1000) / kwh_100km * 100 if kwh_100km > 0 else float("nan")
    return {"scenario": name, "0_100_time_s": round(t_0_100,2) if t_0_100 else None,
            "efficiency_kwh_100km": round(kwh_100km,2), "peak_motor_temp_c": round(peak_temp,1),
            "est_range_km": round(range_km,1)}

# =====================================================================
# Scenarios
# =====================================================================
results = []
results.append(evaluate_scenario("A. Baseline (normal, no wear)", mass_delta=0, r_motor=0.06, motor_eff=0.93, cruise_kmh=90))
results.append(evaluate_scenario("B. +150kg cargo/passengers", mass_delta=150, r_motor=0.06, motor_eff=0.93, cruise_kmh=90))
results.append(evaluate_scenario("C. Cold weather (approx -20C effect)", mass_delta=0, r_motor=0.06, motor_eff=0.93, cruise_kmh=90, cold_penalty=0.06))
results.append(evaluate_scenario("D. Aged motor (Day 13, cycle 240 wear)", mass_delta=0, r_motor=0.4048, motor_eff=0.870, cruise_kmh=90))
results.append(evaluate_scenario("E. Higher highway speed (110 km/h cruise)", mass_delta=0, r_motor=0.06, motor_eff=0.93, cruise_kmh=110))

df = pd.DataFrame(results)
print(df.to_string(index=False))
df.to_csv("/home/claude/evdt/day14_whatif_results.csv", index=False)

baseline = df.iloc[0]
print("\nDelta vs Baseline:")
for i, row in df.iloc[1:].iterrows():
    d_range = row.est_range_km - baseline.est_range_km
    d_eff = row.efficiency_kwh_100km - baseline.efficiency_kwh_100km
    d_temp = row.peak_motor_temp_c - baseline.peak_motor_temp_c
    print(f"  {row.scenario:45s} range {d_range:+.1f} km | efficiency {d_eff:+.2f} kWh/100km | peak temp {d_temp:+.1f}C")

# ---------------- plot ----------------
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
labels = [s.split(".")[0] for s in df.scenario]
colors = ["#1B3A5C", "#3E7CB1", "#7CA8C9", "#D9A441", "#B8462E"]

axs[0].bar(labels, df.est_range_km, color=colors)
axs[0].set_title("Estimated Range (km)"); axs[0].grid(alpha=0.3, axis="y")
for i,v in enumerate(df.est_range_km): axs[0].text(i, v+2, f"{v:.0f}", ha="center", fontsize=9)

axs[1].bar(labels, df["0_100_time_s"], color=colors)
axs[1].set_title("0-100 km/h Time (s)"); axs[1].grid(alpha=0.3, axis="y")
for i,v in enumerate(df["0_100_time_s"]):
    if v is not None: axs[1].text(i, v+0.15, f"{v:.1f}", ha="center", fontsize=9)

axs[2].bar(labels, df.peak_motor_temp_c, color=colors)
axs[2].set_title("Peak Motor Temp during test (\u00B0C)"); axs[2].grid(alpha=0.3, axis="y")
for i,v in enumerate(df.peak_motor_temp_c): axs[2].text(i, v+0.5, f"{v:.1f}", ha="center", fontsize=9)

for ax in axs: ax.tick_params(axis="x", labelsize=8)
plt.suptitle("What-If Scenario Comparison \u2014 same drive protocol, different conditions", fontsize=12)
plt.tight_layout()
plt.savefig("/home/claude/evdt/day14_whatif_comparison.png", dpi=150)
print("\nSaved day14_whatif_comparison.png")
print("Saved day14_whatif_results.csv")
