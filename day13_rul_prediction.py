"""
EV Motor & Drivetrain Digital Twin — Day 13
Degradation & RUL Prediction
Simulates progressive motor degradation across many drive cycles (bearing/
insulation wear -> rising thermal resistance, power-law wear model -- common
shape in RUL literature). Extracts a per-cycle health indicator (peak motor
temp reached in that cycle), fits a trend using only EARLY cycles, and
predicts the End-of-Life cycle / Remaining Useful Life. Validated against the
TRUE failure cycle, which is known here because we control the ground truth.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
sys.path.insert(0, "/home/claude/evdt")
from day6_engine import EVDrivetrainEngine

rng = np.random.default_rng(13)

def target_speed_kmh(ti, cruise=54.0):
    if ti < 10: return 0.0
    elif ti < 25: return cruise * (ti - 10) / 15
    elif ti < 55: return cruise
    elif ti < 70: return cruise * (1 - (ti - 55) / 15)
    else: return 0.0

def run_one_cycle(eng, r_motor, motor_eff, dt=0.2, duration=80.0, cruise=54.0, kp=0.05, cooldown_s=420.0):
    """Runs one drive cycle on a PERSISTENT engine (thermal state carries over between
    cycles, like a real vehicle) after applying a realistic cool-down gap (parked between
    drives) -- NOT a full reset to ambient. Wear-updated R_motor/eff apply for this cycle."""
    eng.R_motor = r_motor
    eng.MOTOR_EFF = motor_eff
    # closed-form passive cool-down (first-order thermal decay) during the rest period --
    # equivalent to stepping the same RC equation with zero heat input, just solved analytically
    eng.motor_temp = eng.T_amb + (eng.motor_temp - eng.T_amb) * np.exp(-cooldown_s / (r_motor * eng.C_motor))

    steps = int(duration / dt)
    peak_temp = eng.motor_temp
    for i in range(steps):
        ti = i * dt
        target = target_speed_kmh(ti, cruise)
        error = target - eng.speed * 3.6
        th = float(np.clip(kp * error, 0, 1)) if error >= 0 else 0.0
        br = float(np.clip(-kp * error, 0, 1)) if error < 0 else 0.0
        s = eng.step(th, br, dt)
        peak_temp = max(peak_temp, s["motor_temp_c"])
    eng.speed = 0.0  # vehicle parked at end of cycle
    return peak_temp

# =====================================================================
# GROUND TRUTH degradation: thermal resistance worsens with cycle count via
# a power-law wear model (R rises slowly at first, accelerates near end of
# life -- standard bearing/insulation degradation shape).
# =====================================================================
R_MOTOR_0 = 0.06
EFF_0 = 0.93
N_CYCLES = 350
K_WEAR = 3.5e-6
POWER = 2.4
FAILURE_TEMP_C = 65.0   # documented safety limit: motor insulation thermal class threshold

true_r_motor = R_MOTOR_0 * (1 + K_WEAR * np.arange(N_CYCLES) ** POWER)
true_eff = EFF_0 - 0.00025 * np.arange(N_CYCLES)   # slow parallel efficiency decline

print(f"Simulating {N_CYCLES} drive cycles with progressive motor wear (persistent thermal state)...")
persistent_engine = EVDrivetrainEngine()
peak_temps = np.array([
    run_one_cycle(persistent_engine, true_r_motor[c], true_eff[c]) + rng.normal(0, 0.4)
    for c in range(N_CYCLES)
])
cycles = np.arange(N_CYCLES)

true_failure_cycle = np.argmax(peak_temps >= FAILURE_TEMP_C)
if peak_temps[true_failure_cycle] < FAILURE_TEMP_C:
    true_failure_cycle = None
print(f"TRUE failure cycle (peak temp crosses {FAILURE_TEMP_C}C): {true_failure_cycle}")

# =====================================================================
# RUL PREDICTION: using only cycles [0, OBSERVED) -- as if we're standing at
# cycle=OBSERVED today and have NOT seen the future -- fit a trend and
# extrapolate to find the predicted failure cycle.
# =====================================================================
# =====================================================================
# RUL PREDICTION at several "today" points -- standard PHM analysis: does
# prediction accuracy improve as more of the degradation trajectory is
# observed? (Early predictions are known to be unreliable because the
# power-law curve hasn't started accelerating yet -- testing that directly.)
# =====================================================================
def power_law_fit(c, a, b, p):
    return a + b * np.power(np.maximum(c, 1e-6), p)

def predict_at(observed):
    c_obs = cycles[:observed]
    t_obs = peak_temps[:observed]
    try:
        popt, _ = curve_fit(power_law_fit, c_obs, t_obs, p0=[25, 1e-4, 2.0], maxfev=20000)
    except RuntimeError:
        return None, None
    c_fine = np.arange(0, N_CYCLES * 3)
    fitted = power_law_fit(c_fine, *popt)
    idx = np.argmax(fitted >= FAILURE_TEMP_C)
    pred_cycle = idx if fitted[idx] >= FAILURE_TEMP_C else None
    return pred_cycle, popt

OBSERVED_POINTS = [100, 150, 200, 240]
results = []
for obs in OBSERVED_POINTS:
    pred_cycle, popt = predict_at(obs)
    pred_rul = pred_cycle - obs if pred_cycle else None
    true_rul = true_failure_cycle - obs if true_failure_cycle else None
    err_pct = (pred_rul - true_rul) / true_rul * 100 if (pred_rul is not None and true_rul) else None
    results.append({"observed_at_cycle": obs, "predicted_failure_cycle": pred_cycle,
                     "predicted_rul": pred_rul, "true_rul": true_rul, "error_pct": err_pct})
    print(f"Standing at cycle {obs:4d}: predicted failure={pred_cycle}, "
          f"predicted RUL={pred_rul}, true RUL={true_rul}, "
          f"error={err_pct:+.1f}%" if err_pct is not None else f"Standing at cycle {obs}: fit failed")

results_df = pd.DataFrame(results)
results_df.to_csv("/home/claude/evdt/day13_rul_predictions_by_observation_point.csv", index=False)

# keep the OBSERVED=120 case as the "headline" single prediction shown in the main plot
OBSERVED = 120
predicted_failure_cycle, popt = predict_at(OBSERVED)
predicted_rul = predicted_failure_cycle - OBSERVED if predicted_failure_cycle else None
true_rul = true_failure_cycle - OBSERVED if true_failure_cycle else None
if popt is not None:
    c_fine = np.arange(0, N_CYCLES * 2)
    fitted = power_law_fit(c_fine, *popt)

print(f"\nStanding at cycle {OBSERVED} (using only cycles 0-{OBSERVED-1} to fit trend):")
print(f"Predicted failure cycle: {predicted_failure_cycle}   (predicted RUL: {predicted_rul} cycles)")
print(f"True failure cycle:      {true_failure_cycle}   (true RUL: {true_rul} cycles)")
if predicted_rul is not None and true_rul is not None:
    err = predicted_rul - true_rul
    print(f"RUL prediction error: {err:+d} cycles ({err/true_rul*100:+.1f}%)")
print(f"\nSee day13_rul_predictions_by_observation_point.csv for how this error shrinks with more observed cycles.")

# ---------------- save data ----------------
df = pd.DataFrame({"cycle": cycles, "peak_motor_temp_c": peak_temps.round(2),
                    "true_r_motor": true_r_motor.round(5), "true_motor_eff": true_eff.round(5)})
df.to_csv("/home/claude/evdt/day13_degradation_trajectory.csv", index=False)

# ---------------- plot ----------------
fig, ax = plt.subplots(figsize=(11, 6))
ax.scatter(cycles[:OBSERVED], peak_temps[:OBSERVED], s=10, color="#1B3A5C", label="Observed cycles (used for fit)")
ax.scatter(cycles[OBSERVED:], peak_temps[OBSERVED:], s=10, color="#8B96A3", alpha=0.6, label="Future (ground truth, held out)")
if popt is not None:
    ax.plot(c_fine, fitted, color="#4C9A2A", linewidth=1.8, label="Fitted degradation trend (power-law)")
ax.axhline(FAILURE_TEMP_C, color="#B8462E", linestyle="--", linewidth=1.3, label=f"Failure threshold ({FAILURE_TEMP_C:.0f}\u00B0C)")
ax.axvline(OBSERVED, color="grey", linestyle=":", linewidth=1, label=f"'Today' (cycle {OBSERVED})")
if true_failure_cycle:
    ax.axvline(true_failure_cycle, color="#8B96A3", linestyle="-.", linewidth=1, label=f"True failure (cycle {true_failure_cycle})")
if predicted_failure_cycle:
    ax.axvline(predicted_failure_cycle, color="#4C9A2A", linestyle="-.", linewidth=1, label=f"Predicted failure (cycle {predicted_failure_cycle})")
ax.set_xlim(0, min(N_CYCLES, (predicted_failure_cycle or N_CYCLES) + 20))
ax.set_xlabel("Drive cycle number")
ax.set_ylabel("Peak motor temperature per cycle (\u00B0C)")
ax.set_title("Motor Degradation Trend & RUL Prediction (power-law wear model)")
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/evdt/day13_rul_prediction.png", dpi=160)
print("\nSaved day13_rul_prediction.png")
print("Saved day13_degradation_trajectory.csv")
