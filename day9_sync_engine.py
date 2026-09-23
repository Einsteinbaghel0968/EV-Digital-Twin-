"""
EV Motor & Drivetrain Digital Twin — Day 9
Digital Twin Synchronization Engine
A twin never knows the real vehicle's parameters exactly. Here the "real vehicle"
uses the true mass; the "twin" starts with a 4% mass error (plausible modeling
error) and receives the SAME throttle/brake commands as the real vehicle.
- twin_no_sync: never corrected -> drifts from reality over time.
- twin_synced:  every 1s, snaps its speed state toward the incoming noisy speed
  sensor reading (a real synchronization step, not just re-simulating).
This demonstrates why synchronization exists: identical control inputs are not
enough to keep a twin's state matching the real system once model parameters
are imperfect.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/claude/evdt")
from day6_engine import EVDrivetrainEngine

rng = np.random.default_rng(7)


def target_speed_kmh(ti):
    v_cruise = 54.0
    if ti < 10: return 0.0
    elif ti < 25: return v_cruise * (ti - 10) / 15
    elif ti < 55: return v_cruise
    elif ti < 70: return v_cruise * (1 - (ti - 55) / 15)
    else: return 0.0


dt = 0.05
sim_duration = 80.0
steps = int(sim_duration / dt)
KP = 0.05
SYNC_INTERVAL_S = 1.0        # how often live telemetry arrives and corrects the twin
SYNC_BLEND = 0.7             # how strongly the twin trusts the incoming sensor at each sync (0-1)
SENSOR_NOISE_KMH = 0.3

# ---------------- "Real vehicle": true parameters, generates the ground-truth run + noisy telemetry ----------------
real = EVDrivetrainEngine()

# ---------------- Digital twin, WITHOUT sync: same commands, imperfect model, never corrected ----------------
twin_no_sync = EVDrivetrainEngine()
twin_no_sync.MASS *= 1.04     # 4% mass error -- a realistic "the twin doesn't know the real car perfectly" gap

# ---------------- Digital twin, WITH sync: same imperfect model, corrected periodically ----------------
twin_synced = EVDrivetrainEngine()
twin_synced.MASS *= 1.04

records = []
next_sync_t = SYNC_INTERVAL_S
sync_events = []

for i in range(steps):
    ti = i * dt

    # --- controller runs on the REAL vehicle's own state (as it would in the field) ---
    target = target_speed_kmh(ti)
    error = target - real.speed * 3.6
    th = float(np.clip(KP * error, 0, 1)) if error >= 0 else 0.0
    br = float(np.clip(-KP * error, 0, 1)) if error < 0 else 0.0

    # same commanded throttle/brake sent to all three (twin receives commands, not raw physics)
    s_real = real.step(th, br, dt)
    s_no_sync = twin_no_sync.step(th, br, dt)
    s_synced = twin_synced.step(th, br, dt)

    # --- periodic synchronization: live noisy speed telemetry arrives, twin corrects toward it ---
    if ti >= next_sync_t:
        noisy_real_speed_kmh = real.speed * 3.6 + rng.normal(0, SENSOR_NOISE_KMH)
        # blend the twin's own predicted speed with the incoming measurement (simple complementary sync)
        corrected_speed_kmh = (1 - SYNC_BLEND) * (twin_synced.speed * 3.6) + SYNC_BLEND * noisy_real_speed_kmh
        twin_synced.speed = corrected_speed_kmh / 3.6
        sync_events.append(ti)
        next_sync_t += SYNC_INTERVAL_S

    records.append({
        "time_s": round(ti, 3),
        "real_speed_kmh": s_real["vehicle_speed_kmh"],
        "twin_no_sync_speed_kmh": s_no_sync["vehicle_speed_kmh"],
        "twin_synced_speed_kmh": s_synced["vehicle_speed_kmh"],
    })

df = pd.DataFrame(records)
df["err_no_sync"] = df.twin_no_sync_speed_kmh - df.real_speed_kmh
df["err_synced"] = df.twin_synced_speed_kmh - df.real_speed_kmh

rmse_no_sync = np.sqrt(np.mean(df.err_no_sync**2))
rmse_synced = np.sqrt(np.mean(df.err_synced**2))
max_no_sync = df.err_no_sync.abs().max()
max_synced = df.err_synced.abs().max()

print(f"Model mismatch: twin mass +4% vs real vehicle (same throttle/brake commands to both)")
print(f"Sync interval: every {SYNC_INTERVAL_S:.0f}s, blend factor {SYNC_BLEND}")
print(f"Number of sync events: {len(sync_events)}")
print()
print(f"RMSE  twin WITHOUT sync: {rmse_no_sync:.3f} km/h   (max error {max_no_sync:.2f} km/h)")
print(f"RMSE  twin WITH sync:    {rmse_synced:.3f} km/h   (max error {max_synced:.2f} km/h)")
print(f"Error reduction from synchronization: {(1 - rmse_synced/rmse_no_sync) * 100:.1f}%")

df.to_csv("/home/claude/evdt/day9_sync_engine.csv", index=False)

# ---------------- Plot ----------------
fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

axs[0].plot(df.time_s, df.real_speed_kmh, color="#1B3A5C", linewidth=2, label="Real vehicle (ground truth)")
axs[0].plot(df.time_s, df.twin_no_sync_speed_kmh, color="#B8462E", linewidth=1.2, linestyle="--",
            label="Twin, no sync (4% mass error, drifts)")
axs[0].plot(df.time_s, df.twin_synced_speed_kmh, color="#4C9A2A", linewidth=1.4,
            label="Twin, synchronized (corrected every 1s)")
for se in sync_events[::5]:  # mark every 5th sync event to avoid clutter
    axs[0].axvline(se, color="grey", alpha=0.15, linewidth=0.8)
axs[0].set_ylabel("Speed (km/h)")
axs[0].set_title("Digital Twin Synchronization \u2014 Same Commands, Imperfect Model")
axs[0].legend(fontsize=9)
axs[0].grid(alpha=0.3)

axs[1].plot(df.time_s, df.err_no_sync, color="#B8462E", linewidth=1.2, linestyle="--",
            label=f"No sync (RMSE={rmse_no_sync:.2f})")
axs[1].plot(df.time_s, df.err_synced, color="#4C9A2A", linewidth=1.4,
            label=f"Synchronized (RMSE={rmse_synced:.2f})")
axs[1].axhline(0, color="grey", linewidth=0.8)
axs[1].set_ylabel("Twin \u2212 Real (km/h)")
axs[1].set_xlabel("Time (s)")
axs[1].legend(fontsize=9)
axs[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/evdt/day9_sync_engine.png", dpi=160)
print("\nSaved day9_sync_engine.png")
print("Saved day9_sync_engine.csv")
