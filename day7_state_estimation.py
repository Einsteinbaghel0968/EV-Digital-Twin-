"""
EV Motor & Drivetrain Digital Twin — Day 7
Motor/Drivetrain State Estimation
Kalman filter fusing a physics-based constant-acceleration prediction with the
noisy incremental-encoder RPM measurement from Day 5, to recover a cleaner
estimate of true motor RPM than the raw sensor alone.

(Note: an earlier attempt used an OCV-based Kalman filter for battery SOC, but
over an 80s cycle SOC only moves ~0.25% -- the OCV signal at that scale is
millivolts, smaller than the voltage sensor's own noise floor. OCV correction
is only meaningful across longer drives with real rest periods, so that's
deferred. Motor RPM is the well-conditioned estimation problem for this stage.)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---------------- physics ground truth (Day 3/4 model) ----------------
MASS = 1580; WHEEL_RADIUS = 0.316; GEAR_RATIO = 8.19; GEAR_EFF = 0.94
CD = 0.29; FRONTAL_AREA = 2.27; AIR_DENSITY = 1.225; CRR = 0.010; G = 9.81
MOTOR_MAX_TORQUE = 320; MOTOR_MAX_POWER = 110000

dt = 0.02  # 50 Hz, matches Day 5 sensor sampling
t = np.arange(0, 80 + dt, dt)

def target_speed(tt):
    v_cruise = 15.0
    v = np.zeros_like(tt)
    for i, ti in enumerate(tt):
        if ti < 10: v[i] = 0
        elif ti < 25: v[i] = v_cruise * (ti - 10) / 15
        elif ti < 55: v[i] = v_cruise
        elif ti < 70: v[i] = v_cruise * (1 - (ti - 55) / 15)
        else: v[i] = 0
    return v

speed_true = target_speed(t)
motor_rpm_true = (speed_true / WHEEL_RADIUS) * GEAR_RATIO * 60 / (2 * np.pi)

# ---------------- Day 5 virtual sensor: noisy, quantized encoder reading ----------------
rpm_sensor = motor_rpm_true.copy()
rpm_sensor = np.round(rpm_sensor / 2.5) * 2.5          # encoder count quantization
rpm_sensor = rpm_sensor + rng.normal(0, 8.0, size=t.shape)  # heavier noise than Day 5 to make filtering worth it

# =====================================================================
# Day 7: 1D Kalman filter, constant-acceleration model
# State x = [rpm, rpm_rate]. Predict with physics-plausible motion model,
# correct with the noisy encoder measurement every step.
# =====================================================================
dt_kf = dt
F = np.array([[1, dt_kf], [0, 1]])         # state transition
H = np.array([[1, 0]])                      # we only measure rpm, not rpm_rate directly
Q = np.array([[0.05, 0], [0, 5.0]])         # process noise (allows real accel/decel changes)
R = np.array([[8.0**2]])                    # measurement noise (matches sensor sigma above)

x = np.array([[0.0], [0.0]])                # initial state estimate
P = np.eye(2) * 100.0                       # initial uncertainty (high -- we don't know yet)

rpm_kf = np.zeros_like(t)
for i in range(len(t)):
    # ---- predict ----
    x = F @ x
    P = F @ P @ F.T + Q

    # ---- correct ----
    z = np.array([[rpm_sensor[i]]])
    y = z - H @ x                            # innovation
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)            # Kalman gain
    x = x + K @ y
    P = (np.eye(2) - K @ H) @ P

    rpm_kf[i] = x[0, 0]

# ---------------- error metrics ----------------
err_raw = rpm_sensor - motor_rpm_true
err_kf = rpm_kf - motor_rpm_true
rmse_raw = np.sqrt(np.mean(err_raw**2))
rmse_kf = np.sqrt(np.mean(err_kf**2))

print(f"RMSE  raw noisy encoder   : {rmse_raw:.3f} RPM")
print(f"RMSE  Kalman-filtered     : {rmse_kf:.3f} RPM")
print(f"RMSE reduction            : {(1 - rmse_kf / rmse_raw) * 100:.1f}%")

df = pd.DataFrame({
    "time_s": t,
    "motor_rpm_true": motor_rpm_true.round(3),
    "motor_rpm_sensor_raw": rpm_sensor.round(2),
    "motor_rpm_kalman_estimated": rpm_kf.round(2),
})
df_out = df.iloc[::5].reset_index(drop=True)  # downsample to 10 Hz for CSV size
df_out.to_csv("/home/claude/evdt/day7_rpm_state_estimation.csv", index=False)

# ---------------- plot ----------------
fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

axs[0].plot(t, rpm_sensor, color="#CBB9AE", linewidth=0.6, label=f"Raw encoder (noisy, RMSE={rmse_raw:.2f})")
axs[0].plot(t, motor_rpm_true, color="#1B3A5C", linewidth=2, label="True RPM (physics ground truth)")
axs[0].plot(t, rpm_kf, color="#4C9A2A", linewidth=1.4, label=f"Kalman-filtered estimate (RMSE={rmse_kf:.2f})")
axs[0].set_ylabel("Motor RPM")
axs[0].set_title("Motor RPM State Estimation \u2014 Kalman Filter vs Raw Encoder")
axs[0].legend(fontsize=9)
axs[0].grid(alpha=0.3)

axs[1].plot(t, err_raw, color="#B8462E", linewidth=0.6, alpha=0.7, label="Raw sensor error")
axs[1].plot(t, err_kf, color="#4C9A2A", linewidth=1.3, label="Kalman-filtered error")
axs[1].axhline(0, color="grey", linewidth=0.8)
axs[1].set_ylabel("Error (RPM)")
axs[1].set_xlabel("Time (s)")
axs[1].legend(fontsize=9)
axs[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/evdt/day7_state_estimation.png", dpi=160)
print("\nSaved day7_state_estimation.png")
print("Saved day7_rpm_state_estimation.csv")
