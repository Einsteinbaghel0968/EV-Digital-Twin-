"""
EV Motor & Drivetrain Digital Twin — Day 6
Real-Time Simulation Engine
Stateful step() based simulator (not batch/vectorized like Day 3-5) so it can be
driven live by a dashboard, a control loop, or real-time inputs.
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class EVDrivetrainEngine:
    """Stateful EV motor+drivetrain simulator. Call .step(throttle, brake, dt) repeatedly."""

    def __init__(self):
        # ---- fixed parameters (same physics as Day 3-5) ----
        self.MASS = 1580
        self.WHEEL_RADIUS = 0.316
        self.GEAR_RATIO = 8.19
        self.GEAR_EFF = 0.94
        self.CD = 0.29
        self.FRONTAL_AREA = 2.27
        self.AIR_DENSITY = 1.225
        self.CRR = 0.010
        self.G = 9.81
        self.BATT_NOMINAL_V = 350
        self.BATT_CAPACITY_WH = 40000
        self.MOTOR_MAX_TORQUE = 320
        self.MOTOR_MAX_POWER = 110000
        self.MOTOR_EFF = 0.93
        self.T_amb = 25.0
        self.C_motor, self.R_motor = 4500.0, 0.06

        # ---- live state ----
        self.t = 0.0
        self.speed = 0.0          # m/s
        self.soc = 100.0          # %
        self.motor_temp = self.T_amb
        self.energy_wh = 0.0

        self.history = []

    def step(self, throttle: float, brake: float, dt: float):
        """
        throttle: 0-1 (fraction of max motor torque requested, drive)
        brake:    0-1 (fraction of max motor torque requested, regen/friction)
        dt:       timestep in seconds
        Advances internal state by dt and returns the new state as a dict.
        """
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)

        wheel_omega = self.speed / self.WHEEL_RADIUS
        motor_omega = wheel_omega * self.GEAR_RATIO

        # torque command: positive (drive) or negative (regen braking)
        motor_torque = throttle * self.MOTOR_MAX_TORQUE - brake * self.MOTOR_MAX_TORQUE
        motor_torque = np.clip(motor_torque, -self.MOTOR_MAX_TORQUE, self.MOTOR_MAX_TORQUE)

        motor_power = motor_torque * motor_omega
        if abs(motor_power) > self.MOTOR_MAX_POWER:
            capped_power = np.sign(motor_power) * self.MOTOR_MAX_POWER
            if abs(motor_omega) > 1e-3:
                motor_torque = capped_power / motor_omega
            motor_power = capped_power

        # wheel force from motor torque through gearbox (uses power-limited torque)
        T_wheel = motor_torque * self.GEAR_RATIO * self.GEAR_EFF
        F_motor = T_wheel / self.WHEEL_RADIUS

        # resistive forces
        F_drag = 0.5 * self.AIR_DENSITY * self.CD * self.FRONTAL_AREA * self.speed**2 * np.sign(self.speed if self.speed != 0 else 1)
        F_roll = self.CRR * self.MASS * self.G * (1 if self.speed > 0.01 else 0)

        F_net = F_motor - F_drag - F_roll
        accel = F_net / self.MASS

        # integrate speed (can't go negative from braking through zero)
        new_speed = self.speed + accel * dt
        if self.speed <= 0.0 and new_speed < 0.0:
            new_speed = 0.0
        self.speed = max(new_speed, 0.0)

        # electrical side
        elec_power = motor_power / self.MOTOR_EFF if motor_power >= 0 else motor_power * self.MOTOR_EFF
        battery_voltage = self.BATT_NOMINAL_V - 0.00025 * abs(elec_power)
        battery_current = elec_power / battery_voltage if battery_voltage != 0 else 0.0

        self.energy_wh += elec_power * dt / 3600
        self.soc = float(np.clip(100 - (self.energy_wh / self.BATT_CAPACITY_WH) * 100, 0, 100))

        # thermal
        motor_loss = (1 - self.MOTOR_EFF) * abs(motor_power) + 150
        dTm = (motor_loss - (self.motor_temp - self.T_amb) / self.R_motor) / self.C_motor * dt
        self.motor_temp += dTm

        self.t += dt

        state = {
            "time_s": round(self.t, 3),
            "vehicle_speed_kmh": round(self.speed * 3.6, 3),
            "motor_rpm": round(motor_omega * 60 / (2 * np.pi), 2),
            "motor_torque_nm": round(float(motor_torque), 2),
            "motor_power_kw": round(float(motor_power) / 1000, 3),
            "battery_voltage_v": round(float(battery_voltage), 2),
            "battery_current_a": round(float(battery_current), 2),
            "battery_soc_pct": round(self.soc, 3),
            "motor_temp_c": round(self.motor_temp, 2),
        }
        self.history.append(state)
        return state

    def history_df(self):
        return pd.DataFrame(self.history)


def target_speed_kmh(ti):
    v_cruise = 54.0
    if ti < 10: return 0.0
    elif ti < 25: return v_cruise * (ti - 10) / 15
    elif ti < 55: return v_cruise
    elif ti < 70: return v_cruise * (1 - (ti - 55) / 15)
    else: return 0.0


if __name__ == "__main__":
    engine = EVDrivetrainEngine()
    dt = 0.05
    sim_duration = 80.0
    steps = int(sim_duration / dt)
    KP = 0.05  # simple proportional controller: throttle/brake driven by live speed error

    wall_start = time.time()
    for i in range(steps):
        ti = i * dt
        target = target_speed_kmh(ti)
        error = target - engine.speed * 3.6      # closed-loop: reads engine's CURRENT state
        if error >= 0:
            th, br = float(np.clip(KP * error, 0, 1)), 0.0
        else:
            th, br = 0.0, float(np.clip(-KP * error, 0, 1))
        engine.step(throttle=th, brake=br, dt=dt)
    wall_elapsed = time.time() - wall_start

    df = engine.history_df()
    df.to_csv("/home/claude/evdt/day6_realtime_engine_run.csv", index=False)

    print(df.iloc[[0, 200, 400, 800, 1100, 1400, 1599]].to_string(index=False))
    print(f"\nSimulated {sim_duration:.0f}s of drive cycle in {steps} steps.")
    print(f"Wall-clock compute time: {wall_elapsed:.4f} s  ->  "
          f"{sim_duration / wall_elapsed:.0f}x faster than real-time (single step() call, no I/O)")

    # ---- plot: engine output driven by throttle/brake commands, not a pre-scripted speed curve ----
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    axs[0, 0].plot(df.time_s, df.vehicle_speed_kmh, color="#1B3A5C")
    axs[0, 0].set_title("Vehicle Speed (km/h) \u2014 emergent from throttle/brake inputs")
    axs[0, 0].grid(alpha=0.3)

    axs[0, 1].plot(df.time_s, df.motor_torque_nm, color="#4C9A2A")
    axs[0, 1].set_title("Motor Torque (Nm) \u2014 direct command each step")
    axs[0, 1].grid(alpha=0.3)

    axs[1, 0].plot(df.time_s, df.battery_soc_pct, color="#5A5A5A")
    axs[1, 0].set_title("Battery SOC (%)")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].grid(alpha=0.3)

    axs[1, 1].plot(df.time_s, df.motor_temp_c, color="#B8462E")
    axs[1, 1].set_title("Motor Temperature (\u00b0C)")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("/home/claude/evdt/day6_engine_run.png", dpi=160)
    print("\nSaved day6_engine_run.png")
    print("Saved day6_realtime_engine_run.csv")
