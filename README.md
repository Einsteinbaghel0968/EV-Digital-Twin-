# EV Motor & Drivetrain Digital Twin

A software-based digital twin of an EV motor and drivetrain system, built end-to-end
over 15 stages: physics modelling → synthetic sensors → real-time simulation engine →
state estimation → synchronization → 3D visualization → AI-based condition monitoring
→ fault diagnosis → degradation/RUL prediction → what-if simulation.

**Reference platform:** Nissan LEAF 40kWh (parameters matched to published specs, see
Day 8 validation below).

**Honest scope:** this is a software-based digital twin *prototype*. All telemetry is
physics-based synthetic data — it is not connected to real hardware. The pipeline
(physics → sensors → sync → AI → prediction → simulation) is real and functional; a
production digital twin would additionally require real sensor ingestion (MQTT/OPC-UA)
and a physical system to synchronize against.

## Pipeline

```
System Modelling → Data → Synchronization → Digital Twin → AI → Prediction → Simulation → Decision
```

## Stage-by-stage summary

| Day | Stage | Key result |
|---|---|---|
| 1 | System Definition & Requirements | Architecture, telemetry list, failure scenarios defined |
| 2 | Motor & Drivetrain Architecture | Battery→Inverter→Motor→Gearbox→Wheels, full parameter sheet |
| 3 | Physics & Mathematical Modelling | Governing equations: F=ma, P=Tω, gearbox torque transfer |
| 4 | Operating Data & Dataset Prep | Synthetic drive-cycle dataset, generated from Day 3 equations |
| 5 | Virtual Sensor & Telemetry Model | Real sensor model (noise, quantization, sampling, thermal lag) |
| 6 | Real-Time Simulation Engine | Stateful `step()` engine, closed-loop cruise control |
| 7 | Motor/Drivetrain State Estimation | Kalman filter, **38.5% RMSE reduction** vs raw noisy encoder |
| 8 | Physics Model Validation | 0-100km/h **-2.9% error**, efficiency **-4.1% error** vs published LEAF specs |
| 9 | Digital Twin Synchronization Engine | Periodic sync, **85.6% error reduction** vs unsynced twin |
| 10 | 3D Digital Twin Visualization | Interactive Three.js console, live component inspection, fault scenarios |
| 11 | AI-Based Condition Monitoring | Isolation Forest on twin residuals, **1.62% FPR, 96-100% detection** |
| 12 | Fault Detection & Diagnosis | Multi-class classifier, **89.2% accuracy** across 3 fault types |
| 13 | Degradation & RUL Prediction | Power-law wear model, RUL error **26.5-77.3%** depending on data observed |
| 14 | What-If / Predictive Simulation | 5-scenario comparison (cargo, cold, aged motor, highway speed) |
| 15 | Complete Integration & Demonstration | This document |

## Real bugs found and fixed along the way

This project surfaced (and fixed) several genuine engineering bugs rather than
producing clean results on the first attempt. Listed honestly because this is the part
that actually demonstrates the physics and the debugging, not just the plots:

1. **Day 6 — power-limit not feeding back into force.** Vehicle accelerated to
   150+ km/h because the motor's power cap was clipped in isolation, not propagated
   back into the torque used for the force calculation. Fixed by recomputing torque
   from the capped power before it reaches the drivetrain.
2. **Day 7 — SOC Kalman filter failed at short timescales.** OCV-based SOC correction
   (the real BMS technique) doesn't work over an 80s cycle — SOC only moves ~0.25%,
   smaller than the voltage sensor's own noise floor. Pivoted to motor RPM estimation,
   the well-conditioned problem for that timescale.
3. **Day 11 — Isolation Forest on raw telemetry barely detected anything (5-6%).**
   A single-signal fault doesn't stand out in 6D raw feature space. Switched to
   twin-vs-actual residuals — then got **0% detection**, because engine and twin were
   identical deterministic physics, so residuals were mathematically exactly zero
   during training (no variance to learn a boundary from). Fixed by reintroducing
   real sensor noise on the system-under-test.
4. **Day 12 — mislabeled training data.** Entire runs were labeled by fault type, so
   a 3-second sensor glitch had its whole 80-second run (77 fault-free seconds
   included) labeled "sensor_glitch." Accuracy was 60.7%. Fixed by labeling ground
   truth to when the fault actually manifests physically. Accuracy jumped to 89.2%.
5. **Day 13 — cold-start temperature ceiling.** Resetting motor temperature to
   ambient every cycle meant peak temp saturated at ~35°C regardless of how badly
   the motor's cooling was degraded (tested up to 50x worse — ceiling never moved).
   Fixed by carrying thermal state across cycles with a realistic cool-down gap.
6. **Day 14 — cosmetic-only fix strikes again.** The cold-weather penalty first
   modified the *returned* telemetry dict, never the engine's actual energy state —
   same mistake as Day 11's first attempt. Zero effect, silently. Fixed by injecting
   the extra resistive loss directly into `energy_wh`.

## Tech stack

Python (NumPy, Pandas, Matplotlib, SciPy, scikit-learn), Three.js (r128, WebGL) for
3D visualization, Jupyter for development. No external physics engine — all dynamics
hand-derived from first principles (Newtonian mechanics, motor torque-speed relations,
thermal RC circuits).

## Limitations / honest disclosure

- No hardware-in-the-loop; all sensor data is physics-based synthetic, clearly labeled
  as such throughout.
- No electronic top-speed limiter modeled (flagged in Day 8).
- OCV-based SOC correction is deferred — only valid over longer drives with rest
  periods (found in Day 7).
- RUL prediction accuracy is limited by how much of the degradation curve has been
  observed — demonstrated directly in Day 13, not just stated.

## Structure

```
physics/        Day 3 equations
engine/          Day 6 stateful EVDrivetrainEngine class
sensors/         Day 5 virtual sensor models
estimation/      Day 7 Kalman filter
sync/            Day 9 synchronization engine
visualization/   Day 10 Three.js 3D console
ai/              Day 11-13 condition monitoring, diagnosis, RUL
whatif/          Day 14 scenario comparison
notebooks/       Jupyter exploration
data/            Generated CSVs per stage
```
