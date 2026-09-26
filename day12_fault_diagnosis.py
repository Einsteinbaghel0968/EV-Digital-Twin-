"""
EV Motor & Drivetrain Digital Twin — Day 12
Fault Detection & Diagnosis
Day 11 answered "is something wrong?" (anomaly vs normal).
Day 12 answers "WHICH fault is it?" -- a multi-class classifier trained on
windowed residual signatures (actual engine vs nominal twin, same architecture
as Day 11/Day 9) for three distinct fault types, each with a genuinely
different residual SHAPE over time:
  - thermal_degradation : resid_temp ramps up slowly (thermal RC lag)
  - efficiency_loss      : resid_current elevated immediately, resid_temp follows with delay
  - sensor_glitch        : resid_current spikes hard and briefly, resid_temp stays ~0 (real physics unaffected)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import sys
sys.path.insert(0, "/home/claude/evdt")
from day6_engine import EVDrivetrainEngine

rng = np.random.default_rng(12)

def target_speed_kmh(ti, cruise=54.0):
    if ti < 10: return 0.0
    elif ti < 25: return cruise * (ti - 10) / 15
    elif ti < 55: return cruise
    elif ti < 70: return cruise * (1 - (ti - 55) / 15)
    else: return 0.0

def run_cycle(engine, twin, dt=0.1, duration=80.0, cruise=54.0, kp=0.05, spike_at=None, spike_mag=120.0):
    rows = []
    steps = int(duration / dt)
    for i in range(steps):
        ti = i * dt
        target = target_speed_kmh(ti, cruise)
        error = target - engine.speed * 3.6
        th = float(np.clip(kp * error, 0, 1)) if error >= 0 else 0.0
        br = float(np.clip(-kp * error, 0, 1)) if error < 0 else 0.0
        s = dict(engine.step(th, br, dt))
        st = twin.step(th, br, dt)
        s["battery_current_a"] += rng.normal(0, 0.8)
        s["motor_temp_c"] += rng.normal(0, 0.3)
        if spike_at is not None and spike_at[0] <= ti <= spike_at[1]:
            s["battery_current_a"] += spike_mag
        s["resid_current"] = s["battery_current_a"] - st["battery_current_a"]
        s["resid_temp"] = s["motor_temp_c"] - st["motor_temp_c"]
        rows.append(s)
    return pd.DataFrame(rows)

def make_run(label, r_motor_mult=1.0, eff_delta=0.0, spike_at=None, spike_mag=120.0, cruise=54.0):
    eng = EVDrivetrainEngine()
    eng.R_motor *= r_motor_mult
    eng.MOTOR_EFF += eff_delta
    twin = EVDrivetrainEngine()
    df = run_cycle(eng, twin, cruise=cruise, spike_at=spike_at, spike_mag=spike_mag)
    # Ground-truth label reflects when the fault actually MANIFESTS physically, not just
    # which run it came from -- a persistent fault (thermal/efficiency) only shows up once
    # torque is nonzero (t~10-70s here); a transient fault (sensor glitch) only shows up
    # during its actual spike window. Labeling the whole run as the fault type (my first
    # attempt) taught the classifier that "normal-looking data" means "sensor_glitch" for
    # 77 of 80 seconds of every glitch run -- that's what broke accuracy down to 60%.
    df["label"] = "normal"
    if label in ("thermal_degradation", "efficiency_loss"):
        df.loc[(df.time_s >= 10) & (df.time_s < 70), "label"] = label
    elif label == "sensor_glitch" and spike_at is not None:
        df.loc[(df.time_s >= spike_at[0]) & (df.time_s <= spike_at[1]), "label"] = label
    return df

WINDOW_S = 2.0

def windowed_features(df, dt=0.1):
    """Rolling window features: mean/max of residuals + rate of change of resid_temp
    (captures the SHAPE difference between fault types, not just magnitude)."""
    win = int(WINDOW_S / dt)
    f = pd.DataFrame({
        "resid_current_mean": df.resid_current.rolling(win).mean(),
        "resid_current_max": df.resid_current.abs().rolling(win).max(),
        "resid_temp_mean": df.resid_temp.rolling(win).mean(),
        "resid_temp_rate": df.resid_temp.diff().rolling(win).mean() / dt,
        "label": df.label,
        "time_s": df.time_s,
    }).dropna().reset_index(drop=True)
    return f

# =====================================================================
# Build TRAINING set: normal + 3 fault types, each with parameter variation
# for diversity (different severity levels), so the classifier learns the
# fault SIGNATURE, not one specific magnitude.
# =====================================================================
train_runs = []
for cr in [48, 54, 60]:
    train_runs.append(make_run("normal", cruise=cr))
for rm in [3.0, 4.0, 5.0]:
    train_runs.append(make_run("thermal_degradation", r_motor_mult=rm, eff_delta=-0.05))
for ed in [-0.10, -0.15, -0.20]:
    train_runs.append(make_run("efficiency_loss", eff_delta=ed))
for mag in [80, 120, 160]:
    train_runs.append(make_run("sensor_glitch", spike_at=(30, 33), spike_mag=mag))

train_feat = pd.concat([windowed_features(r) for r in train_runs], ignore_index=True)
X_train = train_feat[["resid_current_mean", "resid_current_max", "resid_temp_mean", "resid_temp_rate"]]
y_train = train_feat.label

clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0, class_weight="balanced")
clf.fit(X_train, y_train)

# =====================================================================
# TEST: held-out runs, DIFFERENT severities/params than training
# =====================================================================
test_runs = [
    make_run("normal", cruise=51),
    make_run("thermal_degradation", r_motor_mult=3.5, eff_delta=-0.07),
    make_run("efficiency_loss", eff_delta=-0.12),
    make_run("sensor_glitch", spike_at=(30, 33), spike_mag=100),
]
test_feat = pd.concat([windowed_features(r) for r in test_runs], ignore_index=True)
X_test = test_feat[["resid_current_mean", "resid_current_max", "resid_temp_mean", "resid_temp_rate"]]
y_test = test_feat.label
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, digits=3))
labels = ["normal", "thermal_degradation", "efficiency_loss", "sensor_glitch"]
cm = confusion_matrix(y_test, y_pred, labels=labels)
print("Confusion matrix (rows=true, cols=predicted):")
print(pd.DataFrame(cm, index=labels, columns=labels))

feat_importance = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nFeature importances:")
print(feat_importance)

# ---------------- plot: confusion matrix + per-run predicted-label timeline ----------------
fig, axs = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.6]})

im = axs[0].imshow(cm, cmap="Greens")
axs[0].set_xticks(range(len(labels))); axs[0].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
axs[0].set_yticks(range(len(labels))); axs[0].set_yticklabels(labels, fontsize=8)
axs[0].set_xlabel("Predicted"); axs[0].set_ylabel("True")
axs[0].set_title("Fault Diagnosis Confusion Matrix")
for i in range(len(labels)):
    for j in range(len(labels)):
        axs[0].text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=10)

colors = {"normal": "#4C9A2A", "thermal_degradation": "#B8462E", "efficiency_loss": "#D9A441", "sensor_glitch": "#3E7CB1"}
label_to_y = {l: i for i, l in enumerate(labels)}
seen_labels = set()
for run in test_runs:
    f = windowed_features(run)
    pred_run = clf.predict(f[["resid_current_mean", "resid_current_max", "resid_temp_mean", "resid_temp_rate"]])
    y_pred_vals = [label_to_y[p] for p in pred_run]
    for lbl in f.label.unique():
        mask = (f.label == lbl).values
        lab_str = lbl if lbl not in seen_labels else None
        axs[1].scatter(f.time_s[mask], np.array(y_pred_vals)[mask], s=6, color=colors[lbl], alpha=0.6,
                        label=f"true={lbl}" if lab_str else None)
        seen_labels.add(lbl)
axs[1].set_yticks(range(len(labels))); axs[1].set_yticklabels(labels, fontsize=8)
axs[1].set_xlabel("Time (s)"); axs[1].set_title("Predicted Label Over Time (held-out test runs)")
axs[1].legend(fontsize=7, loc="center right")
axs[1].grid(alpha=0.2)

plt.tight_layout()
plt.savefig("/home/claude/evdt/day12_fault_diagnosis.png", dpi=150)
print("\nSaved day12_fault_diagnosis.png")

report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
report_df.to_csv("/home/claude/evdt/day12_classification_report.csv")
print("Saved day12_classification_report.csv")
