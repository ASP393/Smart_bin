import os
import csv
import numpy as np
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "results")
err_path = os.path.join(results_dir, "camera_position_errors.csv")

if not os.path.exists(err_path):
    print(f"Error file not found: {err_path}. Run Webots first!")
    exit(1)

times = []
true_pts = []
est_pts = []
errors = []

with open(err_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        times.append(float(r["time"]))
        true_pts.append([float(r["true_x"]), float(r["true_y"]), float(r["true_z"])])
        est_pts.append([float(r["est_x"]), float(r["est_y"]), float(r["est_z"])])
        errors.append(float(r["error_m"]))

if not errors:
    print("No camera error samples recorded yet. Run Webots for a few throws first.")
    exit(0)

true_pts = np.array(true_pts)
est_pts = np.array(est_pts)
errors = np.array(errors)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Smart Dustbin - Camera & RangeFinder Accuracy (Day 8)", fontsize=14, fontweight="bold")

# 1. 3D Trajectory Comparison (X vs Z - Side View)
ax1 = axes[0]
ax1.plot(true_pts[:, 0], true_pts[:, 2], "b-", label="True Trajectory (Ground Truth)", linewidth=2)
ax1.scatter(est_pts[:, 0], est_pts[:, 2], color="r", s=15, alpha=0.7, label="Camera + Depth Estimates")
ax1.set_xlabel("World X (m) [Towards Person]", fontsize=11, fontweight="bold")
ax1.set_ylabel("World Z (m) [Height]", fontsize=11, fontweight="bold")
ax1.set_title("Trash Trajectory: Camera vs. Ground Truth", fontsize=12, fontweight="bold")
ax1.legend(loc="upper right", frameon=True)

# 2. Position Error over time / histogram
ax2 = axes[1]
ax2.plot(times, errors * 100, "g-", linewidth=1.5, label="Euclidean Error (cm)")
mean_err = np.mean(errors) * 100
max_err = np.max(errors) * 100
ax2.axhline(mean_err, color="orange", linestyle="--", label=f"Mean Error: {mean_err:.1f} cm")
ax2.set_xlabel("Simulation Time (s)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Position Error (cm)", fontsize=11, fontweight="bold")
ax2.set_title(f"3D Pinhole Estimation Error (Mean: {mean_err:.1f} cm)", fontsize=12, fontweight="bold")
ax2.set_ylim(0, max(15, max_err * 1.2))
ax2.legend(loc="upper right", frameon=True)

plt.tight_layout()
out_png = os.path.join(results_dir, "camera_position_error.png")
plt.savefig(out_png, dpi=300)
print(f"Saved Day 8 error analysis graph to: {out_png}")
