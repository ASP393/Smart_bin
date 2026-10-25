import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "results")
baseline_file = os.path.join(results_dir, "throws_baseline.csv")
predict_file = os.path.join(results_dir, "throws_predict.csv")

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found!")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = []
        for r in reader:
            data.append({
                "throw": int(r["throw"]),
                "offset": float(r["offset"]),
                "land_x": float(r["land_x"]),
                "land_y": float(r["land_y"]),
                "bin_moved": float(r["bin_moved"]),
                "caught_rim": int(r["caught_rim"]),
                "inside_final": int(r["inside_final"]),
            })
        return data

baseline = load_data(baseline_file)
predict = load_data(predict_file)

if not predict:
    print("No prediction data to plot.")
    exit(1)

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Smart Dustbin - Phase 3 Experimental Results", fontsize=16, fontweight="bold", y=0.98)

# -------------------------------------------------------------
# 1. Overall Catch Rate Comparison (Bar Chart)
# -------------------------------------------------------------
ax1 = axes[0, 0]
categories = ["Baseline (Static Bin)", "Phase 3 (Active Prediction)"]
b_caught = sum(r["caught_rim"] for r in baseline) if baseline else 0
b_total = len(baseline) if baseline else 1
p_caught = sum(r["caught_rim"] for r in predict)
p_total = len(predict)

rates = [(b_caught / b_total) * 100, (p_caught / p_total) * 100]
colors = ["#94a3b8", "#10b981"]

bars = ax1.bar(categories, rates, color=colors, width=0.5, edgecolor="#1e293b", linewidth=1.2)
ax1.set_ylabel("Catch Rate (%)", fontsize=11, fontweight="bold")
ax1.set_ylim(0, 100)
ax1.set_title("Overall Catch Rate (50 Throws)", fontsize=12, fontweight="bold")

for bar, rate, c, tot in zip(bars, rates, [b_caught, p_caught], [b_total, p_total]):
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, y + 2, f"{rate:.1f}%\n({c}/{tot})", 
             ha="center", va="bottom", fontsize=11, fontweight="bold")

# -------------------------------------------------------------
# 2. Catch Rate vs Offset Distance (Binned Comparison)
# -------------------------------------------------------------
ax2 = axes[0, 1]
bins = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
bin_labels = ["0.0 - 0.5 m", "0.5 - 1.0 m", "1.0 - 1.5 m", "1.5 - 2.0 m"]

b_rates = []
p_rates = []

for lo, hi in bins:
    b_subset = [r for r in baseline if lo <= r["offset"] < hi]
    p_subset = [r for r in predict if lo <= r["offset"] < hi]
    
    b_rates.append((sum(r["caught_rim"] for r in b_subset) / len(b_subset) * 100) if b_subset else 0)
    p_rates.append((sum(r["caught_rim"] for r in p_subset) / len(p_subset) * 100) if p_subset else 0)

x = np.arange(len(bin_labels))
width = 0.35

ax2.bar(x - width/2, b_rates, width, label="Baseline (Static)", color="#94a3b8", edgecolor="#334155")
bars_p = ax2.bar(x + width/2, p_rates, width, label="Phase 3 (Predict)", color="#0ea5e9", edgecolor="#0284c7")

ax2.set_xlabel("Throw Distance from Center (Offset)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Catch Rate (%)", fontsize=11, fontweight="bold")
ax2.set_title("Catch Rate vs. Offset Distance", fontsize=12, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(bin_labels)
ax2.set_ylim(0, 110)
ax2.legend(loc="upper right", frameon=True)

for bar in bars_p:
    y = bar.get_height()
    if y > 0:
        ax2.text(bar.get_x() + bar.get_width()/2, y + 2, f"{int(round(y))}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color="#0284c7")

# -------------------------------------------------------------
# 3. Bin Travel Distance vs. Throw Offset (Scatter)
# -------------------------------------------------------------
ax3 = axes[1, 0]
offsets = [r["offset"] for r in predict]
moved = [r["bin_moved"] for r in predict]
caught_status = [r["caught_rim"] for r in predict]

caught_offsets = [o for o, c in zip(offsets, caught_status) if c == 1]
caught_moved = [m for m, c in zip(moved, caught_status) if c == 1]
miss_offsets = [o for o, c in zip(offsets, caught_status) if c == 0]
miss_moved = [m for m, c in zip(moved, caught_status) if c == 0]

ax3.scatter(caught_offsets, caught_moved, color="#10b981", s=60, alpha=0.9, edgecolors="black", label="Caught")
ax3.scatter(miss_offsets, miss_moved, color="#ef4444", s=60, marker="x", linewidth=2, label="Missed")

# Ideal tracking line
ax3.plot([0, 2], [0, 2], linestyle="--", color="#64748b", label="Ideal Tracking (1:1)")
ax3.set_xlabel("Throw Offset Distance (m)", fontsize=11, fontweight="bold")
ax3.set_ylabel("Actual Distance Bin Moved (m)", fontsize=11, fontweight="bold")
ax3.set_title("Bin Movement Response vs. Target Offset", fontsize=12, fontweight="bold")
ax3.set_xlim(0, 2.1)
ax3.set_ylim(0, 2.1)
ax3.legend(loc="upper left", frameon=True)

# -------------------------------------------------------------
# 4. Top-Down Landing Spatial Distribution
# -------------------------------------------------------------
ax4 = axes[1, 1]
c_lx = [r["land_x"] for r in predict if r["caught_rim"] == 1]
c_ly = [r["land_y"] for r in predict if r["caught_rim"] == 1]
m_lx = [r["land_x"] for r in predict if r["caught_rim"] == 0]
m_ly = [r["land_y"] for r in predict if r["caught_rim"] == 0]

# Initial bin position circle
bin_radius = 0.25
bin_circle = plt.Circle((0, 0), bin_radius, color="#3b82f6", fill=True, alpha=0.3, label="Bin Start Area")
ax4.add_patch(bin_circle)

ax4.scatter(c_lx, c_ly, color="#10b981", s=50, label="Trash Caught", edgecolors="black")
ax4.scatter(m_lx, m_ly, color="#ef4444", s=50, marker="x", linewidth=2, label="Trash Missed")

ax4.set_xlabel("World X (m)", fontsize=11, fontweight="bold")
ax4.set_ylabel("World Y (m)", fontsize=11, fontweight="bold")
ax4.set_title("Top-Down Landing Points & Catch Map", fontsize=12, fontweight="bold")
ax4.set_xlim(-2.5, 2.5)
ax4.set_ylim(-2.5, 2.5)
ax4.set_aspect("equal")
ax4.axhline(0, color="#cbd5e1", linestyle=":")
ax4.axvline(0, color="#cbd5e1", linestyle=":")
ax4.legend(loc="upper right", frameon=True)

plt.tight_layout()
dashboard_path = os.path.join(results_dir, "phase3_results_dashboard.png")
plt.savefig(dashboard_path, dpi=300)
print(f"Saved complete dashboard to: {dashboard_path}")

# -------------------------------------------------------------
# Standalone Catch Rate vs Distance (For Report / Presentation)
# -------------------------------------------------------------
fig2, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(bin_labels))
bars1 = ax.bar(x - width/2, b_rates, width, label="Baseline (Stationary Bin)", color="#94a3b8", edgecolor="#334155")
bars2 = ax.bar(x + width/2, p_rates, width, label="Phase 3 (Autonomous Prediction)", color="#0ea5e9", edgecolor="#0369a1")

ax.set_xlabel("Throw Offset Distance (m)", fontsize=12, fontweight="bold")
ax.set_ylabel("Catch Rate (%)", fontsize=12, fontweight="bold")
ax.set_title("Catch Rate vs. Throw Distance from Bin", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(bin_labels, fontsize=11)
ax.set_ylim(0, 115)
ax.legend(fontsize=11, frameon=True)

for bar in bars2:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, y + 2, f"{int(round(y))}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#0369a1")

plt.tight_layout()
single_path = os.path.join(results_dir, "catch_rate_vs_distance.png")
plt.savefig(single_path, dpi=300)
print(f"Saved catch rate vs distance graph to: {single_path}")
