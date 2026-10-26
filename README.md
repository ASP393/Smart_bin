# 🗑️ Autonomous Smart Dustbin: Mid-Air Ballistic Trash Interception

> **B.Tech IoT Programming Project (CSPC 209 / ITPC 209) — Academic Year 2026**  
> *Autonomous mobile receptacle that predicts 3D projectile flight trajectories and navigates to intercept airborne trash before it hits the ground, powered by Webots R2025a, Pinhole Depth Sensing, and Custom YOLOv8 Object Detection.*

---

## 📌 Project Overview

Traditional "smart dustbins" are completely stationary and merely open their lids when a user stands directly in front of an ultrasonic sensor. This project introduces an **active, autonomous mobile robotic system** capable of:
1. **Detecting thrown trash in mid-air** using an integrated spatial vision suite (RGB Camera + Depth RangeFinder).
2. **Solving the 3D parabolic trajectory** in real-time using least-squares projectile mechanics to predict the exact impact coordinate $(x_{land}, y_{land})$ and crossing time $t_{hit}$.
3. **Navigating an omni-directional 4-wheel mobile chassis** to the predicted landing zone before the object crosses the rim height ($Z = 0.6\text{ m}$).
4. **Deploying Deep Learning (YOLOv8n)** for lighting-invariant, shape-based trash detection running at **70 FPS** on standard CPU hardware.

---

## 📊 Key Performance Metrics

| Metric | Result | Description |
| :--- | :---: | :--- |
| **Baseline Catch Rate** | **12.0%** (6/50) | Receptacle stationary at home position |
| **Active Interception Rate** | **68.0%** (34/50) | Real-time prediction + omni-directional catching |
| **YOLOv8n Inference Speed** | **14.36 ms / frame** | **~70 FPS** real-time throughput on CPU |
| **OpenCV HSV Latency** | **0.21 ms / frame** | **4,860 FPS** ultra-fast fallback tracking |
| **Pinhole Depth Accuracy** | **< 3.8 cm error** | 3D world coordinate estimation from 2D pixel + depth |
| **Physics Step Time** | **16 ms (60 Hz)** | Stable Webots physics simulation loop |

---

## 📁 Repository Directory Structure

```text
smart_bin/
│
├── controllers/                           # Webots robot controller source code
│   ├── bin_controller/
│   │   └── bin_controller.py             # Brain of the smart bin: vision, prediction & motion
│   └── thrower/
│       └── thrower.py                     # Supervisor controller: spawns & launches trash throws
│
├── worlds/                                # Webots simulation world files
│   ├── smart_bin.wbt                      # Main world (Campus Courtyard arena, bin, pedestrian, props)
│   └── smart_bin_backup.wbt               # Safety backup of original world configuration
│
├── dataset/                               # YOLOv8 object detection dataset (600 samples)
│   ├── data.yaml                          # Dataset configuration (classes, image paths)
│   ├── images/
│   │   ├── train/                         # 480 training frames (320x240 RGB)
│   │   └── val/                           # 120 validation frames
│   └── labels/
│       ├── train/                         # YOLO normalized bounding box labels
│       └── val/
│
├── models/                                # Trained neural network weights
│   └── best.pt                            # Best YOLOv8n fine-tuned weights for crumpled trash
│
├── results/                               # Experimental data, CSV logs & analytical plots
│   ├── throws_baseline.csv                # 50 throws with bin standing still (12% catch)
│   ├── throws_predict.csv                 # 50 throws with active prediction (68% catch)
│   ├── camera_position_errors.csv         # Euclidean camera estimation error logs
│   ├── catch_rate_vs_distance.png         # Catch rate decay curve vs. launch offset
│   ├── phase3_results_dashboard.png       # 4-panel experimental results dashboard
│   ├── camera_position_error.png          # Pinhole 3D reconstruction accuracy plot
│   └── colour_vs_yolo_comparison.md       # Benchmarking comparison table
│
├── runs/                                  # Ultralytics YOLOv8 training outputs
│   └── trash_yolov8n/                     # Confusion matrices, PR curves, batch predictions
│
├── train_yolo.py                          # Script to train YOLOv8n model on the custom dataset
├── compare_vision.py                      # Benchmarks HSV Colour vs. YOLOv8n inference latency
├── plot_results.py                        # Generates 4-panel experimental analysis dashboard
├── plot_camera_errors.py                  # Generates camera depth tracking error visualizations
├── requirements.txt                       # Python library dependencies
└── README.md                              # Project documentation and guide
```

---

## ⚙️ Installation & Setup Guide

### 1. Prerequisites
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
- **Python:** Python 3.10 or 3.11 installed
- **Webots Simulator:** Webots **R2025a** (or R2023b+) installed  
  👉 Download Webots: [cyberbotics.com](https://cyberbotics.com/)

---

### 2. Clone Repository
```bash
git clone https://github.com/<your-username>/smart_bin.git
cd smart_bin
```

---

### 3. Install Python Dependencies
Open your terminal in the `smart_bin` folder and install the required dependencies:

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

### 4. Configure Webots Python Runtime (One-Time Setup)
1. Open **Webots**.
2. Go to **`Tools` ➔ `Preferences` ➔ `Python Command`**.
3. Set the Python command to your Python executable:
   - If using system Python on Windows: `python` or `C:\Users\<User>\AppData\Local\Programs\Python\Python310\python.exe`
   - If using the virtual environment: `<path-to-smart_bin>\venv\Scripts\python.exe`

---

## 🚀 How to Run the Simulation

### Step 1: Open World in Webots
1. In Webots, click **`File` ➔ `Open World...`** (or press `Ctrl + O`).
2. Navigate to `smart_bin/worlds/` and open **`smart_bin.wbt`**.
3. You will see the **Smart Campus Courtyard** arena with the pedestrian thrower on the left and the Smart Bin waiting at the home station $(0, 0)$.

---

### Step 2: Select Operational Mode in `bin_controller.py`
Open [`controllers/bin_controller/bin_controller.py`](controllers/bin_controller/bin_controller.py) and check line 17:

```python
# ==============================================================
# MODE SELECTION
# --------------------------------------------------------------
# "YOLO"    : Uses trained YOLOv8 model for deep learning detection
# "CATCH"   : Uses OpenCV HSV colour thresholding + RangeFinder
# "COLLECT" : Auto-saves 600 labeled frames to dataset/ for training
# ==============================================================
MODE = "YOLO"
```

- **`MODE = "YOLO"` (Default / Recommended):** Uses our fine-tuned `models/best.pt` neural network. It includes seamless automatic fallback to OpenCV if the ball is temporarily occluded at extreme distance.
- **`MODE = "CATCH"`:** Uses pure OpenCV HSV color segmentation.
- **`MODE = "COLLECT"`:** Disables catching and collects 600 labeled camera frames to `dataset/`.

---

### Step 3: Run the Simulation
1. Press the **Play** button (or press `Spacebar`) in Webots.
2. The Webots console will output:
   ```text
   ============================================================
   Smart Dustbin Controller Started | Active Mode: YOLO
   [YOLO Loaded] Using weights: .../models/best.pt
   >>> DAY 11 ACTIVE: Running YOLOv8 Deep Learning inference!
   ============================================================
   ```
3. The human pedestrian will launch randomized throws (0–2m offset).
4. Watch the bin dynamically calculate the parabolic curve and accelerate to catch the trash mid-air!
5. After 50 throws, `thrower.py` outputs the final catch percentage and writes results to `results/throws_predict.csv`.

---

## 🧠 Technical Architecture & Algorithm

### 1. Sensing & 3D Pinhole Projection
The bin is equipped with an integrated forward-tilted camera mast ($45^\circ$ elevation angle, $1.8\text{ rad FOV}$):
1. **RGB Camera ($320 \times 240$):** Detects 2D image coordinates $(u, v)$ via YOLOv8 bounding box center or OpenCV centroid.
2. **RangeFinder Sensor:** Queries the exact depth $Z_c = \text{range}[v][u]$.
3. **Inverse Pinhole Camera Math:**
   $$X_c = \frac{(u - C_x) \cdot Z_c}{F_x}, \quad Y_c = \frac{(v - C_y) \cdot Z_c}{F_y}$$
4. **Coordinate Frame Transformation:** Camera coordinates $\mathbf{P}_c$ are mapped to global world coordinates $\mathbf{P}_{world}$ using the bin's current pose $(x_b, y_b, \theta_b)$ and rotation matrix $\mathbf{R}_{cam}$.

---

### 2. Trajectory Prediction (Least-Squares Parabolic Fit)
Given a buffer of $N \ge 5$ observed 3D points $(t_i, x_i, y_i, z_i)$:
- Fit horizontal uniform velocity: $x(t) = x_0 + v_x t$, $y(t) = y_0 + v_y t$
- Fit vertical gravitational ballistic motion: $z(t) = z_0 + v_z t - \frac{1}{2} g t^2$
- Solve for the rim impact time $t_{hit}$ where $z(t_{hit}) = RIM\_H$ ($0.6\text{ m}$):
  $$t_{hit} = \frac{v_z + \sqrt{v_z^2 - 2g(RIM\_H - z_0)}}{g}$$
- The predicted destination is:
  $$\mathbf{P}_{target} = \begin{bmatrix} x_0 + v_x t_{hit} \\ y_0 + v_y t_{hit} \end{bmatrix}$$

---

### 3. Acceleration-Bounded Motion Planning
The holonomic base drives toward the target while strictly adhering to physical motor constraints:
- Maximum speed: $V_{max} = 3.0\text{ m/s}$
- Maximum acceleration: $A_{max} = 5.0\text{ m/s}^2$
- Velocity clamping step:
  $$\mathbf{v}_{next} = \mathbf{v}_{curr} + \text{clamp}\left(\frac{\mathbf{v}_{des} - \mathbf{v}_{curr}}{\Delta t}, -A_{max}, A_{max}\right) \Delta t$$

---

## 🤖 Retraining the YOLOv8 Object Detection Model

If you collect new data or want to retrain the neural network:

```bash
python train_yolo.py
```

### What this script does:
1. Loads the base pretrained `yolov8n.pt` architecture.
2. Trains on the 600 custom labeled images in `dataset/` (480 train, 120 val) for 25 epochs.
3. Automatically uses single-process data loading (`workers=0`) to guarantee stability on Windows CPU/GPU.
4. Saves the best weights directly to `models/best.pt`.
5. Runs a latency benchmark and prints FPS performance.

---

## 📈 Benchmarks & Analysis Scripts

### 1. Vision Benchmark (HSV vs. YOLOv8)
Compare speed and throughput between OpenCV colour thresholding and deep learning:
```bash
python compare_vision.py
```
*Outputs:* Saves the comparison matrix to [`results/colour_vs_yolo_comparison.md`](results/colour_vs_yolo_comparison.md).

---

### 2. Experimental Catch Results Dashboard
Generate the 4-panel visual dashboard comparing static baseline vs. active prediction:
```bash
python plot_results.py
```
*Outputs:* Generates and displays [`results/phase3_results_dashboard.png`](results/phase3_results_dashboard.png).

---

### 3. Camera Depth Reconstruction Error Plot
Visualize tracking Euclidean error between ground truth physics and camera estimates:
```bash
python plot_camera_errors.py
```
*Outputs:* Generates and displays [`results/camera_position_error.png`](results/camera_position_error.png).

---

## 🔧 Troubleshooting FAQ

#### Q1: "Host raw.githubusercontent.com not found" error in Webots?
- **Solution:** We resolved this by using Webots' native built-in `Background` node in `smart_bin.wbt` with zero internet dependency. The world runs 100% offline.

#### Q2: The simulation view is tilted sideways?
- **Solution:** Reload the world (`Ctrl + Shift + R`). The viewpoint has been permanently calibrated to a level, zero-roll perspective (`orientation 1 0 0 1.333`).

#### Q3: "ModuleNotFoundError: No module named 'ultralytics'" in Webots console?
- **Solution:** Webots is using a different Python environment than the one where you ran `pip install`. Open Webots ➔ `Tools` ➔ `Preferences` ➔ `Python Command` and set it to the Python interpreter where your packages are installed.

#### Q4: PyTorch CUDA / GPU error?
- **Solution:** The project automatically detects whether CUDA is available and defaults to CPU mode without crashing. Training and inference run smoothly on standard Intel/AMD CPUs.

---

## 👥 Team & Academic Details

- **Course:** B.Tech IoT Programming (CSPC 209 / ITPC 209)
- **Year:** 2026
- **Branch:** Computer Science & Engineering / Information Technology
- **Project Topic:** Autonomous Mobile Smart Dustbin with Deep Learning & Projectile Prediction

---

## 📜 License
This project is developed for educational and academic research purposes. All simulation assets and algorithms are open for collaborative team use.
