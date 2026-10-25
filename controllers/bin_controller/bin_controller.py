import csv
import math
import os
import random
import sys
import numpy as np
import cv2
from controller import Supervisor

# ==============================================================
# MODE SELECTION (Day 10 & 11)
# --------------------------------------------------------------
# "COLLECT" : Day 10 - Auto-saves 600 labeled YOLO frames to dataset/
# "YOLO"    : Day 11 - Uses trained YOLOv8 model for object detection
# "CATCH"   : Phase 4 - Uses OpenCV colour + RangeFinder detection
# ==============================================================
MODE = "YOLO"

DEBUG = True
USE_BRAIN = True
MAX_DATASET_SAMPLES = 600

G = 9.81
RIM_H = 0.6               # bin rim height (m)
V_MAX = 3.0               # bin max speed (m/s)
A_MAX = 5.0               # bin max acceleration (m/s^2)
NOISE = 0.02              # simulated sensor noise (m)
LATENCY = 0.05            # processing latency (s)
FIRST_SEEN = 0.05         # minimum observation time before tracking (s)
MIN_PTS = 5               # points needed before predicting
HOME = (0.0, 0.0)         # where the bin waits between throws
HOME_Z = 0.36             # bin centre height with chassis + wheels
IDLE_BEFORE_HOME = 1.2    # s after ball is gone before driving home

# Camera intrinsics (matching smart_bin.wbt)
CAM_W = 320
CAM_H = 240
CAM_FOV = 1.8             # radians (~103 degrees)
CAM_OFFSET = np.array([0.0, 0.24, 0.45])  # mast location relative to bin center
THETA_CAM = -0.7854       # tilted 45 degrees towards the thrower (-X)
R_CAM = np.array([
    [math.cos(THETA_CAM), 0.0, math.sin(THETA_CAM)],
    [0.0, 1.0, 0.0],
    [-math.sin(THETA_CAM), 0.0, math.cos(THETA_CAM)]
])
FX = (CAM_W / 2.0) / math.tan(CAM_FOV / 2.0)
FY = FX
CX = CAM_W / 2.0
CY = CAM_H / 2.0

robot = Supervisor()
dt_ms = int(robot.getBasicTimeStep())
dt = dt_ms / 1000.0
me = robot.getSelf()
trash = robot.getFromDef("TRASH")

# Setup Camera and RangeFinder sensors
cam = robot.getDevice("camera")
range_finder = robot.getDevice("range_finder")
if cam is not None:
    cam.enable(dt_ms)
if range_finder is not None:
    range_finder.enable(dt_ms)

# Reset bin to HOME on startup
me.getField("translation").setSFVec3f([HOME[0], HOME[1], HOME_Z])
me.getField("rotation").setSFRotation([0, 0, 1, 0])
me.resetPhysics()

base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "..", "..", "results")
dataset_dir = os.path.join(base_dir, "..", "..", "dataset")
models_dir = os.path.join(base_dir, "..", "..", "models")
os.makedirs(results_dir, exist_ok=True)

# Day 8 camera position error logging
err_log_path = os.path.join(results_dir, "camera_position_errors.csv")
err_file = open(err_log_path, "w", newline="")
err_writer = csv.writer(err_file)
err_writer.writerow(["time", "true_x", "true_y", "true_z", "est_x", "est_y", "est_z", "error_m"])
err_file.flush()

# Load YOLO model if in YOLO mode
yolo_model = None
if MODE == "YOLO":
    try:
        from ultralytics import YOLO
        best_pt = os.path.join(models_dir, "best.pt")
        model_load = best_pt if os.path.exists(best_pt) else "yolov8n.pt"
        yolo_model = YOLO(model_load)
        print(f"[YOLO Loaded] Using weights: {model_load}")
    except Exception as e:
        print(f"[YOLO Error] Could not load YOLO: {e}. Falling back to OpenCV mode.")
        MODE = "CATCH"

sample_count = 0


def line_fit(ts, ys):
    """Least-squares line y = a + b*t. Returns (slope, intercept)."""
    n = len(ts)
    st, sy = sum(ts), sum(ys)
    stt = sum(t * t for t in ts)
    sty = sum(t * y for t, y in zip(ts, ys))
    denom = n * stt - st * st
    if abs(denom) < 1e-12:
        return 0.0, sy / max(n, 1)
    b = (n * sty - st * sy) / denom
    return b, (sy - b * st) / n


def predict(ts, pts):
    """Fit the projectile curve. Return (x, y) where the ball crosses rim height."""
    vx, x0 = line_fit(ts, [p[0] for p in pts])
    vy, y0 = line_fit(ts, [p[1] for p in pts])
    vz, z0 = line_fit(ts, [p[2] + 0.5 * G * t * t for t, p in zip(ts, pts)])
    disc = vz * vz - 2 * G * (RIM_H - z0)
    if disc < 0:
        return None
    t_hit = (vz + math.sqrt(disc)) / G
    return (x0 + vx * t_hit, y0 + vy * t_hit)


def next_velocity(target, pos, v_now):
    """Speed-and-acceleration-limited move toward target."""
    wx = wy = 0.0
    if target is not None:
        dx, dy = target[0] - pos[0], target[1] - pos[1]
        dist = math.hypot(dx, dy)
        if dist > 0.01:
            s = min(V_MAX, math.sqrt(2 * A_MAX * dist))
            wx, wy = s * dx / dist, s * dy / dist
    dvx, dvy = wx - v_now[0], wy - v_now[1]
    n = math.hypot(dvx, dvy)
    lim = A_MAX * dt
    if n > lim:
        dvx, dvy = dvx * lim / n, dvy * lim / n
    return v_now[0] + dvx, v_now[1] + dvy


def save_yolo_sample(img_bgr, true_p, bpos, byaw):
    """Day 10: Automatically save frame & YOLO bounding box label."""
    global sample_count
    if sample_count >= MAX_DATASET_SAMPLES:
        return

    R_bin = np.array([
        [math.cos(byaw), -math.sin(byaw), 0.0],
        [math.sin(byaw),  math.cos(byaw), 0.0],
        [0.0,             0.0,            1.0]
    ])
    p_cam = R_CAM.T @ (R_bin.T @ (np.array(true_p) - np.array(bpos)) - CAM_OFFSET)
    xc, yc, zc = p_cam

    if zc <= 0.1 or zc >= 9.5:
        return

    u = CX + FX * (xc / zc)
    v = CY + FY * (yc / zc)

    # Check image boundaries
    if not (5 <= u < CAM_W - 5 and 5 <= v < CAM_H - 5):
        return

    r = 0.055
    r_px = FX * (r / zc) * 1.2
    w_norm = min(1.0, max(0.02, (2.0 * r_px) / CAM_W))
    h_norm = min(1.0, max(0.02, (2.0 * r_px) / CAM_H))
    xc_norm = min(1.0, max(0.0, u / CAM_W))
    yc_norm = min(1.0, max(0.0, v / CAM_H))

    split = "train" if (sample_count % 5 != 0) else "val"
    fname = f"trash_{sample_count:05d}"

    img_path = os.path.join(dataset_dir, "images", split, f"{fname}.jpg")
    cv2.imwrite(img_path, img_bgr)

    lbl_path = os.path.join(dataset_dir, "labels", split, f"{fname}.txt")
    with open(lbl_path, "w") as f:
        f.write(f"0 {xc_norm:.5f} {yc_norm:.5f} {w_norm:.5f} {h_norm:.5f}\n")

    sample_count += 1
    if sample_count % 50 == 0 or sample_count == MAX_DATASET_SAMPLES:
        print(f"[Day 10 YOLO Collector] Collected {sample_count}/{MAX_DATASET_SAMPLES} samples ({split})")
        if sample_count == MAX_DATASET_SAMPLES:
            print(">>> Day 10 Dataset Complete! Ready to train: run 'python train_yolo.py'")


def detect_trash_vision(bpos, byaw):
    """Detect trash via Camera + RangeFinder."""
    if cam is None or range_finder is None:
        return None
    raw_img = cam.getImage()
    range_arr = range_finder.getRangeImageArray()
    if raw_img is None or range_arr is None:
        return None

    img = np.frombuffer(raw_img, dtype=np.uint8).reshape((CAM_H, CAM_W, 4))[:, :, :3]

    # Day 10 auto-collection hook
    if MODE == "COLLECT" and trash is not None:
        tp = trash.getPosition()
        if tp[2] > RIM_H:
            save_yolo_sample(img, tp, bpos, byaw)

    u, v = None, None

    # Day 11 YOLO Detection
    if MODE == "YOLO" and yolo_model is not None:
        res = yolo_model.predict(img, conf=0.15, verbose=False)
        if res and len(res[0].boxes) > 0:
            box = res[0].boxes[0]
            xyxy = box.xyxy[0].cpu().numpy()
            u = int((xyxy[0] + xyxy[2]) / 2.0)
            v = int((xyxy[1] + xyxy[3]) / 2.0)

    # Phase 4 OpenCV Colour Detection
    if u is None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([30, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 50, 40]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
        m = cv2.moments(mask)
        if m["m00"] >= 1:
            u = int(m["m10"] / m["m00"])
            v = int(m["m01"] / m["m00"])

    if u is None:
        return None

    u = max(0, min(CAM_W - 1, u))
    v = max(0, min(CAM_H - 1, v))

    depth = range_arr[v][u]
    if math.isnan(depth) or math.isinf(depth) or depth <= 0.05 or depth >= 9.5:
        return None

    xc = (u - CX) * depth / FX
    yc = (v - CY) * depth / FY
    zc = depth
    P_cam = np.array([xc, yc, zc])

    R_bin = np.array([
        [math.cos(byaw), -math.sin(byaw), 0.0],
        [math.sin(byaw),  math.cos(byaw), 0.0],
        [0.0,             0.0,            1.0]
    ])
    return np.array(bpos) + R_bin @ (CAM_OFFSET + R_CAM @ P_cam)


# ---------- state ----------
target = None
tracking = False
t0 = 0.0
last_visible = -1e9
n_tracks = 0
obs_t, obs_p, pending = [], [], []
cmd_vx, cmd_vy = 0.0, 0.0
vision_count = 0

print("=" * 60)
print(f"Smart Dustbin Controller Started | Active Mode: {MODE}")
if MODE == "COLLECT":
    print(">>> DAY 10 ACTIVE: Auto-collecting YOLO training images & labels!")
elif MODE == "YOLO":
    print(">>> DAY 11 ACTIVE: Running YOLOv8 Deep Learning inference!")
else:
    print(">>> PHASE 4 ACTIVE: Running OpenCV Colour Detection")
print("=" * 60)
dbg_next = 0.0


def fmt(t):
    return "None" if t is None else "(%.2f,%.2f)" % (t[0], t[1])


while robot.step(dt_ms) != -1:
    now = robot.getTime()
    true_p = trash.getPosition()
    pos = me.getPosition()
    rot = me.getField("rotation").getSFRotation()
    yaw = rot[3] if abs(rot[2]) > 0.5 else 0.0

    detected_p = detect_trash_vision(pos, yaw)
    is_vision = detected_p is not None

    if is_vision:
        vision_count += 1
        err = np.linalg.norm(detected_p - np.array(true_p))
        err_writer.writerow([
            round(now, 3),
            round(true_p[0], 3), round(true_p[1], 3), round(true_p[2], 3),
            round(detected_p[0], 3), round(detected_p[1], 3), round(detected_p[2], 3),
            round(err, 4)
        ])
        err_file.flush()

    # Fallback to simulated observation if outside camera lens
    if detected_p is None and true_p[2] > RIM_H:
        detected_p = [true_p[i] + random.gauss(0, NOISE) for i in range(3)]

    visible = (detected_p is not None) and (true_p[2] > RIM_H)
    if visible:
        last_visible = now
    if visible and not tracking:
        tracking = True
        n_tracks += 1
        t0 = now
        obs_t, obs_p, pending = [], [], []
        target = None
        vision_count = 0
    if not visible and (now - last_visible > 0.20):
        tracking = False

    # Delay / latency buffer
    if tracking and detected_p is not None and (now - t0 >= FIRST_SEEN):
        pending.append((now + LATENCY, now - t0, detected_p))
    while pending and pending[0][0] <= now:
        _, ts, ps = pending.pop(0)
        obs_t.append(ts)
        obs_p.append(ps)

    if USE_BRAIN:
        if tracking and len(obs_t) >= MIN_PTS:
            out = predict(obs_t, obs_p)
            if out is not None:
                if target is None and n_tracks <= 5:
                    print(f"Throw {n_tracks}: Target ({out[0]:.2f}, {out[1]:.2f}) | "
                          f"Bin ({pos[0]:.2f}, {pos[1]:.2f}) | Mode: {MODE}")
                target = out
        elif not tracking and now - last_visible > IDLE_BEFORE_HOME:
            target = HOME
    else:
        target = None

    cmd_vx, cmd_vy = next_velocity(target, pos, (cmd_vx, cmd_vy))
    me.setVelocity([cmd_vx, cmd_vy, 0, 0, 0, 0])

    if DEBUG and 1 <= n_tracks <= 3 and (tracking or target is not None) and now >= dbg_next:
        dbg_next = now + 0.25
        print("t=%.2f throw=%d mode=%s samples=%d bin=(%.2f,%.2f) cmd_v=(%.2f,%.2f)"
              % (now, n_tracks, MODE, sample_count, pos[0], pos[1], cmd_vx, cmd_vy))
