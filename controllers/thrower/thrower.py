import csv
import math
import os
import random
from controller import Supervisor

G = 9.81
RIM_H = 0.6                  # bin rim height (m)
BIN_R = 0.25                 # opening radius (m)
PERSON = (-2.5, 0.0, 1.35)   # the person's hand: the ball starts here
HOME = (0.0, 0.0)            # where the bin waits (same as bin_controller.py)
N_THROWS = 50
MAX_OFFSET = 2.0             # ball lands up to this far from the bin (m)
JUDGE_AFTER = 1.0            # s after the ball crosses the rim, then judge the result
OUT_NAME = "throws_predict.csv"   # use throws_baseline.csv when USE_BRAIN = False
random.seed(1)               # same throws every run

robot = Supervisor()
dt_ms = int(robot.getBasicTimeStep())
trash = robot.getFromDef("TRASH")
bin_node = robot.getFromDef("BIN")
trash_pos = trash.getField("translation")

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
os.makedirs(out_dir, exist_ok=True)
f = open(os.path.join(out_dir, OUT_NAME), "w", newline="")
w = csv.writer(f)
w.writerow(["throw", "offset", "land_x", "land_y", "bin_x", "bin_y", "bin_moved",
            "caught_rim", "inside_final"])

n = 0
n_caught = 0
n_inside = 0
phase = "start"
t_release = 0.0
t_cross = None
t_wait = 0.0
prev = None
caught = None
info = None
launch_v = None
checked = False
bin_start = None
bin_moved = 0.0

while robot.step(dt_ms) != -1:
    now = robot.getTime()

    if phase == "start":
        if n >= N_THROWS:
            f.close()
            print("done:", N_THROWS, "throws saved in", OUT_NAME)
            print("catch rate (rim rule): %.0f%%   physically inside: %.0f%%"
                  % (100.0 * n_caught / N_THROWS, 100.0 * n_inside / N_THROWS))
            break
        # the ball is thrown to land a random distance from where the bin stands now
        b = bin_node.getPosition()
        T = random.uniform(0.8, 1.2)               # flight time to rim height
        off = random.uniform(0, MAX_OFFSET)
        a = random.uniform(0, 2 * math.pi)
        lx = b[0] + off * math.cos(a)
        ly = b[1] + off * math.sin(a)
        vx = (lx - PERSON[0]) / T
        vy = (ly - PERSON[1]) / T
        vz = (RIM_H - PERSON[2] + 0.5 * G * T * T) / T
        trash_pos.setSFVec3f(list(PERSON))         # ball appears in the person's hand
        trash.resetPhysics()                       # step 1: move + reset only
        info = (n + 1, off, lx, ly)
        launch_v = [vx, vy, vz, 0, 0, 0]
        phase = "launch"                           # speed is set in the NEXT step

    elif phase == "launch":
        trash.setVelocity(launch_v)                # step 2: now give it the throw speed
        t_release = now
        caught = None
        t_cross = None
        checked = False
        bin_start = bin_node.getPosition()
        prev = list(PERSON)
        phase = "flying"

    elif phase == "flying":
        p = trash.getPosition()
        if not checked and now - t_release >= 0.1:
            checked = True                         # the ball should be going UP by now
            if p[2] <= PERSON[2]:
                print("WARNING: throw", info[0], "launch failed, ball height =", round(p[2], 2))
        if caught is None and p[2] <= RIM_H:
            # find the exact point where the path crosses rim height
            frac = (prev[2] - RIM_H) / (prev[2] - p[2])
            cx = prev[0] + frac * (p[0] - prev[0])
            cy = prev[1] + frac * (p[1] - prev[1])
            b = bin_node.getPosition()
            caught = math.hypot(cx - b[0], cy - b[1]) <= BIN_R
            bin_moved = math.hypot(b[0] - bin_start[0], b[1] - bin_start[1])
            t_cross = now
        prev = p
        if (t_cross is not None and now - t_cross >= JUDGE_AFTER) or now - t_release >= 6.0:
            b = bin_node.getPosition()
            inside = (abs(p[0] - b[0]) < BIN_R and abs(p[1] - b[1]) < BIN_R
                      and 0.03 < p[2] < RIM_H)
            w.writerow([info[0], round(info[1], 3), round(info[2], 3), round(info[3], 3),
                        round(bin_start[0], 3), round(bin_start[1], 3), round(bin_moved, 3),
                        int(bool(caught)), int(inside)])
            f.flush()
            print("throw", info[0], "| bin moved", round(bin_moved, 2), "m | caught_rim =",
                  bool(caught), "| inside_final =", inside)
            n_caught += int(bool(caught))
            n_inside += int(inside)
            n += 1
            t_wait = now
            phase = "wait_home"

    elif phase == "wait_home":
        # next throw only when the bin is back home and standing still
        b = bin_node.getPosition()
        v = bin_node.getVelocity()
        near = (math.hypot(b[0] - HOME[0], b[1] - HOME[1]) < 0.05
                and math.hypot(v[0], v[1]) < 0.05)
        if near or now - t_wait >= 10.0:
            phase = "start"