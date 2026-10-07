"""Is a measurement pose even reachable in this episode?

An episode should only be scored against what the layout allows. A pose counts
as feasible when it is

  * chest-facing (|phi| < 45 deg),
  * inside the standoff band the robot is meant to hold,
  * clear of every obstacle by the robot's radius,
  * in line of sight to the person, and
  * reachable from the robot's start through free space.

Reachability is a flood fill over a grid of the room inflated by the robot
radius, so poses walled off behind furniture do not count.

  python eval/feasibility.py --layout room
"""
import argparse
import os
import sys
from collections import deque

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from radar_env import RadarPoseEnv, ROBOT_RADIUS
from geometry import MIN_CENTER_DISTANCE
from shield import HOLD_MAX_RANGE

CELL = 0.05                 # m, flood-fill resolution
CHEST_HALF_ANGLE = 45.0     # deg, what counts as chest-facing
N_CANDIDATES = 72           # candidate bearings around the person
N_RADII = 8


def free(env, x, y, margin=ROBOT_RADIUS):
    return all(env._surface_distance(o, x, y)[0] > margin for o in env.obstacles)


def line_of_sight(env, x, y):
    """Person visible from (x, y)? Reuses the env's ray caster by moving the
    robot there temporarily."""
    x0, y0 = env.robot_x, env.robot_y
    env.robot_x, env.robot_y = x, y
    d = np.hypot(env.person_x - x, env.person_y - y)
    ok = env._person_visible(d)
    env.robot_x, env.robot_y = x0, y0
    return ok


def reachable_mask(env, lo, hi):
    """Flood fill of free space from the robot's start over the bounding box."""
    (x0, y0), (x1, y1) = lo, hi
    nx, ny = int((x1 - x0) / CELL) + 1, int((y1 - y0) / CELL) + 1
    ok = np.zeros((nx, ny), bool)
    for i in range(nx):
        for j in range(ny):
            ok[i, j] = free(env, x0 + i * CELL, y0 + j * CELL)
    seen = np.zeros_like(ok)
    si, sj = int((env.robot_x - x0) / CELL), int((env.robot_y - y0) / CELL)
    si, sj = np.clip(si, 0, nx - 1), np.clip(sj, 0, ny - 1)
    # The robot is physically at its start, so that cell is free by
    # construction even when the grid rounds it into an inflated obstacle --
    # which happens in the wedged start, where clearance is under a centimetre.
    ok[si, sj] = True
    q = deque([(si, sj)])
    seen[si, sj] = True
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and ok[a, b] and not seen[a, b]:
                seen[a, b] = True
                q.append((a, b))
    return seen, (x0, y0)


def feasible(env):
    """Does a reachable, chest-facing, unoccluded measurement pose exist?"""
    xs = [env.person_x, env.robot_x] + [o[1] for o in env.obstacles]
    ys = [env.person_y, env.robot_y] + [o[2] for o in env.obstacles]
    lo = (min(xs) - 1.0, min(ys) - 1.0)
    hi = (max(xs) + 1.0, max(ys) + 1.0)
    seen, origin = reachable_mask(env, lo, hi)

    for k in range(N_CANDIDATES):
        off = np.radians(-CHEST_HALF_ANGLE + 2 * CHEST_HALF_ANGLE * k / (N_CANDIDATES - 1))
        ang = env.person_phi + off
        for r in np.linspace(MIN_CENTER_DISTANCE + 0.02, HOLD_MAX_RANGE, N_RADII):
            x, y = env.person_x + r * np.cos(ang), env.person_y + r * np.sin(ang)
            i, j = int((x - origin[0]) / CELL), int((y - origin[1]) / CELL)
            if not (0 <= i < seen.shape[0] and 0 <= j < seen.shape[1]) or not seen[i, j]:
                continue
            if free(env, x, y) and line_of_sight(env, x, y):
                return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="room", choices=["empty", "pillars", "room", "bedroom"])
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed0", type=int, default=5000)
    p.add_argument("--tight", action="store_true")
    args = p.parse_args()

    ok = []
    for i in range(args.episodes):
        env = RadarPoseEnv(layout=None if args.layout == "pillars" else args.layout,
                           tight_start=args.tight)
        env.reset(seed=args.seed0 + i)
        ok.append(feasible(env))
    ok = np.array(ok)
    print(f"{args.layout:8s}{' tight' if args.tight else '':6s} "
          f"feasible {ok.sum():3d}/{len(ok)}")
    np.save(f"eval/results/feasible_{args.layout}{'_tight' if args.tight else ''}.npy", ok)


if __name__ == "__main__":
    main()
