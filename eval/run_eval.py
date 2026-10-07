"""Evaluation protocol for the paper's tables.

Fixed seeds, so every condition sees identical spawns, body orientations and
sensor noise. Deterministic policy. Per-episode records are written to
eval/results/<name>.json, and the tables and figures are generated from those
files rather than by re-running, so every number stays checkable.

  python eval/run_eval.py --layout empty   --rules none
  python eval/run_eval.py --layout empty   --rules shield+hold
  python eval/run_eval.py --layout empty   --rules oracle
  python eval/run_eval.py --layout pillars --rules shield+hold
  python eval/run_eval.py --layout bedroom --rules shield+hold   # held out
  python eval/run_eval.py --layout bedroom --rules shield+hold --tight
  python eval/run_eval.py --layout empty --rules shield+hold --disturb brief
"""
import argparse
import json
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import geometry as g
from radar_env import RadarPoseEnv, ROBOT_RADIUS
from shield import ChestShield, Hold, Reacquire
from eval.oracle import oracle_controller

SEED0 = 5000              # fixed across conditions
N_EPISODES = 100
SETTLE_WINDOW = 50        # last 5s, for stillness metrics
GOOD_READING = 9.0        # predicted quality a reading must reach to be usable
SETTLE_TOL = 0.02         # m -- movement below this counts as stopped
SETTLE_TOL_DEG = 1.0      # deg -- and rotation below this: a robot turning in
                          # place is no more measurable than one driving, since
                          # turning changes which scatterer the phase comes from
MIN_SETTLED_STEPS = 20    # must stay put at least 2s to count as settled
DISTURB_AT = 200          # step at which scripted subject motion happens
DISTURB_WINDOW = 60       # steps after it, for measuring the robot's response

# Which trained policy goes with which layout. The furnished layouts are held
# out of training, so they are evaluated with the pillar-trained policy.
# Which trained policy goes with which layout. The fixed bedroom is never
# trained on: it is evaluated with the policy trained on randomised rooms,
# so it measures transfer to an unseen furnished environment.
MODELS = {"empty": "ppo_radar_empty", "pillars": "ppo_radar_pillars",
          "room": "ppo_radar_room", "bedroom": "ppo_radar_room"}


def load_policy(name):
    venv = VecNormalize.load(f"{name}_vecnorm.pkl", DummyVecEnv([RadarPoseEnv]))
    venv.training = False
    return PPO.load(name), venv


def episode(model, venv, layout, rules, seed, steps, tight=False, disturb=None,
            keep_path=False):
    env = RadarPoseEnv(max_steps=steps, layout=None if layout == "pillars" else layout,
                       tight_start=tight)
    obs, _ = env.reset(seed=seed)
    shield, hold, search = ChestShield(), Hold(), Reacquire()
    act_oracle = oracle_controller(env) if rules == "oracle" else None
    phi0 = env.person_phi

    d0, b0, a0 = env._true_state()
    start_quality = (g.predicted_band_snr(max(d0, 1e-3), np.degrees(b0), np.degrees(a0))
                     if env._person_visible(d0) else g.CARDIAC_NOISE_FLOOR)

    ret = 0.0
    xy = []                    # robot position each step, for settle time
    readings = []              # quality of each completed measurement window
    first_good = None          # step at which the first usable reading finished
    was_measuring = False
    shield_steps = hold_steps = occluded_steps = contact_steps = 0
    clearance = np.inf
    path, settle, disturb_win = [], [], []
    first_move = None
    pose_at_disturb = None
    for t in range(steps):
        # scripted subject motion, once the robot has had time to settle
        if disturb and t == DISTURB_AT:
            env.person_phi = phi0 + np.radians(70 if disturb == "brief" else 90)
            pose_at_disturb = (env.robot_x, env.robot_y)
        if disturb == "brief" and t == DISTURB_AT + 10:
            env.person_phi = phi0

        if act_oracle is not None:
            action, on_shield, on_hold = act_oracle(), False, False
        else:
            action = model.predict(venv.normalize_obs(obs[None]), deterministic=True)[0][0]
            on_shield = on_hold = on_search = False
            if rules == "shield+hold+search":
                action, on_search = search(obs, action)
            if rules in ("shield", "shield+hold", "shield+hold+search"):
                action, on_shield = shield(obs, action)
            if rules in ("shield+hold", "shield+hold+search"):
                action, on_hold = hold(obs, action)
        obs, reward, *_ = env.step(action)
        ret += reward

        d, b, a = env._true_state()
        visible = env._person_visible(d)
        # a measurement window that just closed yields one reading, worth the
        # quality of the pose it was taken from
        measuring = getattr(env, "measuring_for", 0.0) > 0.0
        if was_measuring and not measuring:
            qv = (g.predicted_band_snr(max(d, 1e-3), np.degrees(b), np.degrees(a))
                  if visible else g.CARDIAC_NOISE_FLOOR)
            readings.append(round(qv, 3))
            if first_good is None and qv >= GOOD_READING:
                first_good = t
        was_measuring = measuring
        shield_steps += on_shield
        hold_steps += on_hold
        xy.append((env.robot_x, env.robot_y, env.robot_yaw))
        occluded_steps += not visible
        for o in env.obstacles:
            gap = env._surface_distance(o, env.robot_x, env.robot_y)[0] - ROBOT_RADIUS
            clearance = min(clearance, gap)
            contact_steps += gap < 0
        if keep_path:
            path.append((round(env.robot_x, 4), round(env.robot_y, 4), round(env.robot_yaw, 4)))
        if t >= steps - SETTLE_WINDOW:
            settle.append((env.robot_x, env.robot_y, env.robot_yaw))
        if disturb and DISTURB_AT - 5 <= t <= DISTURB_AT + DISTURB_WINDOW:
            disturb_win.append((env.robot_x, env.robot_y, env.robot_yaw))
            if (first_move is None and pose_at_disturb is not None
                    and np.hypot(env.robot_x - pose_at_disturb[0],
                                 env.robot_y - pose_at_disturb[1]) > 0.02):
                first_move = (t - DISTURB_AT) * env.dt

    d, b, a = env._true_state()
    visible = env._person_visible(d)

    # Time to settle: the first step after which the robot never again moves
    # more than SETTLE_TOL from where it finished, in position OR heading.
    # None if it never settles.
    P = np.array(xy)
    yaw = np.unwrap(P[:, 2])
    far = ((np.hypot(P[:, 0] - P[-1, 0], P[:, 1] - P[-1, 1]) > SETTLE_TOL)
           | (np.abs(np.degrees(yaw - yaw[-1])) > SETTLE_TOL_DEG))
    settled_at = (int(np.max(np.flatnonzero(far))) + 1) if far.any() else 0
    if settled_at >= len(P) - MIN_SETTLED_STEPS:
        settled_at = None                      # never stayed put long enough

    settle = np.array(settle)
    rec = {
        "seed": seed,
        "return": round(ret, 2),
        "distance": round(d, 4),
        "bearing_deg": round(abs(np.degrees(b)), 2),
        "aspect_deg": round(abs(np.degrees(a)), 2),
        # quality of the final pose; zero line of sight means no reading at all
        "quality": round(g.predicted_band_snr(max(d, 1e-3), np.degrees(b), np.degrees(a))
                         if visible else g.CARDIAC_NOISE_FLOOR, 3),
        "visible_at_end": bool(visible),
        "in_fov": bool(abs(np.degrees(b)) < g.FOV_GATE_DEG),
        "holding_at_end": bool(hold_steps and on_hold),
        "shield_frac": round(shield_steps / steps, 4),
        "hold_frac": round(hold_steps / steps, 4),
        "occluded_frac": round(occluded_steps / steps, 4),
        "contact_steps": int(contact_steps),
        "min_clearance": round(float(clearance), 4) if np.isfinite(clearance) else None,
        "n_obstacles": len(env.obstacles),
        "drift_cm": round(float(np.hypot(np.ptp(settle[:, 0]), np.ptp(settle[:, 1])) * 100), 2),
        "heading_swing_deg": round(float(np.degrees(np.ptp(np.unwrap(settle[:, 2])))), 2),
        "person_phi_deg": round(float(np.degrees(env.person_phi)), 2),
        "reward_per_step": round(ret / steps, 3),
        "steps": steps,
        "start_quality": round(start_quality, 3),
        "readings": readings,
        "best_reading": (max(readings) if readings else None),
        "seconds_to_good_reading": (round(first_good * env.dt, 1)
                                    if first_good is not None else None),
        "quality_gain": round((g.predicted_band_snr(max(d, 1e-3), np.degrees(b), np.degrees(a))
                               if visible else g.CARDIAC_NOISE_FLOOR) - start_quality, 3),
        # Success = ended chest-facing AND stationary, so a reading can
        # actually be taken. Defined on the robot's motion rather than on the
        # hold being engaged, so every condition is scored the same way.
        "success": bool(abs(np.degrees(a)) < 45 and settled_at is not None),
        "steps_to_settle": settled_at,
        "seconds_to_settle": (round(settled_at * env.dt, 1) if settled_at is not None else None),
    }
    if disturb:
        w = np.array(disturb_win)
        rec["disturb_drift_cm"] = round(float(np.hypot(np.ptp(w[:, 0]), np.ptp(w[:, 1])) * 100), 2)
        rec["disturb_swing_deg"] = round(float(np.degrees(np.ptp(np.unwrap(w[:, 2])))), 2)
        rec["seconds_to_move"] = first_move
    if keep_path:
        rec["path"] = path
        rec["person"] = [round(env.person_x, 4), round(env.person_y, 4),
                         round(float(env.person_phi), 4)]
        rec["obstacles"] = [[o[0]] + [round(float(v), 4) for v in o[1:]]
                            for o in env.obstacles]
    return rec


def summarise(records):
    a = np.array([r["aspect_deg"] for r in records])
    q = np.array([r["quality"] for r in records])
    n = len(records)
    settle = [r["seconds_to_settle"] for r in records if r["seconds_to_settle"] is not None]
    to_good = [r["seconds_to_good_reading"] for r in records
               if r.get("seconds_to_good_reading") is not None]
    out = {
        "n": n,
        "success": int(sum(r["success"] for r in records)),
        "median_seconds_to_settle": (round(float(np.median(settle)), 1) if settle else None),
        "p90_seconds_to_settle": (round(float(np.percentile(settle, 90)), 1) if settle else None),
        "got_good_reading": int(sum(r.get("best_reading") is not None
                                    and r["best_reading"] >= GOOD_READING for r in records)),
        "median_seconds_to_good_reading": (round(float(np.median(to_good)), 1) if to_good else None),
        "mean_readings_taken": round(float(np.mean([len(r.get("readings", [])) for r in records])), 1),
        "mean_start_quality": round(float(np.mean([r["start_quality"] for r in records])), 2),
        "mean_quality_gain": round(float(np.mean([r["quality_gain"] for r in records])), 2),
        "chest_facing": int(np.sum(a < 45)),
        "side_on": int(np.sum((a >= 75) & (a < 105))),
        "behind": int(np.sum(a > 135)),
        "mean_quality": round(float(q.mean()), 2),
        "median_distance": round(float(np.median([r["distance"] for r in records])), 2),
        "median_bearing": round(float(np.median([r["bearing_deg"] for r in records])), 1),
        "in_fov": int(sum(r["in_fov"] for r in records)),
        "holding_at_end": int(sum(r["holding_at_end"] for r in records)),
        "occluded_any": int(sum(r["occluded_frac"] > 0 for r in records)),
        "occluded_at_end": int(sum(not r["visible_at_end"] for r in records)),
        "contact_episodes": int(sum(r["contact_steps"] > 0 for r in records)),
        "mean_obstacles": round(float(np.mean([r["n_obstacles"] for r in records])), 1),
        "median_drift_cm": round(float(np.median([r["drift_cm"] for r in records])), 2),
        "median_heading_swing_deg": round(float(np.median([r["heading_swing_deg"] for r in records])), 2),
        "fully_still": int(sum(r["heading_swing_deg"] < 0.1 and r["drift_cm"] < 0.1
                               for r in records)),
        "mean_reward_per_step": round(float(np.mean([r["reward_per_step"] for r in records])), 2),
        "shield_frac": round(float(np.mean([r["shield_frac"] for r in records])), 3),
    }
    if "disturb_drift_cm" in records[0]:
        moved = [r["seconds_to_move"] for r in records if r["seconds_to_move"] is not None]
        out.update({
            "disturb_median_drift_cm": round(float(np.median([r["disturb_drift_cm"] for r in records])), 2),
            "disturb_median_swing_deg": round(float(np.median([r["disturb_swing_deg"] for r in records])), 2),
            "disturb_unmoved": int(sum(r["disturb_drift_cm"] < 1.0 for r in records)),
            "disturb_responded": len(moved),
            "disturb_median_seconds_to_move": (round(float(np.median(moved)), 1) if moved else None),
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="empty", choices=["empty", "pillars", "room", "bedroom"])
    p.add_argument("--rules", default="shield+hold", choices=["none", "shield", "shield+hold", "shield+hold+search", "oracle"])
    p.add_argument("--model", default=None)
    p.add_argument("--episodes", type=int, default=N_EPISODES)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--tight", action="store_true", help="start wedged in the bedroom alcove")
    p.add_argument("--disturb", default=None, choices=["brief", "persist"],
                   help="person turns at t=20s: 70deg for 1s, or 90deg and stays")
    p.add_argument("--paths", type=int, default=0, help="store trajectories for the first N episodes")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    model_name = args.model or MODELS[args.layout]
    model, venv = load_policy(model_name)
    steps = args.steps if not (args.tight or args.disturb) else max(args.steps, 300 if args.tight else 400)

    records = [episode(model, venv, args.layout, args.rules, SEED0 + i, steps,
                       tight=args.tight, disturb=args.disturb, keep_path=i < args.paths)
               for i in range(args.episodes)]
    out = {
        "config": {"layout": args.layout, "rules": args.rules, "model": model_name,
                   "episodes": args.episodes, "steps": steps, "tight": args.tight,
                   "disturb": args.disturb, "seed0": SEED0},
        "summary": summarise(records),
        "episodes": records,
    }
    name = args.name or "_".join(
        [args.layout, args.rules.replace("+", "-")]
        + (["tight"] if args.tight else []) + ([args.disturb] if args.disturb else []))
    os.makedirs("eval/results", exist_ok=True)
    with open(f"eval/results/{name}.json", "w") as f:
        json.dump(out, f, indent=1)

    s = out["summary"]
    print(f"  readings {s['mean_readings_taken']}/episode   good reading in "
          f"{s['got_good_reading']}/{s['n']} eps after {s['median_seconds_to_good_reading']}s")
    print(f"[{name}] model={model_name} n={s['n']}  SUCCESS {s['success']}/{s['n']}"
          f"  settle {s['median_seconds_to_settle']}s (p90 {s['p90_seconds_to_settle']}s)"
          f"  quality {s['mean_start_quality']} -> {s['mean_quality']} (+{s['mean_quality_gain']})")
    print(f"  chest-facing {s['chest_facing']}  side-on {s['side_on']}  behind {s['behind']}"
          f"  in-FOV {s['in_fov']}  holding {s['holding_at_end']}")
    print(f"  quality {s['mean_quality']}  distance {s['median_distance']}m"
          f"  bearing {s['median_bearing']}deg  reward/step {s['mean_reward_per_step']}")
    print(f"  obstacles/ep {s['mean_obstacles']}  occluded any {s['occluded_any']}"
          f"  at end {s['occluded_at_end']}  contacts {s['contact_episodes']}")
    print(f"  settle: drift {s['median_drift_cm']}cm  swing {s['median_heading_swing_deg']}deg"
          f"  fully still {s['fully_still']}")
    if "disturb_median_drift_cm" in s:
        print(f"  around the turn: drift {s['disturb_median_drift_cm']}cm"
              f"  swing {s['disturb_median_swing_deg']}deg  unmoved {s['disturb_unmoved']}"
              f"  responded {s['disturb_responded']} after {s['disturb_median_seconds_to_move']}s")


if __name__ == "__main__":
    main()
