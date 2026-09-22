import os
import time

import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from radar_env import RadarPoseEnv
from geometry import PERSON_RADIUS

VEC_NORMALIZE_PATH = "vec_normalize.pkl"


def yaw_to_quaternion(yaw):
    return p.getQuaternionFromEuler([0, 0, yaw])


def make_person_marker(x, y):
    visual = p.createVisualShape(
        p.GEOM_CYLINDER, radius=PERSON_RADIUS, length=1.6, rgbaColor=[0.9, 0.1, 0.1, 1.0]
    )
    return p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=visual, basePosition=[x, y, 0.8]
    )


def make_front_marker(x, y, phi):
    # small green sphere in the direction the person's chest faces (phi) --
    # for our eyes only, this is hidden from the agent during training
    offset = PERSON_RADIUS + 0.15
    fx = x + offset * np.cos(phi)
    fy = y + offset * np.sin(phi)
    visual = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.08, rgbaColor=[0.1, 0.9, 0.1, 1.0]
    )
    return p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=visual, basePosition=[fx, fy, 0.8]
    )


def load_obs_normalizer():
    """Training runs under VecNormalize, so the policy learned on scaled
    observations. Replaying without the same statistics feeds it inputs on a
    different scale and the behaviour degrades silently -- no error, just a
    worse-looking robot. Load them, or say clearly that we couldn't."""
    if not os.path.exists(VEC_NORMALIZE_PATH):
        print(f"WARNING: {VEC_NORMALIZE_PATH} not found -- replaying on raw "
              f"observations. Behaviour will not match training. Re-run "
              f"train.py to regenerate it.")
        return None
    norm = VecNormalize.load(VEC_NORMALIZE_PATH,
                             DummyVecEnv([lambda: RadarPoseEnv()]))
    norm.training = False
    norm.norm_reward = False
    return norm


def main():
    env = RadarPoseEnv()
    model = PPO.load("ppo_radar")
    normalizer = load_obs_normalizer()
    obs, info = env.reset()

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)

    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("r2d2.urdf", [env.robot_x, env.robot_y, 0.3])
    make_person_marker(env.person_x, env.person_y)
    make_front_marker(env.person_x, env.person_y, env.person_phi)

    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        policy_obs = normalizer.normalize_obs(obs) if normalizer else obs
        action, _ = model.predict(policy_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if step % 10 == 0:
            d, b, a = env._true_state()
            seen = f"{obs[4]:.0f}{obs[5]:.0f}{obs[6]:.0f}"
            print(f"step {step:3d}  d={d:.2f}  bearing={np.degrees(b):6.1f}  "
                  f"phi={np.degrees(a):7.1f}  reward={reward:7.2f}  "
                  f"channels={seen}")
        step += 1
        p.resetBasePositionAndOrientation(
            robot_id,
            [env.robot_x, env.robot_y, 0.3],
            yaw_to_quaternion(env.robot_yaw),
        )
        p.stepSimulation()
        time.sleep(env.dt)

    # report TRUE pose, not the observed one -- observed channels read 0
    # when the sensor had nothing to report, which is misleading here
    d, b, a = env._true_state()
    print(f"ended: terminated={terminated} truncated={truncated} "
          f"final distance={d:.2f} bearing={np.degrees(b):.0f}deg "
          f"phi={np.degrees(a):.0f}deg reward={reward:.2f} "
          f"channels(range/phi/snr)={obs[4]:.0f}/{obs[5]:.0f}/{obs[6]:.0f}")
    time.sleep(2)
    p.disconnect()


if __name__ == "__main__":
    main()
