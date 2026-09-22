import time

import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO

from radar_env import RadarPoseEnv
from geometry import PERSON_RADIUS


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


def main():
    env = RadarPoseEnv()
    model = PPO.load("ppo_radar")
    obs, info = env.reset()

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)

    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("r2d2.urdf", [env.robot_x, env.robot_y, 0.3])
    make_person_marker(env.person_x, env.person_y)
    make_front_marker(env.person_x, env.person_y, env.person_phi)

    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print("STEP:", action, reward, terminated, truncated)
        p.resetBasePositionAndOrientation(
            robot_id,
            [env.robot_x, env.robot_y, 0.3],
            yaw_to_quaternion(env.robot_yaw),
        )
        p.stepSimulation()
        time.sleep(env.dt)

    print(f"ended: terminated={terminated} truncated={truncated} "
          f"final distance={obs[0]:.2f} bearing={obs[1]:.2f} reward={reward:.2f} "
          f"person_phi={np.degrees(env.person_phi):.0f}deg")
    time.sleep(2)
    p.disconnect()


if __name__ == "__main__":
    main()
