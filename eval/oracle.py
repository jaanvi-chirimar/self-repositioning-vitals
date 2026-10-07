"""Scripted upper-bound controller.

Given the person's TRUE position and orientation and a hard-coded target
standoff, drive around them to a chest-on pose and stop facing them. This is
not a baseline -- it assumes exactly the information the robot has to infer,
and cannot exist on hardware. It is reported as a bound, and it is what
established that the reward ranks circumnavigation correctly: when this
controller outscored the trained policy under the same reward, the failure
was exploration, not reward design.
"""
import numpy as np

TARGET_RANGE = 0.7       # m, standoff it aims for
ORBIT_RANGE = 1.0        # m, radius it swings around the person at
STEP_DEG = 30.0          # granularity of the way round
SPEED = 0.5
HEADING_GAIN = 3.0
ARRIVED = 0.1            # m


def oracle_controller(env):
    """Returns a closure producing the next action for this episode."""
    person = np.array([env.person_x, env.person_y])
    facing = env.person_phi
    start = np.arctan2(env.robot_y - person[1], env.robot_x - person[0]) - facing
    start = np.arctan2(np.sin(start), np.cos(start))

    n = int(abs(start) // np.radians(STEP_DEG))
    angles = [start - np.sign(start) * np.radians(STEP_DEG) * (k + 1) for k in range(n)]
    waypoints = [person + ORBIT_RANGE * np.array([np.cos(facing + a), np.sin(facing + a)])
                 for a in angles]
    waypoints.append(person + TARGET_RANGE * np.array([np.cos(facing), np.sin(facing)]))

    def act():
        here = np.array([env.robot_x, env.robot_y])
        while waypoints and np.linalg.norm(waypoints[0] - here) < ARRIVED:
            waypoints.pop(0)
        target = waypoints[0] if waypoints else person
        heading = np.arctan2(*(target - here)[::-1])
        err = np.arctan2(np.sin(heading - env.robot_yaw), np.cos(heading - env.robot_yaw))
        v = (SPEED if abs(err) < 0.4 else 0.0) if waypoints else 0.0
        return np.array([v, np.clip(HEADING_GAIN * err, -1.5, 1.5)], dtype=np.float32)

    return act
