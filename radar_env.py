import numpy as np
from gymnasium import Env, spaces
from geometry import distance_and_angle, aspect_angle, reward as compute_reward


class RadarPoseEnv(Env):
    def __init__(self, dt=0.1, max_steps=200):
        self.action_space = spaces.Box(
            low=np.array([-0.5, -1.5], dtype=np.float32),
            high=np.array([0.5, 1.5], dtype=np.float32),
        )
        # observation is [distance, bearing, aspect]. ASSUMPTION, stated
        # explicitly, not a measured capability: phi (person_phi, see
        # reset()) is observable at deployment via some sensor (e.g. camera
        # pose estimation like BlazePose/MEBOW, or LiDAR shape+motion --
        # see drl_reward_design notes). That's untested on real hardware,
        # and has a plausible failure mode (a stationary subject gives no
        # motion history to resolve front/back ambiguity) that hasn't been
        # checked either. aspect is exposed to the agent on the strength of
        # this assumption, unlike the earlier POMDP version of this env
        # where phi was hidden. If this assumption doesn't hold, this
        # observation space doesn't either -- name it as an assumption in
        # any writeup, don't present it as settled.
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.pi, -np.pi], dtype=np.float32),
            high=np.array([5.0, np.pi, np.pi], dtype=np.float32),
        )
        self.dt = dt
        self.max_steps = max_steps

    def _obs(self):
        d, b = distance_and_angle(
            self.robot_x, self.robot_y, self.robot_yaw,
            self.person_x, self.person_y,
        )
        a = aspect_angle(
            self.robot_x, self.robot_y,
            self.person_x, self.person_y, self.person_phi,
        )
        return np.array([d, b, a], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_x, self.robot_y, self.robot_yaw = 0.0, 0.0, 0.0
        self.person_x, self.person_y = 2.0, 0.0
        self.person_phi = self.np_random.uniform(-np.pi, np.pi)
        self.step_count = 0

        return self._obs(), {}

    def step(self, action):
        v, omega = action

        self.robot_x += v * np.cos(self.robot_yaw) * self.dt
        self.robot_y += v * np.sin(self.robot_yaw) * self.dt
        self.robot_yaw += omega * self.dt
        self.robot_yaw = np.arctan2(np.sin(self.robot_yaw), np.cos(self.robot_yaw))

        obs = self._obs()
        d, b, a = obs
        r = compute_reward(d, b, a)

        self.step_count += 1
        # FOV violation and collision are handled entirely through the
        # reward (Q=0 / a penalty, see geometry.reward) -- not termination.
        # Ending the episode on either would let the agent avoid future
        # reward loss by never approaching in the first place, which is
        # exactly backwards from what we want it to learn.
        diverged = d > 5.0
        terminated = bool(diverged)
        truncated = self.step_count >= self.max_steps

        return obs, r, terminated, truncated, {}


if __name__ == "__main__":
    env = RadarPoseEnv()
    obs, info = env.reset()
    print("initial obs:", obs)

    # drive straight forward at max speed and watch distance shrink
    for i in range(50):
        obs, r, terminated, truncated, info = env.step(np.array([0.5, 0.0]))
        if i % 10 == 0 or terminated or truncated:
            print(f"step {i}: obs={obs} reward={r:.3f} terminated={terminated} truncated={truncated}")
        if terminated or truncated:
            break
