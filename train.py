import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # set before torch imports

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from radar_env import RadarPoseEnv

env = Monitor(RadarPoseEnv())
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb/")
model.learn(total_timesteps=500_000, progress_bar=True)
model.save("ppo_radar")