import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # set before torch imports

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from radar_env import RadarPoseEnv

# Returns in this task span roughly -16,000 to +2,000 (per-step reward runs
# -162..+21 over a ~100-step effective horizon at gamma=0.99), while the
# signal that actually separates a good pose from a mediocre one is worth
# about 10-25. Asking the value network to fit targets four orders of
# magnitude larger than the differences that matter is why explained_variance
# sat at exactly 0 across every reward function tried, with value_loss ~7e4.
#
# VecNormalize rescales returns to unit variance. It is deliberately the
# neutral fix: it changes nothing about what the reward MEANS or the relative
# weight of any term, only the scale the optimizer sees.
env = DummyVecEnv([lambda: Monitor(RadarPoseEnv())])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb/")
# Back to 500k (2026-09-21). With rate-limited actuators the learning curve
# plateaus by ~300k: the 2M run on the cardiac reward sat flat (+-10%) from
# 300k to 2M. The extra 1.5M steps were just noise, at 4x the wall time.
model.learn(total_timesteps=500_000, progress_bar=True)
model.save("ppo_radar")

# The running mean/var must be reloaded at eval time or the policy sees
# differently-scaled inputs than it trained on.
env.save("vec_normalize.pkl")
