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
# 500k was enough while the drive responded instantly, but rate-limited
# actuators make this a second-order control problem -- commands take ~5 steps
# to take effect, so the policy has to anticipate rather than react. At 500k
# the learning curve was still climbing steeply (-11400 -> -698 and rising),
# i.e. cut off mid-improvement rather than converged.
model.learn(total_timesteps=2_000_000, progress_bar=True)
model.save("ppo_radar")

# The running mean/var must be reloaded at eval time or the policy sees
# differently-scaled inputs than it trained on.
env.save("vec_normalize.pkl")
