import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # set before torch imports

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from radar_env import RadarPoseEnv

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--layout", default="pillars", choices=["empty", "pillars", "room"],
                    help="training environment. 'room' draws a fresh furnished room each "
                         "episode; the fixed bedroom is never trained on and is held out "
                         "as the test environment.")
parser.add_argument("--steps", type=int, default=1_000_000)
parser.add_argument("--seed", type=int, default=None,
                    help="training seed; set it to measure run-to-run variance")
parser.add_argument("--out", default=None, help="model name (default: ppo_radar_<layout>)")
args = parser.parse_args()
layout = None if args.layout == "pillars" else args.layout
out = args.out or f"ppo_radar_{args.layout}"

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
env = DummyVecEnv([lambda: Monitor(RadarPoseEnv(layout=layout))])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb/", seed=args.seed)
# The learning curve plateaus by ~300k in the empty room; obstacle layouts
# are harder, so both conditions train for 1M steps to keep the reported
# setup uniform across the paper's tables.
model.learn(total_timesteps=args.steps, progress_bar=True)
model.save(out)

# The running mean/var must be reloaded at eval time or the policy sees
# differently-scaled inputs than it trained on.
env.save(f"{out}_vecnorm.pkl")
