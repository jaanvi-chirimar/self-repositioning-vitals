# self-repositioning-vitals

A reinforcement-learning system that teaches a robot where to stand so a contactless
mmWave radar can get a good heart-rate reading. Part of BURE summer research at Cornell
Tech, advised by Prof. Rajalakshmi Nandakumar.

## The problem

Radar-based vital-sign sensing only works well within a fairly narrow field of view and
body orientation relative to the sensor — outside of it, signal quality drops off sharply.
A robot carrying the radar needs to actively reposition itself to a pose that will
actually produce a usable reading, rather than sensing from wherever it happens to be
parked.

This repo trains a policy (PPO, via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/))
to solve that repositioning problem in simulation, using a reward function built from real
radar signal-quality measurements rather than a hand-guessed heuristic.

The signal-processing / radar-capture side of the project (the pipeline that actually
extracts a heart rate from raw radar data, and the data used to build the reward tables
here) lives in a separate repo, [`mmResp`](https://github.com/adnan-armouti/mmResp).

## Files

| File | What it does |
|---|---|
| `geometry.py` | The reward function. Scores a pose as `Q = TableA(distance) × TableB(theta, phi)`, built from real captured `band_snr` signal-quality data, minus soft penalties for the collision zone, the sensor's field-of-view gate, and the body-orientation dead zone. |
| `radar_env.py` | A [Gymnasium](https://gymnasium.farama.org/) environment wrapping that reward — defines what the robot observes (distance, bearing, body-orientation angle), what actions it can take (drive speed, turn rate), and one episode's dynamics. |
| `train.py` | Trains a PPO policy against that environment for 500,000 timesteps, logging to TensorBoard (`tb/`, gitignored). Saves the result as `ppo_radar.zip`. |
| `visualize_radar_env.py` | Loads a trained policy and plays one episode back through PyBullet, so you can watch it drive to a pose. |
| `test_pybullet.py` | A standalone PyBullet smoke test (loads a plane + robot, no connection to the RL code) — used early on to confirm PyBullet was working before wiring up `visualize_radar_env.py`. |
| `ppo_radar.zip` | A trained model checkpoint. |
| `quality_vs_angle.png` | A generated plot of signal quality vs. angle. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or your preferred env manager
pip install -r requirements.txt
```

On macOS, PyTorch and NumPy can conflict over the OpenMP runtime. If you hit an
`OMP: Error #15` crash, run with:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python train.py
```

## Training

```bash
python train.py
```

Runs PPO for 500,000 timesteps (~2,500 episodes at 200 steps/episode), saves
`ppo_radar.zip`, and logs training curves to `tb/` — view with `tensorboard --logdir tb`.

## Watching a trained policy

```bash
python visualize_radar_env.py
```

Opens a PyBullet window and plays back one episode of the saved policy driving toward a
good pose, given a randomly-oriented person.

## Current scope / known limitations

- Only the person's body orientation (φ) is randomized between training episodes — the
  robot's starting position and the person's location are fixed. The policy has not been
  trained or tested on varying start distances/positions, which would be the natural next
  step before deploying more broadly.
- The reward's `(theta, phi)` lookup table is built from a sparse, non-uniform grid of real
  measurements (see `mmResp`) — nearest-neighbor lookup, not interpolation.
- This trains and evaluates entirely in simulation. The policy has not yet been deployed on
  real hardware.
