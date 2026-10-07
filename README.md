# self-repositioning-vitals

A reinforcement-learning system that teaches a robot where to stand so a contactless
mmWave radar can get a good heart-rate reading. Part of BURE research at Cornell Tech,
advised by Prof. Rajalakshmi Nandakumar.

## The problem

Radar-based vital-sign sensing only works within a fairly narrow field of view and body
orientation relative to the sensor. Outside it, signal quality collapses — and side-on it
collapses for a reason no amount of transmit power fixes: the radar sees only the
component of chest displacement along the line of sight, so the heartbeat stops modulating
the return at all. A robot carrying the radar has to actively reposition to a pose that
will produce a usable reading, rather than sensing from wherever it happens to be parked.

Two things make that harder than ordinary navigation:

- **Signal quality cannot be computed from geometry alone.** It is a property of the
  returned signal, so it has to be *measured* — and measuring it requires the robot to
  stop, because platform vibration displaces the antenna by about as much as the chest
  motion being measured.
- **The robot cannot see which way the person faces until it is close.** A 2D lidar
  locates someone across a room, but resolving body orientation needs shape and motion, so
  the robot must commit to an approach before it knows which side the chest is on.

This repo trains a policy (PPO, via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/))
in simulation against a reward built from real radar signal-quality measurements rather
than a hand-guessed heuristic.

The signal-processing side — the pipeline that extracts a heart rate from raw radar data,
and the captures the reward model is anchored to — lives in a separate repo,
[`mmResp`](https://github.com/adnan-armouti/mmResp).

## Results

Measured over 100 episodes per condition with fixed seeds, so every condition sees
identical spawns, body orientations and sensor noise, with the policy acting
deterministically. An episode **succeeds** only if the robot ends both chest-facing
(|φ| < 45°) *and* stationary, since a reading needs both. Reproduce with
`python eval/run_eval.py`; per-episode records land in `eval/results/*.json` and every
number below is read back from those files.

Unfurnished room, one policy trained on `layout="empty"`:

| | Policy | + shield | + hold | Oracle |
|---|---|---|---|---|
| Success | 0 | 0 | **95** | 100 |
| Chest-facing | 25 | 69 | **100** | 100 |
| Settled at all | 0 | 0 | **95** | 100 |
| Settle time, median (s) | — | — | **7.5** | 7.6 |
| Predicted quality, final | 8.27 | 9.18 | **11.83** | 13.11 |

Quality is the reward model's predicted `cardiac_band_snr`, whose noise floor is 5.3 and
best achievable value 13.3. The oracle gets ground-truth pose and a hard-coded target; it
bounds performance rather than baselining it.

**The shield alone is not enough, and neither is the policy.** PPO learns to approach and
aim, but it parks at 0.32 m — hard against the standoff — on whichever side it arrives,
and never comes to rest. The shield more than doubles chest-facing (25 → 69) but cannot
finish the job on its own, because it only engages within 1.2 m and never stops the robot.
Only the hold brings it to a halt, and the two together take chest-facing to 100 and
success to 95.

Across three training seeds the full stack gives 90, 93 and 89 successes in 100, so the
result is reproducible rather than a fortunate run.

In cluttered layouts (policy, shield, hold and re-acquire): 76/100 with cylinders, 80/100
in randomised furnished rooms, and **67/100 in a fixed bedroom that is never trained on**,
with zero furniture contacts. Started wedged in an alcove with centimetres of clearance on
three sides, it extracts itself and succeeds in 94/100.

### Why there are rules at all

The orbit to the chest side is unreachable by policy gradient under this reward:
circumnavigating scores higher than remaining side-on, but the manoeuvre *begins by turning
away*, so every intermediate step scores worse than standing still and the better optimum
is never sampled. Reward shaping, a curriculum of side-on starts, longer training and
longer-range aspect sensing all left it unlearned. Folding the rules back into the learning
problem does produce the behaviour, but unreliably — across three seeds the learned variant
spans 19 to 81 successes where the rule-based stack spans 89 to 93.

## Files

| File | What it does |
|---|---|
| `geometry.py` | The reward model. Predicts signal quality from pose using a radar link budget (two-way R⁻⁴), a logistic saturation curve fitted to two real measured anchors, and a cos²φ body-aspect term with a small measured front/back asymmetry. Plus soft penalties for the collision zone and the field-of-view gate. |
| `radar_env.py` | A [Gymnasium](https://gymnasium.farama.org/) environment wrapping that reward. Defines the observation (lidar range/bearing/body-orientation plus validity flags, radar quality, commanded velocities, 8 lidar sector ranges), the action (drive speed, turn rate under acceleration limits), obstacles and occlusion, the stop-to-measure gate, and subject movement. |
| `shield.py` | The three rules that mediate between the policy and the wheels. `ChestShield` routes around the person to the chest side; `Hold` turns in place, stops, and judges a 2 s reading; `Reacquire` dead-reckons the person's last known position when line of sight is lost. All read only the observation, so they transfer to hardware unchanged. |
| `train.py` | Trains a PPO policy for 1,000,000 timesteps against one layout (`--layout empty\|pillars\|room`), with observation and return normalisation. Saves `ppo_radar_<layout>.zip` and its `_vecnorm.pkl`. |
| `eval/run_eval.py` | The evaluation protocol behind the numbers above — fixed seeds, deterministic policy, selectable rule stack (`--rules none\|shield\|shield+hold\|shield+hold+search\|oracle`), optional subject-disturbance tests. Writes JSON to `eval/results/`. |
| `eval/oracle.py` | The scripted ground-truth controller used as the upper bound. |
| `eval/feasibility.py` | Checks whether a layout admits any reachable, unoccluded, chest-facing pose. 18% of generated rooms did not; filtering them raised success from 52 to 80. |
| `figures/make_figures.py` | Regenerates every paper figure from the saved JSON. |
| `visualize_radar_env.py` | Loads a trained policy and plays one episode back through PyBullet, so you can watch it drive. |
| `test_pybullet.py` | A standalone PyBullet smoke test, used early to confirm PyBullet worked before wiring up `visualize_radar_env.py`. |

Trained policies (`*.zip`, `*_vecnorm.pkl`) are gitignored — regenerate them with
`train.py`.

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
python train.py --layout empty      # unfurnished, isolates positioning
python train.py --layout pillars    # open space with 1-5 cylinders
python train.py --layout room       # randomised furnished rooms
```

One policy per layout, 1,000,000 timesteps each (about seven minutes on a laptop), logging
to `tb/` — view with `tensorboard --logdir tb`. The fixed bedroom is never trained on; it
is evaluated with the policy trained on randomised rooms.

Normalising returns and observations is essential, not optional: episode returns span a far
wider range than the differences that distinguish a good pose from a mediocre one, and with
raw returns explained variance stayed at zero across every reward function tried.

## Evaluating

```bash
python eval/run_eval.py --layout empty   --rules none
python eval/run_eval.py --layout empty   --rules shield+hold
python eval/run_eval.py --layout empty   --rules oracle
python eval/run_eval.py --layout bedroom --rules shield+hold+search    # held out
python eval/run_eval.py --layout bedroom --rules shield+hold --tight   # wedged start
python eval/run_eval.py --layout empty   --rules shield+hold --disturb brief
```

Add `--paths N` to store trajectories for the first N episodes, which
`figures/make_figures.py` needs.

## Watching a trained policy

```bash
python visualize_radar_env.py
```

Opens a PyBullet window and plays back one episode of the saved policy driving toward a
good pose, given a randomly-oriented person.

## Current scope / known limitations

- **Simulation only.** The reward model is anchored to real radar captures, but the policy
  is trained and evaluated entirely in simulation. It has not been deployed on hardware.
- **The simulator supplies quality from the reward model rather than simulated radar
  returns.** That is the principal sim-to-real gap.
- **Chest versus back rests on a thin margin.** cos²φ is symmetric — chest-on and back-on
  both present a surface moving along the line of sight — and the measured front/back
  difference is only 0.4 dB. The strong preference the model expresses is for the coronal
  plane over side-on; the choice between front and back is closer to a system convention
  than a measured effect.
- **The chassis is assumed to settle instantly once the wheels stop.** On hardware this is
  a short additional wait; measuring it is a straightforward experiment on a real rig.
- **Re-acquire is a local heuristic with no map.** It helps where sight lines are actually
  lost (the bedroom, 63 → 67) and is neutral or slightly negative elsewhere.
- One person, assumed present and localisable. Distinguishing occlusion from absence, and
  selecting among several people, are out of scope. The person may change orientation but
  does not walk.
