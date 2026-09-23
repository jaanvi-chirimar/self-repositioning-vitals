import numpy as np
from gymnasium import Env, spaces
from geometry import (distance_and_angle, aspect_angle, reward as compute_reward,
                      predicted_band_snr, link_budget_db, MIN_CENTER_DISTANCE,
                      ROBOT_RADIUS, PERSON_RADIUS, CARDIAC_NOISE_FLOOR)

# --- sensor model ---------------------------------------------------------
# The robot does NOT read simulator state. It reads two sensors, each of
# which can independently fail to return anything, which is the whole point:
# "what if gives null value for all three from person -> move around until
# see person randomly" only means something if the person CAN go unseen.
#
# Split follows the hardware: geometry from the LiDAR, signal from the radar.
#
# Note the asymmetry between range/bearing and phi. Getting a person's
# POSITION off a 2D LiDAR is ordinary clustering. Getting their body
# ORIENTATION is a much harder shape+motion estimation problem (per the
# project notes: shape models plus a motion-derived heading, with front/back
# ambiguity as a known failure mode), and it needs enough angular resolution
# on the target to work at all -- so it drops out at a much shorter range
# and is far noisier when it does resolve.
LIDAR_MAX_RANGE = 4.0        # person detectable (position) out to here (m)
LIDAR_PHI_MAX_RANGE = 2.0    # body ORIENTATION only resolvable this close (m)
LIDAR_RANGE_NOISE = 0.02     # m, 1 sigma
LIDAR_BEARING_NOISE_DEG = 1.0
LIDAR_PHI_NOISE_DEG = 10.0   # phi estimation is the unreliable one

# The radar reports cardiac_band_snr (mmResp/cardiac_quality.py) -- pre-CNN,
# computed straight from the phase at the peak range bin, so it is something
# the real robot can actually read live. It is only reported when the radar
# has a target at all: too far, too far off-boresight, or too side-on and the
# link budget drops below RADAR_SNR_FLOOR_DB and there is no peak bin to read.
#
# Noise is the measured spread: repeat captures of the same pose vary by
# std ~3-6 on a median of ~10-13 (table_a_cardiac.csv), i.e. ~2 dB.
RADAR_SNR_FLOOR_DB = -20.0
RADAR_SNR_NOISE_DB = 2.0
SNR_OBS_SCALE = 20.0         # keeps the dB value in roughly [0, 1] for the net

# --- actuator limits ------------------------------------------------------
# "Not to have it jerk around", as a CONSTRAINT rather than a preference.
#
# A reward penalty on step-to-step action change was tried first and failed
# badly (mean reward 16.8 -> 5.8, 18/40 episodes diverged, and jitter went UP).
# The reason is structural, not a bad weight: PPO explores by SAMPLING actions
# from a distribution with std ~1, so consecutive samples differ by ~1.4
# regardless of what the policy's mean is doing. The penalty therefore taxes
# exploration noise rather than behaviour, and the cheapest way to avoid it is
# to hold one constant action and go nowhere -- which is exactly what it
# learned.
#
# Rate-limiting the actuators instead makes jerking physically impossible,
# leaves exploration untouched, and needs no weight balanced against the
# reward. It is also just more honest: the old env let velocity jump from
# -0.5 to +0.5 within one 100ms tick, which no real differential drive can do.
MAX_LINEAR_ACCEL = 1.0    # m/s^2
MAX_ANGULAR_ACCEL = 3.0   # rad/s^2

# Robot spawn distance from the person, drawn fresh each episode (see
# reset()). SPAWN_R_MIN is kept above MIN_CENTER_DISTANCE (the collision
# floor, ~0.62m) so every spawn is automatically clear of the collision
# zone -- no resampling/retry loop needed, it's guaranteed by construction.
SPAWN_R_MIN = 1.0
SPAWN_R_MAX = 3.0
assert SPAWN_R_MIN > MIN_CENTER_DISTANCE

# --- obstacles -------------------------------------------------------------
# Upright circular obstacles (chair legs, table bases, a person standing by).
# They do three things, and the third is the point:
#   1. the robot can drive into them (penalty, same shape as the person's),
#   2. they block the LiDAR rays the robot navigates by,
#   3. they OCCLUDE the person -- both the LiDAR and the radar are
#      line-of-sight, so behind an obstacle the person is simply not there:
#      no range, no phi, no SNR, every validity flag 0. That is the case the
#      project notes asked for ("null value for all three from person ->
#      move around until see person"), and until now it could not happen.
N_OBSTACLES = (2, 5)         # per episode, drawn uniformly (inclusive)
OBSTACLE_RADIUS = (0.15, 0.35)
OBSTACLE_AREA = 3.0          # m, obstacles land within this of the person
OBSTACLE_CLEAR_PERSON = 0.7  # keep this clear of the person, so a pose in
                             # front of them is never impossible to reach
OBSTACLE_CLEAR_ROBOT = 0.5   # and this clear of wherever the robot spawns
OBSTACLE_PENALTY_SCALE = 200.0 * 0.316   # same per-metre weight as the person's

# The navigation LiDAR the robot avoids obstacles with: rays around the
# robot, each reporting range to the nearest thing it hits. Coarse on
# purpose -- a real 2D LiDAR gives hundreds of rays, but the policy only
# needs enough to tell "blocked left / blocked ahead / clear right", and 8
# keeps the observation small enough to stay trainable.
N_LIDAR_RAYS = 8
RAYS_PER_SECTOR = 3          # a real LiDAR is dense; each reported sector is
                             # the MINIMUM over this many rays, so a thin
                             # obstacle between two ray angles is not missed


class RadarPoseEnv(Env):
    def __init__(self, dt=0.1, max_steps=200):
        self.action_space = spaces.Box(
            low=np.array([-0.5, -1.5], dtype=np.float32),
            high=np.array([0.5, 1.5], dtype=np.float32),
        )
        # [range, bearing, phi, snr, range_ok, phi_ok, snr_ok, cmd_v, cmd_omega]
        #
        # The two velocity terms are there because the drive is rate-limited:
        # a command no longer takes effect instantly, so the robot's current
        # velocity is part of the state. Without it in the observation the
        # policy can't tell what its next command will actually do, and the
        # problem stops being Markovian.
        #
        # The last three are validity flags, and they are the reason this
        # env is worth anything: a value of 0.0 in the phi slot means
        # "facing straight at me" when phi_ok=1, and "I have no idea" when
        # phi_ok=0. Without the flag those two are indistinguishable and the
        # policy would treat an absent reading as a confident one.
        #
        # Reward is still computed from TRUE state (see step()) -- sensing
        # quality is real whether or not the robot perceives it. Only the
        # OBSERVATION is degraded. Standard POMDP arrangement.
        # ...followed by N_LIDAR_RAYS obstacle ranges, normalised to
        # [0, 1] by LIDAR_MAX_RANGE, starting straight ahead and going
        # counter-clockwise.
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.pi, -np.pi, -5.0, 0.0, 0.0, 0.0, -0.5, -1.5]
                         + [0.0] * N_LIDAR_RAYS, dtype=np.float32),
            high=np.array([LIDAR_MAX_RANGE, np.pi, np.pi, 5.0, 1.0, 1.0, 1.0, 0.5, 1.5]
                          + [1.0] * N_LIDAR_RAYS, dtype=np.float32),
        )
        self.dt = dt
        self.max_steps = max_steps

    def _ray_hit(self, angle, max_range=LIDAR_MAX_RANGE):
        """Range from the robot to the first obstacle along a world-frame
        ray, or max_range if it hits nothing."""
        dx, dy = np.cos(angle), np.sin(angle)
        best = max_range
        for ox, oy, r in self.obstacles:
            # closest approach of the ray to the obstacle centre
            fx, fy = ox - self.robot_x, oy - self.robot_y
            t = fx * dx + fy * dy
            if t <= 0:
                continue
            perp2 = (fx - t * dx) ** 2 + (fy - t * dy) ** 2
            if perp2 >= r * r:
                continue
            hit = t - np.sqrt(r * r - perp2)
            if 0 < hit < best:
                best = hit
        return best

    def _person_visible(self, d):
        """Line of sight to the person. Both sensors are line-of-sight, so
        an obstacle in the way hides them from BOTH."""
        angle = np.arctan2(self.person_y - self.robot_y, self.person_x - self.robot_x)
        return self._ray_hit(angle, max_range=d + 1e-6) >= d - 1e-6

    def _resolve_contacts(self):
        """Obstacles and the person are SOLID. If the step drove the robot
        into one, push it back out along the surface normal -- so it stops
        against the obstacle, or slides along it, but never passes through.
        The reward's contact penalty still applies (it charges for crossing
        the safety margin, which is a larger radius than the body itself).

        Two passes, so being squeezed between two obstacles still resolves
        instead of oscillating."""
        bodies = [(ox, oy, r + ROBOT_RADIUS) for ox, oy, r in self.obstacles]
        bodies.append((self.person_x, self.person_y,
                       PERSON_RADIUS + ROBOT_RADIUS))
        for _ in range(2):
            for bx, by, keep_out in bodies:
                dx, dy = self.robot_x - bx, self.robot_y - by
                dist = np.hypot(dx, dy)
                if dist >= keep_out:
                    continue
                if dist < 1e-6:            # exactly on top of it: pick a direction
                    dx, dy, dist = 1.0, 0.0, 1.0
                self.robot_x = bx + dx / dist * keep_out
                self.robot_y = by + dy / dist * keep_out
                self.cmd_v = 0.0           # contact stops the drive

    def _obstacle_penalty(self):
        """How far the robot has pushed into an obstacle's safe zone."""
        pen = 0.0
        for ox, oy, r in self.obstacles:
            gap = np.hypot(ox - self.robot_x, oy - self.robot_y) - (r + ROBOT_RADIUS + 0.1)
            pen += OBSTACLE_PENALTY_SCALE * max(0.0, -gap)
        return pen

    def _true_state(self):
        """Ground truth (distance, bearing, aspect) -- for the REWARD only.
        The policy never sees this; it gets _obs() instead."""
        d, b = distance_and_angle(
            self.robot_x, self.robot_y, self.robot_yaw,
            self.person_x, self.person_y,
        )
        a = aspect_angle(
            self.robot_x, self.robot_y,
            self.person_x, self.person_y, self.person_phi,
        )
        return d, b, a

    def _obs(self):
        """What the sensors actually report this step. Each channel can come
        back empty; the flags say which did."""
        d, b, a = self._true_state()

        visible = self._person_visible(d)
        range_ok = d <= LIDAR_MAX_RANGE and visible
        if range_ok:
            d_obs = d + self.np_random.normal(0.0, LIDAR_RANGE_NOISE)
            b_obs = b + self.np_random.normal(
                0.0, np.radians(LIDAR_BEARING_NOISE_DEG))
        else:
            d_obs, b_obs = 0.0, 0.0

        # body orientation needs the target resolved well enough to fit a
        # shape to -- available over a shorter range than mere detection
        phi_ok = range_ok and d <= LIDAR_PHI_MAX_RANGE
        a_obs = (a + self.np_random.normal(0.0, np.radians(LIDAR_PHI_NOISE_DEG))
                 if phi_ok else 0.0)

        # cardiac_band_snr in dB. Not modelled yet: the real reading needs a
        # 10s+ window, so on hardware it lags the pose by several seconds.
        snr_ok = visible and link_budget_db(max(d, 1e-3), np.degrees(b)) >= RADAR_SNR_FLOOR_DB
        true_snr = 10.0 * np.log10(predicted_band_snr(max(d, 1e-3), np.degrees(b), np.degrees(a)))
        snr_obs = (true_snr + self.np_random.normal(0.0, RADAR_SNR_NOISE_DB)
                   if snr_ok else 0.0)

        return np.array([
            np.clip(d_obs, 0.0, LIDAR_MAX_RANGE),
            np.arctan2(np.sin(b_obs), np.cos(b_obs)),
            np.arctan2(np.sin(a_obs), np.cos(a_obs)),
            np.clip(snr_obs / SNR_OBS_SCALE, -5.0, 5.0),
            float(range_ok), float(phi_ok), float(snr_ok),
            self.cmd_v, self.cmd_omega,
            *self._lidar_scan(),
        ], dtype=np.float32)

    def _lidar_scan(self):
        """Obstacle ranges in sectors around the robot, normalised to
        [0, 1]. Each sector reports its closest hit."""
        n = N_LIDAR_RAYS * RAYS_PER_SECTOR
        rays = [self._ray_hit(self.robot_yaw + 2 * np.pi * i / n) for i in range(n)]
        return [min(rays[i * RAYS_PER_SECTOR:(i + 1) * RAYS_PER_SECTOR]) / LIDAR_MAX_RANGE
                for i in range(N_LIDAR_RAYS)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.person_x, self.person_y = 2.0, 0.0
        self.person_phi = self.np_random.uniform(-np.pi, np.pi)

        # Robot spawns at a random (r, theta) offset from the person --
        # random distance within [SPAWN_R_MIN, SPAWN_R_MAX], random angle
        # anywhere around them (not just in front), so the policy has to
        # navigate to a good pose rather than start near one.
        r = self.np_random.uniform(SPAWN_R_MIN, SPAWN_R_MAX)
        theta = self.np_random.uniform(-np.pi, np.pi)
        self.robot_x = self.person_x + r * np.cos(theta)
        self.robot_y = self.person_y + r * np.sin(theta)
        self.robot_yaw = self.np_random.uniform(-np.pi, np.pi)

        # obstacles last, so they can be kept clear of both bodies
        self.obstacles = []
        n = self.np_random.integers(N_OBSTACLES[0], N_OBSTACLES[1] + 1)
        for _ in range(int(n)):
            for _try in range(20):
                ang = self.np_random.uniform(-np.pi, np.pi)
                rad = self.np_random.uniform(0.8, OBSTACLE_AREA)
                ox = self.person_x + rad * np.cos(ang)
                oy = self.person_y + rad * np.sin(ang)
                orad = self.np_random.uniform(*OBSTACLE_RADIUS)
                if (np.hypot(ox - self.person_x, oy - self.person_y)
                        < OBSTACLE_CLEAR_PERSON + orad):
                    continue
                if (np.hypot(ox - self.robot_x, oy - self.robot_y)
                        < OBSTACLE_CLEAR_ROBOT + orad):
                    continue
                self.obstacles.append((ox, oy, orad))
                break

        self.step_count = 0
        self.cmd_v, self.cmd_omega = 0.0, 0.0   # starts at rest

        return self._obs(), {}

    def step(self, action):
        # the action is a velocity REQUEST; the drive ramps toward it within
        # its acceleration limits rather than snapping there instantly
        want_v, want_omega = float(action[0]), float(action[1])
        max_dv = MAX_LINEAR_ACCEL * self.dt
        max_dw = MAX_ANGULAR_ACCEL * self.dt
        self.cmd_v += np.clip(want_v - self.cmd_v, -max_dv, max_dv)
        self.cmd_omega += np.clip(want_omega - self.cmd_omega, -max_dw, max_dw)

        self.robot_x += self.cmd_v * np.cos(self.robot_yaw) * self.dt
        self.robot_y += self.cmd_v * np.sin(self.robot_yaw) * self.dt
        self._resolve_contacts()
        self.robot_yaw += self.cmd_omega * self.dt
        self.robot_yaw = np.arctan2(np.sin(self.robot_yaw), np.cos(self.robot_yaw))

        # reward from TRUE state, observation from the sensors -- the robot
        # is scored on where it actually is, not on what it managed to see
        d, b, a = self._true_state()
        r = compute_reward(d, b, a) - self._obstacle_penalty()
        # the radar is line-of-sight too: behind an obstacle there is no
        # reading to be had, so the pose is worth nothing no matter how good
        # its geometry looks. Reward follows the same rule the sensor does.
        if not self._person_visible(d):
            r -= predicted_band_snr(max(d, 1e-3), np.degrees(b), np.degrees(a)) - CARDIAC_NOISE_FLOOR
        obs = self._obs()

        self.step_count += 1
        # NOTHING terminates early -- every episode runs the full budget.
        #
        # FOV violation and collision are handled through the reward, for the
        # reason that ending an episode on them would let the agent dodge
        # future reward loss by never approaching at all.
        #
        # Driving away used to terminate at d > 5.0, which was the same
        # mistake: quitting early CAPS the accumulated penalty, so running
        # away was itself an escape hatch. It only became load-bearing once
        # the drive was rate-limited -- turning around from a 3m spawn traces
        # a wide arc that routinely crosses 5m mid-manoeuvre, so legitimate
        # turn-arounds were being killed off and the agent was being taught
        # that swinging wide ends the pain. 19/40 episodes were terminating
        # this way, with mean final distance 2.71m. Now there is no payoff:
        # sit out at range and you simply earn ~0 for 200 steps instead of ~24.
        terminated = False
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
