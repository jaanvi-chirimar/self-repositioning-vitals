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

# --- when a reading exists at all ----------------------------------------
# The radar can only report cardiac_band_snr while the platform is at rest.
# Robot vibration displaces the antenna by roughly as much as the chest
# motion being measured (Kobayashi et al. 2023 report robot vibration in the
# same amplitude range as the vital signs themselves, and restrict sensing to
# stationary operation), so a reading taken while driving measures the robot,
# not the person. The robot therefore stops, waits SETTLE_S for the chassis
# to come to rest, and integrates for SNR_WINDOW_S; if the reading is poor it
# moves on and tries elsewhere.
STILL_V = 0.01               # m/s below which the drive counts as stopped
STILL_W = 0.02               # rad/s
SETTLE_S = 0.0               # rest this long before a reading is trusted: the
                             # time for the chassis transient to decay once the
                             # wheels stop. The measurement window itself
                             # (shield.SNR_WINDOW_S) runs after this, so the
                             # robot stops for SETTLE_S + SNR_WINDOW_S in total.

# --- person movement ------------------------------------------------------
# Random body movement (RBM) is the dominant error source in radar vital
# sensing: the chest moves 1-12mm with respiration and only 0.2-0.5mm with
# the heartbeat, while a shift in posture moves the torso by centimetres.
# That energy lands in-band and raises the spectral median, which is the
# denominator of cardiac_band_snr, so the metric degrades as
#     Q_obs = floor + (Q - floor) / (1 + (A_rbm / A_cardiac)^2)
# i.e. motion of the same amplitude as the heartbeat halves the usable
# signal, and a centimetre-scale shift destroys it. A_rbm follows a slow
# random walk (postural sway) punctuated by occasional larger shifts.
A_CARDIAC_MM = 0.35          # heartbeat chest displacement, mid of 0.2-0.5mm
SWAY_MM = 0.15               # steady-state postural sway amplitude
SWAY_TAU_S = 3.0             # how fast sway wanders
SHIFT_PROB_PER_S = 0.05      # rate of larger postural shifts
SHIFT_MM = (2.0, 15.0)       # amplitude of one, uniformly drawn
SHIFT_DURATION_S = (0.5, 3.0)
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
# A furnished room, for demos and for testing transfer out of the random
# pillar layout the policy trains on. Walls plus a bed, a nightstand and a
# dresser, with the person sitting on the edge of the bed -- the bed is the
# interesting part, because it is long, low and usually lies between the
# robot and the person, so whole approach directions are simply unavailable.
# Randomized furnished rooms, available as layout="room". NOT used in
# training: the policy is trained on open space with random pillars only, so
# the furnished-room results are a zero-shot transfer test into a layout it
# has never seen, which is the situation a deployed robot is actually in.
ROOM_SIZE_RANGE = ((3.2, 5.0), (2.8, 4.2))
ROOM_FURNITURE = (2, 4)      # pieces per room, besides the walls

BEDROOM_SIZE = (4.0, 3.2)
BEDROOM_PERSON = (1.5, 1.75)     # sitting on the near edge of the bed
BEDROOM_FACING_SPREAD_DEG = 70.0  # person faces into the room, +-this

# A deliberately awkward start: the gap between the dresser and the two walls
# behind it, about 40cm wide for a 34cm-wide robot. The robot begins boxed in
# on three sides with the person out in the room, so it has to reverse out of
# the alcove before any of the sensing geometry is reachable at all. Used with
# layout="bedroom" as a stress case, not in training.
TIGHT_START = (3.62, 2.98)

N_LIDAR_RAYS = 8
RAYS_PER_SECTOR = 3          # a real LiDAR is dense; each reported sector is
                             # the MINIMUM over this many rays, so a thin
                             # obstacle between two ray angles is not missed


class RadarPoseEnv(Env):
    def __init__(self, dt=0.1, max_steps=200, layout=None, tight_start=False):
        self.layout = layout        # None = random pillars, "bedroom"/"room" = furnished
        self.tight_start = tight_start   # bedroom only: start wedged in the alcove
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

    # Obstacles are ("circle", x, y, r) or ("box", cx, cy, half_w, half_h, yaw).
    # Boxes are what a room is actually made of -- walls, a bed, a dresser --
    # and a bed in particular is nothing like a cylinder: it is long, low and
    # usually the thing between the robot and the person.

    @staticmethod
    def _to_box_frame(o, px, py):
        _, cx, cy, hw, hh, yaw = o
        c, sn = np.cos(-yaw), np.sin(-yaw)
        dx, dy = px - cx, py - cy
        return dx * c - dy * sn, dx * sn + dy * c, hw, hh

    def _surface_distance(self, o, px, py):
        """Distance from a point to the obstacle's surface (negative inside),
        with the outward unit normal at the closest point."""
        if o[0] == "circle":
            _, cx, cy, r = o
            dx, dy = px - cx, py - cy
            dist = np.hypot(dx, dy)
            if dist < 1e-9:
                return -r, 1.0, 0.0
            return dist - r, dx / dist, dy / dist
        lx, ly, hw, hh = self._to_box_frame(o, px, py)
        qx, qy = abs(lx) - hw, abs(ly) - hh
        if qx > 0 or qy > 0:                      # outside
            ox, oy = max(qx, 0.0), max(qy, 0.0)
            dist = np.hypot(ox, oy)
            nx = np.sign(lx) * (ox / dist if dist else 0.0)
            ny = np.sign(ly) * (oy / dist if dist else 0.0)
        else:                                     # inside: nearest face
            dist = max(qx, qy)
            if qx > qy:
                nx, ny = np.sign(lx) or 1.0, 0.0
            else:
                nx, ny = 0.0, np.sign(ly) or 1.0
        yaw = o[5]
        c, sn = np.cos(yaw), np.sin(yaw)
        return dist, nx * c - ny * sn, nx * sn + ny * c

    def _ray_hit(self, angle, max_range=LIDAR_MAX_RANGE):
        """Range from the robot to the first obstacle along a world-frame
        ray, or max_range if it hits nothing."""
        dx, dy = np.cos(angle), np.sin(angle)
        best = max_range
        for o in self.obstacles:
            if o[0] == "circle":
                _, ox, oy, r = o
                fx, fy = ox - self.robot_x, oy - self.robot_y
                t = fx * dx + fy * dy
                if t <= 0:
                    continue
                perp2 = (fx - t * dx) ** 2 + (fy - t * dy) ** 2
                if perp2 >= r * r:
                    continue
                hit = t - np.sqrt(r * r - perp2)
            else:
                # slab method in the box's own frame
                yaw = o[5]
                c, sn = np.cos(-yaw), np.sin(-yaw)
                lx, ly, hw, hh = self._to_box_frame(o, self.robot_x, self.robot_y)
                rx, ry = dx * c - dy * sn, dx * sn + dy * c
                t0, t1 = 0.0, best
                for p, half, dr in ((lx, hw, rx), (ly, hh, ry)):
                    if abs(dr) < 1e-9:
                        if abs(p) > half:
                            t1 = -1.0
                        continue
                    ta, tb = (-half - p) / dr, (half - p) / dr
                    t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
                if t1 < t0:
                    continue
                hit = t0
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
        keep_out = ROBOT_RADIUS
        for _ in range(3):
            for o in self.obstacles:
                dist, nx, ny = self._surface_distance(o, self.robot_x, self.robot_y)
                if dist >= keep_out:
                    continue
                push = keep_out - dist
                self.robot_x += nx * push
                self.robot_y += ny * push
                self.cmd_v = 0.0           # contact stops the drive
            dx, dy = self.robot_x - self.person_x, self.robot_y - self.person_y
            dist = np.hypot(dx, dy)
            body = PERSON_RADIUS + ROBOT_RADIUS
            if dist < body:
                if dist < 1e-6:
                    dx, dy, dist = 1.0, 0.0, 1.0
                self.robot_x = self.person_x + dx / dist * body
                self.robot_y = self.person_y + dy / dist * body
                self.cmd_v = 0.0

    def _obstacle_penalty(self):
        """How far the robot has pushed into an obstacle's safe zone."""
        pen = 0.0
        for o in self.obstacles:
            dist = self._surface_distance(o, self.robot_x, self.robot_y)[0]
            gap = dist - (ROBOT_RADIUS + 0.1)
            pen += OBSTACLE_PENALTY_SCALE * max(0.0, -gap)
        return pen

    def _walls(self, W, H, t=0.05):
        return [("box", W / 2, -t, W / 2, t, 0.0),
                ("box", W / 2, H + t, W / 2, t, 0.0),
                ("box", -t, H / 2, t, H / 2, 0.0),
                ("box", W + t, H / 2, t, H / 2, 0.0)]

    def _free(self, x, y, margin):
        return all(self._surface_distance(o, x, y)[0] > margin for o in self.obstacles)

    def _chest_pose_exists(self):
        """Cheap feasibility test: is there at least one spot on the chest
        side of the person that the robot fits in and can see them from?

        This is the generator's filter, so it has to be affordable per
        episode: it samples the chest sector rather than flood-filling the
        room, and pairs with a line-of-sight check from the robot's start,
        which keeps both bodies in the same open region. eval/feasibility.py
        does the full reachability version when scoring results."""
        for k in range(16):
            ang = self.person_phi + np.radians(-45 + 90 * k / 15)
            for r in (MIN_CENTER_DISTANCE + 0.05, 0.75, 0.95):
                x = self.person_x + r * np.cos(ang)
                y = self.person_y + r * np.sin(ang)
                if not all(self._surface_distance(o, x, y)[0] > ROBOT_RADIUS
                           for o in self.obstacles):
                    continue
                x0, y0 = self.robot_x, self.robot_y
                self.robot_x, self.robot_y = x, y
                seen = self._person_visible(np.hypot(self.person_x - x, self.person_y - y))
                self.robot_x, self.robot_y = x0, y0
                if seen:
                    return True
        return False

    def _make_random_room(self, tries=20):
        """A room like the bedroom but drawn fresh: walls, a few pieces of
        furniture at random sizes, places and angles, person and robot
        dropped into whatever floor is left.

        Layouts where no chest-facing pose is reachable are redrawn rather
        than trained on: the robot cannot succeed in them, so they teach it
        nothing and only add noise to the return. 18% of unfiltered layouts
        were unwinnable in this way.
        """
        for _ in range(tries):
            self._draw_random_room()
            if self._chest_pose_exists() and self._person_visible(
                    np.hypot(self.person_x - self.robot_x,
                             self.person_y - self.robot_y)):
                return

    def _draw_random_room(self):
        W = self.np_random.uniform(*ROOM_SIZE_RANGE[0])
        H = self.np_random.uniform(*ROOM_SIZE_RANGE[1])
        self.obstacles = self._walls(W, H)
        n = int(self.np_random.integers(ROOM_FURNITURE[0], ROOM_FURNITURE[1] + 1))
        for _ in range(n):
            hw = self.np_random.uniform(0.2, 1.0)
            hh = self.np_random.uniform(0.2, 0.7)
            for _try in range(20):
                cx = self.np_random.uniform(hw, W - hw)
                cy = self.np_random.uniform(hh, H - hh)
                yaw = self.np_random.uniform(-np.pi, np.pi)
                cand = ("box", cx, cy, hw, hh, yaw)
                if self._surface_distance(cand, W / 2, H / 2)[0] > 0.0:
                    self.obstacles.append(cand)
                    break

        for _ in range(200):       # person somewhere with room to be approached
            px = self.np_random.uniform(0.6, W - 0.6)
            py = self.np_random.uniform(0.6, H - 0.6)
            if self._free(px, py, PERSON_RADIUS + 0.55):
                self.person_x, self.person_y = px, py
                break
        else:
            self.person_x, self.person_y = W / 2, H / 2
        self.person_phi = self.np_random.uniform(-np.pi, np.pi)

        for _ in range(200):
            x = self.np_random.uniform(0.3, W - 0.3)
            y = self.np_random.uniform(0.3, H - 0.3)
            if (np.hypot(x - self.person_x, y - self.person_y) > SPAWN_R_MIN
                    and self._free(x, y, ROBOT_RADIUS + 0.15)):
                self.robot_x, self.robot_y = x, y
                break
        self.robot_yaw = self.np_random.uniform(-np.pi, np.pi)

    def _make_bedroom(self):
        """Walls, bed, nightstand, dresser; person on the bed edge facing
        into the room; robot dropped anywhere with clearance."""
        W, H = BEDROOM_SIZE
        self.obstacles = self._walls(W, H) + [
            ("box", 1.1, 0.85, 1.0, 0.7, 0.0),     # bed
            ("box", 2.45, 0.30, 0.25, 0.20, 0.0),  # nightstand
            ("box", 3.65, 2.20, 0.25, 0.60, 0.0),  # dresser
        ]
        self.person_x, self.person_y = BEDROOM_PERSON
        self.person_phi = self.np_random.uniform(
            np.pi / 2 - np.radians(BEDROOM_FACING_SPREAD_DEG),
            np.pi / 2 + np.radians(BEDROOM_FACING_SPREAD_DEG))

        if self.tight_start:
            self.robot_x, self.robot_y = TIGHT_START
        else:
            for _ in range(200):
                x = self.np_random.uniform(0.3, W - 0.3)
                y = self.np_random.uniform(0.3, H - 0.3)
                if np.hypot(x - self.person_x, y - self.person_y) < SPAWN_R_MIN:
                    continue
                self.robot_x, self.robot_y = x, y
                if all(self._surface_distance(o, x, y)[0] > ROBOT_RADIUS + 0.15
                       for o in self.obstacles):
                    break
        self.robot_yaw = self.np_random.uniform(-np.pi, np.pi)

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

    def _person_motion_mm(self):
        """Advance the person's random body movement and return its current
        amplitude in mm. A slow random walk around SWAY_MM, with occasional
        larger postural shifts on top."""
        a = self.dt / SWAY_TAU_S
        self.sway_mm = abs((1 - a) * self.sway_mm
                           + a * SWAY_MM
                           + self.np_random.normal(0.0, SWAY_MM * 0.5) * a ** 0.5)
        if self.shift_left > 0.0:
            self.shift_left -= self.dt
            return self.sway_mm + self.shift_amp
        if self.np_random.uniform() < SHIFT_PROB_PER_S * self.dt:
            self.shift_amp = self.np_random.uniform(*SHIFT_MM)
            self.shift_left = self.np_random.uniform(*SHIFT_DURATION_S)
            return self.sway_mm + self.shift_amp
        return self.sway_mm

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

        # cardiac_band_snr in dB, and only when a reading can physically
        # exist: line of sight, a target in the link budget, and the platform
        # at rest long enough for the chassis to settle.
        at_rest = self.still_for >= SETTLE_S
        snr_ok = (visible and at_rest
                  and link_budget_db(max(d, 1e-3), np.degrees(b)) >= RADAR_SNR_FLOOR_DB)
        q = predicted_band_snr(max(d, 1e-3), np.degrees(b), np.degrees(a))
        # the person's own movement raises the in-band noise the metric
        # divides by, so it degrades the reading rather than merely offsetting it
        rbm = self._person_motion_mm()
        q = CARDIAC_NOISE_FLOOR + (q - CARDIAC_NOISE_FLOOR) / (1.0 + (rbm / A_CARDIAC_MM) ** 2)
        true_snr = 10.0 * np.log10(max(q, 1e-6))
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

        if self.layout == "empty":
            # No obstacles at all: the LiDAR sectors all read max range and
            # the person is never occluded. Same 17-value observation as the
            # obstacle layouts, so one code base and one observation format
            # covers every condition the paper reports.
            self.obstacles = []
        elif self.layout == "bedroom":
            self._make_bedroom()
        elif self.layout == "room":
            self._make_random_room()
        else:
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
                    self.obstacles.append(("circle", ox, oy, orad))
                    break

        self.step_count = 0
        self.cmd_v, self.cmd_omega = 0.0, 0.0   # starts at rest
        self.still_for = 0.0                    # seconds at rest, for SETTLE_S
        self.sway_mm = SWAY_MM                  # person's current RBM amplitude
        self.shift_left = 0.0                   # seconds of postural shift left

        return self._obs(), {}

    def step(self, action):
        # the action is a velocity REQUEST; the drive ramps toward it within
        # its acceleration limits rather than snapping there instantly
        want_v, want_omega = float(action[0]), float(action[1])
        max_dv = MAX_LINEAR_ACCEL * self.dt
        max_dw = MAX_ANGULAR_ACCEL * self.dt
        self.cmd_v += np.clip(want_v - self.cmd_v, -max_dv, max_dv)
        self.cmd_omega += np.clip(want_omega - self.cmd_omega, -max_dw, max_dw)

        if abs(self.cmd_v) <= STILL_V and abs(self.cmd_omega) <= STILL_W:
            self.still_for += self.dt
        else:
            self.still_for = 0.0

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
