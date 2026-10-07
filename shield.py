import numpy as np

# --- chest-facing shield ---------------------------------------------------
# PPO handles approach and positioning, but it never learns to get around
# to the person's chest once it arrives somewhere else: leaving means
# driving around them, which for a diff drive with a front-mounted radar
# means turning away and losing the reading for a while -- every
# intermediate step scores worse, so trial and error never finds it. The
# reward is not the problem; a scripted orbit scores higher under it.
#
# So this is a shield in the sense of Alshiekh et al. (AAAI 2018): it lets
# the policy's action through untouched, and takes over only while the
# robot is close and NOT in front of the person (side-on or at the back).
#
# It is split into two parts, the usual layering in mobile robot navigation:
#   goal_pose() -- WHERE to go: in front of the chest, facing them, at a
#                  safe standoff. Robot-agnostic.
#   _drive_to() -- HOW to get there: a simple planner that goes wide around
#                  the person. This is the only robot-specific part, and the
#                  part that would need obstacle avoidance once the sim has
#                  obstacles.
# Once the robot is in front (|phi| < RELEASE_PHI_DEG) it hands back to the
# policy, which settles the final distance by SNR as usual -- the shield
# only decides which SIDE of the person, never the final spot.
#
# Chest over back is a deliberate choice: the SNR model ranks them nearly
# equal (0.4 dB), but the chest is the better-supported side. If the front
# is unreachable, `front_blocked=True` falls back to the nearer good side.
# Nothing sets it yet -- the sim has no obstacles besides the person.
#
# It reads the OBSERVATION only (LiDAR range/bearing/phi and their validity
# flags), never simulator state, and recomputes the goal every step, so it
# needs no odometry and follows the person if they turn.

ENGAGE_RANGE = 1.2          # m -- only close in, where aspect actually costs signal
ENGAGE_PHI_DEG = 60.0       # |phi| above this -> take over (side-on or back)
RELEASE_PHI_DEG = 30.0      # |phi| below this -> hand back (hysteresis; releasing
                            # at 40 left PPO drifting back to ~47deg)
BACK_OK_DEG = 150.0         # front_blocked only: the back counts as good past this

GOAL_RANGE = 0.9            # m -- standoff for the goal pose and the way round.
                            # ~28cm outside the collision floor (0.62m); PPO
                            # closes in from here if SNR says it should
WAYPOINT_STEP_DEG = 30.0    # planner: step around the person this far at a time
DRIVE_SPEED = 0.4           # m/s
HEADING_GAIN = 3.0          # omega per rad of heading error


def goal_pose(d, bearing, phi, toward_back=False):
    """WHERE: the pose in front of the person (or behind, if toward_back),
    facing them, at GOAL_RANGE -- in the robot's own frame, from one LiDAR
    reading. Returns (goal_xy, person_xy, side), where side is the goal's
    angle around the person."""
    person = d * np.array([np.cos(bearing), np.sin(bearing)])
    # phi is the robot's angle around the person measured from their facing
    # direction, and the robot sits at angle (bearing + pi) as seen from them
    facing = bearing + np.pi - phi
    side = facing + np.pi if toward_back else facing
    goal = person + GOAL_RANGE * np.array([np.cos(side), np.sin(side)])
    return goal, person, side


class ChestShield:
    def __init__(self):
        self.active = False
        self.toward_back = False

    def reset(self):
        self.active = False

    def __call__(self, obs, policy_action, front_blocked=False):
        """Returns (action, overridden). obs is the env's raw observation."""
        d, bearing, phi = float(obs[0]), float(obs[1]), float(obs[2])
        range_ok, phi_ok = obs[4] > 0.5, obs[5] > 0.5
        abs_phi = abs(np.degrees(phi))
        at_back = front_blocked and abs_phi > BACK_OK_DEG

        if not (range_ok and phi_ok):
            self.active = False
        elif self.active:
            if abs_phi < RELEASE_PHI_DEG or at_back:
                self.active = False
        elif d < ENGAGE_RANGE and abs_phi > ENGAGE_PHI_DEG and not at_back:
            self.active = True
            self.toward_back = front_blocked and abs_phi > 90.0

        if not self.active:
            return policy_action, False
        goal, person, side = goal_pose(d, bearing, phi, self.toward_back)
        return self._drive_to(goal, person, side), True

    def _drive_to(self, goal, person, side):
        """HOW: if the goal is more than one step
        around the person, aim for the next point on the GOAL_RANGE circle
        in that direction -- so the path spirals out to a safe standoff and
        round, never cutting across the person. Otherwise go straight."""
        here = np.arctan2(-person[1], -person[0])      # robot's angle around the person
        gap = np.arctan2(np.sin(side - here), np.cos(side - here))
        if abs(gap) > np.radians(WAYPOINT_STEP_DEG):
            ang = here + np.sign(gap) * np.radians(WAYPOINT_STEP_DEG)
            target = person + GOAL_RANGE * np.array([np.cos(ang), np.sin(ang)])
        else:
            target = goal
        err = np.arctan2(target[1], target[0])          # robot faces +x in its own frame
        v = DRIVE_SPEED if abs(err) < 0.5 else 0.0
        omega = np.clip(HEADING_GAIN * err, -1.5, 1.5)
        return np.array([v, omega], dtype=np.float32)


# --- hold ----------------------------------------------------------------
# Once the robot is in a good spot it should STOP, completely, and stay
# stopped. Two reasons:
#   1. The radar needs it. cardiac_band_snr reads mm-scale chest motion from
#      phase over a 10s+ window; any robot motion -- even the policy's +-3deg
#      settling wiggle -- adds phase change far larger than a heartbeat.
#   2. People. A robot that keeps twitching beside someone is exactly what
#      bothers them. Deployed robots arrive within a tolerance and stop.
#
# Stop on POSE (instant, from LiDAR). Stay on SNR -- averaged over a window
# the way the real reading is, which is only meaningful once stationary.
# Move again only if the SNR stays low, or the pose is clearly wrong, for
# RELEASE_PERSIST_S straight -- so a person turning for a second doesn't
# set the robot off, but a person who actually turns away does.

AIM_GAIN = 2.0              # omega per rad of bearing error while aiming
HOLD_THETA_DEG = 15.0       # enter: facing the person to within this
HOLD_PHI_DEG = 30.0         #        and within this of their chest
HOLD_MAX_RANGE = 1.0        #        and at least this close (m)
LEAVE_THETA_DEG = 30.0      # leave: facing off by more than this...
LEAVE_PHI_DEG = 45.0        #        or off the chest by more than this ("usable to 45")
LEAVE_MAX_RANGE = 1.2       #        or farther than this...
SNR_HOLD_MIN = 9.0          #        or averaged cardiac_band_snr below this
                            #        (13.2 peak at the chest, 9.2 at phi=45deg,
                            #        5.3 noise floor)
SNR_WINDOW_S = 2.0          # integrate a reading for this long once at rest.
                            # The robot stops, the radar becomes valid after
                            # the chassis settles, and this is how long the
                            # reading is averaged before it is judged: good
                            # enough to keep, or move on and try elsewhere.
RELEASE_PERSIST_S = 2.0     # ...for this long straight, not a blip


class Hold:
    def __init__(self, dt=0.1, snr_obs_scale=20.0):
        self.dt = dt
        self.snr_obs_scale = snr_obs_scale
        self.reset()

    def reset(self):
        self.holding = False
        self.bad_for = 0.0
        self.snr_avg = None
        self.held_for = 0.0
        self.meas_for = 0.0

    def __call__(self, obs, action):
        """Returns (action, holding). Zero velocity while holding."""
        d, bearing, phi = float(obs[0]), float(obs[1]), float(obs[2])
        range_ok, phi_ok, snr_ok = obs[4] > 0.5, obs[5] > 0.5, obs[6] > 0.5
        theta, aphi = abs(np.degrees(bearing)), abs(np.degrees(phi))

        if not self.holding:
            in_place = (range_ok and phi_ok and d < HOLD_MAX_RANGE
                        and aphi < HOLD_PHI_DEG)
            if in_place and theta >= HOLD_THETA_DEG:
                # Right spot, wrong heading: turn in place to face the person
                # and then hold. Turning costs nothing -- it moves the robot
                # nowhere -- and without it the robot can sit at a good
                # distance and aspect but a few tens of degrees off boresight
                # and never settle. That is the common case in a cluttered
                # room, where the policy stops wherever it fits rather than
                # wherever it is aimed (26deg median in the bedroom layout,
                # against 7deg in open space).
                return np.array([0.0, np.clip(AIM_GAIN * bearing, -1.5, 1.5)],
                                dtype=np.float32), True
            if in_place and theta < HOLD_THETA_DEG:
                self.holding, self.bad_for = True, 0.0
                self.held_for = self.meas_for = 0.0
                self.snr_avg = None            # restart the reading once still
            else:
                return action, False

        # Holding: integrate the reading, but only while it actually exists.
        # The radar channel is invalid until the platform has been at rest
        # long enough to settle, so measurement time is counted separately
        # from time spent holding.
        self.held_for += self.dt
        if snr_ok:
            self.meas_for += self.dt
            snr = 10.0 ** (obs[3] * self.snr_obs_scale / 10.0)
            a = max(self.dt / SNR_WINDOW_S, self.dt / self.meas_for)
            self.snr_avg = snr if self.snr_avg is None else (1 - a) * self.snr_avg + a * snr
        snr_ready = self.meas_for >= SNR_WINDOW_S

        pose_bad = (not range_ok or d > LEAVE_MAX_RANGE or theta > LEAVE_THETA_DEG
                    or (phi_ok and aphi > LEAVE_PHI_DEG))
        snr_bad = snr_ready and self.snr_avg is not None and self.snr_avg < SNR_HOLD_MIN

        # Both a poor reading and a poor pose must persist before the robot
        # gives up on the spot, which in practice means a second measurement
        # window agreeing with the first. Without it, a single postural shift
        # by the person -- which drives the metric to its noise floor for a
        # second or two -- would send the robot away from a good pose.
        self.bad_for = self.bad_for + self.dt if (pose_bad or snr_bad) else 0.0
        if self.bad_for >= RELEASE_PERSIST_S:
            self.holding = False
            return action, False
        return np.zeros(2, dtype=np.float32), True


# --- re-acquire -----------------------------------------------------------
# When the person goes behind furniture every channel reads zero, and the
# observation carries no memory, so the robot has nothing to act on and
# stalls wherever it happens to be. In randomised furnished rooms this is
# half of all failures: the robot settles ~1.5m out with the bed between it
# and the person, unable to notice.
#
# This rule gives the stack the one piece of memory it needs: the person's
# last known position, dead-reckoned from the robot's own commanded velocity
# while they are out of sight. If they stay lost, it drives an arc around
# that position -- toward whichever side the LiDAR says is clearer -- until
# line of sight returns. Sideways is the only useful direction here: driving
# at the last known position keeps the same obstacle in the way.

LOST_S = 1.0                # unseen this long before circling
REACQUIRE_MAX_R = 3.5       # only circle for someone lost from WITHIN lidar range
ARC_OFFSET_DEG = 60.0       # how far around the last known position to aim
ARC_SPEED = 0.35            # m/s
N_SECTORS = 8               # lidar sectors in the observation, from index 9


class Reacquire:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.reset()

    def reset(self):
        self.last = None        # person position in the robot's frame
        self.lost_for = 0.0

    def __call__(self, obs, action):
        """Returns (action, searching)."""
        d, bearing = float(obs[0]), float(obs[1])
        range_ok = obs[4] > 0.5
        v, w = float(obs[7]), float(obs[8])

        if range_ok:
            self.last = (d * np.cos(bearing), d * np.sin(bearing))
            self.lost_for = 0.0
            return action, False

        self.lost_for += self.dt
        if self.last is None:
            return action, False                 # never seen them; nothing to go on

        # dead-reckon the last known position as the robot drives
        px, py = self.last
        c, s = np.cos(w * self.dt), np.sin(w * self.dt)
        px, py = px - v * self.dt, py
        self.last = (px * c + py * s, -px * s + py * c)

        if self.lost_for < LOST_S:
            return action, False

        px, py = self.last
        r = float(np.hypot(px, py))
        # Only a person who WAS in lidar range and then vanished has been
        # occluded. obs[4] also drops when they are simply beyond the lidar's
        # reach, and circling then fights the policy's own approach.
        if r > REACQUIRE_MAX_R:
            return action, False
        b = float(np.arctan2(py, px))
        # circle toward the side with more room, judged from the lidar sectors
        sectors = np.asarray(obs[9:9 + N_SECTORS], dtype=float)
        left = sectors[1:4].mean()               # 45-135 deg
        right = sectors[-3:].mean()              # -135 to -45 deg
        side = 1.0 if left >= right else -1.0
        target = b + side * np.radians(ARC_OFFSET_DEG)
        err = np.arctan2(np.sin(target), np.cos(target))
        speed = ARC_SPEED if abs(err) < 0.6 and r > 0.4 else 0.0
        return np.array([speed, np.clip(HEADING_GAIN * err, -1.5, 1.5)],
                        dtype=np.float32), True

