import numpy as np


def distance_and_angle(robot_x, robot_y, robot_yaw,
                       person_x, person_y):
    """
    Compute how well-placed the robot is relative to the person.
    Returns (distance, bearing) where:
      - distance = how far the robot is from the person (meters)
      - bearing  = angle to the person relative to where the robot faces
                   (0 = robot pointing straight at person)
    """
    dx = person_x - robot_x
    dy = person_y - robot_y

    distance = np.hypot(dx, dy)

    angle_to_person = np.arctan2(dy, dx)
    bearing = angle_to_person - robot_yaw
    # sin/cos round-trip forces the result into [-pi, pi] regardless of how
    # far angle_to_person and robot_yaw drift apart
    bearing = np.arctan2(np.sin(bearing), np.cos(bearing))

    return distance, bearing


def aspect_angle(robot_x, robot_y, person_x, person_y, person_yaw):
    """
    Where is the robot standing relative to which way the person is facing?
    0 = directly in front of the person, +-pi = directly behind them.
    """
    dx = robot_x - person_x
    dy = robot_y - person_y

    angle_from_person = np.arctan2(dy, dx)
    aspect = angle_from_person - person_yaw
    aspect = np.arctan2(np.sin(aspect), np.cos(aspect))

    return aspect


# distance is measured center-to-center between robot and person, so the
# "minimum standoff" needs to account for how wide each body actually is,
# otherwise a center distance that looks safe on paper still overlaps in
# reality (r2d2's footprint alone is ~0.17m).
ROBOT_RADIUS = 0.172   # r2d2 footprint half-width, from pybullet AABB (m)
PERSON_RADIUS = 0.15   # person marker radius (m)
MIN_STANDOFF = 0.3     # desired clear surface gap once at min distance (m)
MAX_STANDOFF = 1.0     # upper bound on center distance (m)
MIN_CENTER_DISTANCE = MIN_STANDOFF + ROBOT_RADIUS + PERSON_RADIUS


# --- Q(distance, theta, phi) = A_lookup(distance) x B_lookup(theta, phi) ---
# Real cnn_band_snr data from mmResp/table_a_distance_posture.csv and
# table_b_angle_phi.csv (built 2026-08-03 from the 60xx distance/posture and
# angle-accuracy trials). Both tables use the same metric. Replaces the
# placeholder geometric reward() this used to be -- the actual sensor model
# now, not a stub.

# Table A: distance(m) -> band_snr, averaged over posture (sit/stand) since
# the sim doesn't model posture. Only 0.7m/0.9m were captured within the
# CNN's trained/validated range (<=1m, see MAX_STANDOFF) -- 1.2m trials were
# excluded on scope, not noise, grounds.
_TABLE_A = {
    0.7: (25.745 + 24.759) / 2,   # 25.252
    0.9: (26.657 + 17.518) / 2,   # 22.088
}

# Table B: (theta_deg, phi_deg) -> band_snr, normalized to its best cell
# (45, 45) = 1.0. theta = bearing (is the person in the sensor's FOV);
# phi = aspect (is the person facing the sensor). Same metric as Table A
# (cnn_band_snr), same median+cap treatment. Sparse, non-rectangular grid --
# nearest-neighbor lookup, not interpolation.
_TABLE_B = {
    (-90.0, -90.0): 0.3627,
    (-45.0, -45.0): 0.6668,
    (-45.0, 0.0): 0.4988,
    (0.0, 0.0): 0.6694,
    (0.0, 180.0): 0.6875,
    (45.0, 0.0): 0.9370,
    (45.0, 45.0): 1.0000,
    (90.0, 90.0): 0.1609,
}

FOV_GATE_DEG = 60.0            # |theta| beyond this -> no trusted reading
COLLISION_PENALTY_SCALE = 200.0  # per meter crossed past the safe-zone floor

# Beyond MAX_STANDOFF / the FOV gate there's no real data, but a hard floor
# at exactly 0 leaves no gradient at all out there -- confirmed empirically
# this makes exploration essentially impossible (4000 random steps across 20
# episodes, reward never once left 0, robot never got closer than 1.44m,
# needs <1.0m). These slopes are deliberately gentler than the narrow
# in-range taper (which drops ~22 over just 0.1m) so a smooth continuation
# of *that* slope would be absurdly steep -- instead they're their own small
# continuous decay, giving a real "closer/more-facing is better" gradient
# everywhere without letting it dominate the actual data once in range.
DISTANCE_DECAY_SLOPE = 2.0   # per meter beyond MAX_STANDOFF
# FOV violation is a quadratic (not linear) additive penalty, scaled large
# enough to dominate even the best possible Q once meaningfully past the
# gate. A weak linear penalty isn't enough to overcome an unexpectedly-good
# nearest-neighbor B lookup (e.g. 70deg, just past the gate, is
# geometrically closer to the table's 45deg cell than anywhere else, so B
# returns a near-best value there) -- and a *multiplicative* suppression on
# Q was tried and rejected: when A(distance) is already negative (past
# MAX_STANDOFF, see _lookup_table_a), shrinking Q by a fraction makes a
# negative number LESS negative, i.e. an FOV violation would look like it
# partially cancels a too-far penalty instead of stacking with it. A large
# quadratic additive term avoids that sign issue entirely and grows fast
# enough to dominate regardless of what B returns.
#
# CAPPED, though -- uncapped quadratic growth explodes at extreme angles
# (bearing=180deg, facing directly away, is completely reachable during
# normal exploration since nothing constrains yaw) to -2189 at 500 scale,
# vs. Q's ~0-25 range and the collision penalty's naturally-bounded ~124
# max. That reward-scale mismatch wrecked PPO's value function in practice
# (confirmed: ep_rew_mean around -5000 and getting worse, value_loss/loss
# swinging between 1e4 and 1e7 iteration to iteration, explained_variance
# stuck at ~0 -- the network can't fit targets that swing between ~20 and
# ~-2200 in the same batch). The cap keeps the steep near-gate growth
# (needed to fix the 70deg bug) while bounding the worst case to something
# comparable to the rest of the reward's scale.
FOV_PENALTY_SCALE = 500.0    # per radian^2 beyond the FOV gate
FOV_PENALTY_CAP = 150.0      # max FOV penalty, however far past the gate

# phi has a different shape than theta: theta is one contiguous valid band
# (a cone in front of the radar), but phi has TWO valid islands (frontal,
# phi~0, and back, phi~180) separated by a dead-zone at the sides (phi~+-90)
# -- chest/back both present a surface with radial motion to the radar,
# side-on gives mostly tangential motion plus arm occlusion. Table B's
# nearest-neighbor lookup already has data points at these angles, but
# nearest-neighbor between sparse points is exactly the failure mode that
# caused the FOV bug above (a phi value between table points can snap to
# a spuriously good neighbor) -- this term adds an explicit, physically-
# motivated push out of the dead-zone regardless of what the lookup
# returns. sin(phi)^2 is 0 at phi=0/180 (both valid islands) and peaks at
# phi=+-90 (the dead-zone) -- the natural shape given radial motion scales
# roughly with cos(phi): sin^2 = 1 - cos^2 is exactly the lost-radial-
# motion term. Soft, not a hard reject, for the same reason the FOV gate
# is soft: a hard cutoff leaves no gradient for the policy to climb out on.
PHI_DEADZONE_SCALE = 12.0    # penalty at phi=+-90 (peak); 0 at phi=0/180


def _angle_diff_deg(a_deg, b_deg):
    """Smallest signed difference between two angles in degrees (handles the
    -180/180 wrap so e.g. 180 and -180 count as 0 apart, not 360)."""
    a, b = np.radians(a_deg), np.radians(b_deg)
    return np.degrees(np.arctan2(np.sin(a - b), np.cos(a - b)))


def _lookup_table_a(distance):
    """A hill, symmetric on both sides of the measured [0.7, 0.9] range --
    NOT monotonically "closer is always better" (that was tried and
    rejected: extending the measured trend downward put the reward peak
    exactly at the collision boundary, MIN_CENTER_DISTANCE, which is both
    unsupported by data -- there are zero measurements below 0.7m -- and
    bad for training, since a peak sitting exactly on a penalty boundary is
    much harder for a noisy policy to balance on than a peak safely inside
    a plateau).

    Between the two measured points, linear interpolation (real data).
    Above 0.9 (data's far edge), tapers linearly to 0 at MAX_STANDOFF, then
    continues its own gentle decay past that (DISTANCE_DECAY_SLOPE) rather
    than flooring at 0. Below 0.7 (data's near edge), mirrors that: tapers
    linearly to 0 at MIN_CENTER_DISTANCE, giving a real "back off" gradient
    that starts before the hard collision penalty, pointing toward the
    measured sweet spot instead of toward the safety boundary."""
    points = sorted(_TABLE_A)
    lo, hi = points[0], points[-1]
    if distance <= lo:
        t = (distance - MIN_CENTER_DISTANCE) / (lo - MIN_CENTER_DISTANCE)
        return _TABLE_A[lo] * max(0.0, t)
    if distance <= hi:
        t = (distance - lo) / (hi - lo)
        return _TABLE_A[lo] + t * (_TABLE_A[hi] - _TABLE_A[lo])
    if distance <= MAX_STANDOFF:
        t = (distance - hi) / (MAX_STANDOFF - hi)
        return _TABLE_A[hi] * (1.0 - t)
    return -DISTANCE_DECAY_SLOPE * (distance - MAX_STANDOFF)


def _lookup_table_b(theta_deg, phi_deg):
    """Nearest-neighbor lookup by circular angular distance -- the (theta,
    phi) grid is too sparse/irregular for interpolation."""
    best_key, best_dist = None, None
    for (t, p) in _TABLE_B:
        dt = _angle_diff_deg(theta_deg, t)
        dp = _angle_diff_deg(phi_deg, p)
        dist = dt * dt + dp * dp
        if best_dist is None or dist < best_dist:
            best_key, best_dist = (t, p), dist
    return _TABLE_B[best_key]


def reward(distance, bearing, aspect):
    """
    Score how good this pose is for sensing: Q = A(distance) x B(theta, phi),
    minus penalties for crossing the collision safe-zone, the FOV gate, or
    phi's side dead-zone (phi near +-90). Higher = better. theta = bearing,
    phi = aspect, both in radians.

    No hard floor at 0 anywhere -- A(distance) keeps sloping down past
    MAX_STANDOFF (see _lookup_table_a) instead of clamping, and instead of
    a hard FOV cutoff, B is still looked up (nearest-neighbor extrapolates
    fine past +-90deg) and a continuous penalty is subtracted for how far
    outside the FOV gate theta is. Same soft-penalty treatment for phi's
    dead-zone (see PHI_DEADZONE_SCALE) -- one smooth surface throughout,
    not a floored value plus separately bolted-on shaping terms.
    """
    theta_deg = np.degrees(bearing)
    phi_deg = np.degrees(aspect)
    q = _lookup_table_a(distance) * _lookup_table_b(theta_deg, phi_deg)

    too_close = max(0.0, MIN_CENTER_DISTANCE - distance)
    collision_penalty = COLLISION_PENALTY_SCALE * too_close

    too_wide = max(0.0, abs(bearing) - np.radians(FOV_GATE_DEG))
    fov_penalty = min(FOV_PENALTY_SCALE * too_wide ** 2, FOV_PENALTY_CAP)

    phi_deadzone_penalty = PHI_DEADZONE_SCALE * np.sin(aspect) ** 2

    return q - collision_penalty - fov_penalty - phi_deadzone_penalty


if __name__ == "__main__":
    # robot at origin facing +x, person 0.8m ahead, chest turned toward the robot
    d, b = distance_and_angle(0, 0, 0, 0.8, 0)
    a = aspect_angle(0, 0, 0.8, 0, np.pi)
    print("distance:", d, "bearing:", b, "aspect:", a, "reward:", reward(d, b, a))
