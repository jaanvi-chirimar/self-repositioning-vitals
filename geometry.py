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


# --- Q(distance, theta, phi) = Q_SCALE x quality(snr_db(distance, theta, phi)) ---
#
# Two steps, replacing the old factorized TableA x TableB lookup (2026-09-20):
#
#   1. snr_db()  -- the radar range equation, every term at once. Range
#      contributes its R^-4 loss, the antenna pattern contributes the
#      angular (theta) loss, and target aspect (phi) contributes how much
#      of the chest's motion is radial to the sensor. All in dB, all added.
#   2. quality() -- one saturating curve mapping SNR to "can the estimator
#      actually pull a heartbeat out of this".
#
# Why not keep the factorized form: it forced the angular penalty to be a
# fixed FRACTION regardless of range, i.e. 45deg off-boresight cost the same
# proportion at 0.7m as at 3m. That's false -- close in you have SNR margin
# to burn, so the angular loss stays above threshold; far out the R^-4 loss
# already spent that margin and the same angular loss drops you under. The
# unified form gets that coupling for free; the factorized one could not
# express it at all. (The project notes had already flagged the unverified
# assumption: "Table B is measured at a single fixed distance (~0.6m) -- the
# multiplicative model assumes the angular penalty shape is distance-
# invariant, which no current data actually tests.")
#
# Everything here is SNR-derived. Ground-truth heart-rate error is NOT used
# anywhere in this model -- it is held out purely as an independent check
# (see check_against_error() at the bottom of this file).

# Table A: distance(m) -> band_snr, averaged over posture (sit/stand) since
# the sim doesn't model posture. Only 0.7m/0.9m were captured within the
# CNN's trained/validated range (<=1m, see MAX_STANDOFF) -- 1.2m trials were
# excluded on scope, not noise, grounds.
_TABLE_A = {
    0.7: (25.745 + 24.759) / 2,   # 25.252
    0.9: (26.657 + 17.518) / 2,   # 22.088
}

# Table B USED to be a measured (theta, phi) -> band_snr lookup (the values
# still live in mmResp/table_b_angle_phi.csv). It was replaced 2026-09-20
# with the physical model below, for three measured reasons:
#
#  1. As a sparse 8-point grid with nearest-neighbor lookup, it was a
#     staircase, not a surface: along theta at phi=0 it took exactly TWO
#     values over the whole 0-90deg range (0.6694 below 25deg, 0.9370 above,
#     one 40% cliff between). Flat plateaus separated by discontinuous
#     jumps -- no gradient anywhere for a policy to climb. Diagnosed from
#     the trained policy's behavior: it solved distance and bearing (0.80m
#     and inside the gate on 40/40 episodes) but left phi completely
#     uncontrolled (mean |phi| 90.8deg, i.e. the dead zone, scattered
#     uniformly) and froze in place (|v|=|omega|=0.000 on every episode).
#     It never orbited, because nothing in B ever told it which way to go.
#  2. Its theta axis contradicted the antenna physics outright -- it scored
#     45deg and 90deg off-boresight as BETTER than boresight (0.9370 vs
#     0.6694), where the two-way pattern says 45deg is ~63x worse. That is
#     the failure mode already documented in the project notes: band_snr
#     "can be misleadingly confident at extreme angles (locks onto a false
#     peak) -- do not use it to detect FOV boundary." Table B spanned the
#     full angular range anyway, including the +-90 cells.
#  3. Its phi axis is session-confounded and cannot be de-confounded by
#     filtering: every toward_sensor/facing_back trial came from capture
#     folders 6-7, every comparable forward trial from folders 1-5, zero
#     overlap. Already flagged in the notes as "a real limitation of Table
#     B's phi axis specifically."
#
# Distance (Table A) is still real measured data -- that axis was never in
# question. This replaces only the ANGULAR shape, on the same principle
# Raja approved for range: use the physics, anchored to real measurements.

# theta: angular loss, as a broad shoulder -- flat-ish out to ~45deg, then
# falling off hard toward 90deg.
#
# Deliberately NOT the raw two-way antenna pattern. That was tried (it would
# charge ~18dB at 45deg) and it contradicts the only robust thing the angle
# captures actually show. It also trained terribly: with Q ~0 outside a few
# degrees of boresight there was no gradient to climb, and the policy went
# back to sprinting away from the person.
#
# What the data supports, and ONLY this: within +-45deg everything is
# indistinguishable; by +-90deg it collapses. Per-condition means look like
# they say more than that, but they don't -- n=4 per cell with std 11-14
# (e.g. theta=45/phi=45 runs 0.7, 1.4, 12.2, 30.4), and the conditions are
# perfectly session-confounded (every phi=0 trial is from capture folders
# 6-7, every phi=theta trial from folders 1-4, zero overlap). So the cell
# means are not fit targets. Two parameters, one empirical fact, no tuning
# against 34 noisy trials.
THETA_MAX_LOSS_DB = 40.0   # loss at 90deg, where the captures do collapse
THETA_LOSS_EXPONENT = 4.0  # how flat the shoulder stays before it bends

# phi: the radar measures RADIAL motion, so what matters is how much of the
# chest's in/out displacement projects onto the line of sight -- cos(phi).
# Power goes as displacement squared, hence cos^2. Peaks at phi=0 (chest to
# sensor) AND phi=180 (back to sensor) -- both present a moving surface --
# and vanishes at +-90, the side-on dead zone. Note this attenuates the
# SIGNAL (the heartbeat modulation), not just total returned power: side-on,
# the chest's motion is tangential, so the modulation carrying the heartbeat
# is absent no matter how strong the bulk reflection is.
ASPECT_FLOOR = 1e-3    # keeps cos^2 -> 0 from becoming -inf dB at exactly +-90

# Reference point for the whole SNR scale: the closest real measurement in
# Table A, at boresight with the subject facing the sensor. band_snr is a
# power ratio (peak/median), so 10*log10 is the right conversion to dB.
SNR_REF_DISTANCE = 0.7
SNR_REF_DB = 10.0 * np.log10(_TABLE_A[SNR_REF_DISTANCE])   # ~14.0 dB
Q_SCALE = _TABLE_A[SNR_REF_DISTANCE]   # keeps Q in the same units as the old tables

# The saturating detection curve. QUALITY_HALF_DB is where quality passes
# 50%; QUALITY_WIDTH_DB sets how sharp the transition is. Calibrated against
# the SNR data alone -- specifically, to reproduce the measured 0.7m -> 0.9m
# drop (25.25 -> 22.09, a 14% fall over 4.36 dB of range loss). That the
# system is still visibly range-sensitive across those two points is what
# fixes the knee: it says 0.7-0.9m sits on the SLOPE, not out on the plateau.
QUALITY_HALF_DB = 4.8
QUALITY_WIDTH_DB = 3.0

FOV_GATE_DEG = 60.0            # |theta| beyond this -> no trusted reading
COLLISION_PENALTY_SCALE = 200.0  # per meter crossed past the safe-zone floor

# Past the measured range there's no real data. Rather than invent a decay
# shape, the extrapolation follows the radar range equation: received power
# falls off as 1/R^4 for a monostatic radar (Pr = Pt.Gt.Ar.sigma.F^4 /
# ((4pi)^2.R^4), i.e. everything but R is fixed for a given rig/target).
# Approved by Raja (2026-09-20) as the basis for synthetic far-range data,
# with one real validation set to be collected later for the paper.
#
# Caveat worth keeping in mind: cnn_band_snr is NOT raw received power --
# it's spectral peakiness of the CNN's output waveform, so R^-4 is the
# right *shape* to borrow from physics, not a first-principles derivation
# of this particular metric. Still far better grounded than the flat
# per-meter slope this replaced.

# FOV violation is a quadratic (not linear) additive penalty, scaled large
# enough to dominate even the best possible Q once meaningfully past the
# gate. A weak linear penalty isn't enough to overcome an unexpectedly-good
# nearest-neighbor B lookup (e.g. 70deg, just past the gate, is
# geometrically closer to the table's 45deg cell than anywhere else, so B
# returns a near-best value there) -- and a *multiplicative* suppression on
# Q was tried and rejected: it would scale with Q, so out at long range
# (where the R^-4 curve has already driven Q near zero) an FOV violation
# would cost almost nothing, exactly where the robot most needs to be told
# it's facing the wrong way. An additive term keeps the violation's cost
# independent of how weak the signal already is.
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


def _theta_loss_db(theta_deg):
    """Angular loss from where the person sits relative to the sensor, in dB
    (0 at boresight, -THETA_MAX_LOSS_DB at 90deg). A shoulder, not a peak:
    the exponent keeps it nearly flat through ~45deg, then bends sharply."""
    t = abs(_angle_diff_deg(theta_deg, 0.0)) / 90.0
    return -THETA_MAX_LOSS_DB * t ** THETA_LOSS_EXPONENT


def _aspect_loss_db(phi_deg):
    """Loss from body aspect, in dB. cos^2(phi) is the fraction of the
    chest's displacement that is radial to the sensor (squared, for power);
    0dB facing or backing the sensor, falling away to the side-on dead
    zone. Floored at ASPECT_FLOOR so exactly +-90 doesn't give -inf."""
    projection = max(np.cos(np.radians(phi_deg)) ** 2, ASPECT_FLOOR)
    return 10.0 * np.log10(projection)


def link_budget_db(distance, theta_deg, phi_deg):
    """INTERNAL physics only -- not a quantity anything outside this module
    should consume. See predicted_band_snr() for the metric that crosses
    interfaces.

    Link budget for this pose, in dB -- the radar range equation with
    every term in place, anchored to the real 0.7m boresight measurement.

    Range contributes -40*log10(R/R_ref) (the equation's R^-4), theta
    contributes the angular shoulder, phi contributes the radial-motion
    projection. All losses, all additive in dB.

    SNR_REF_DB only sets the units. The policy cares about the SHAPE of this
    surface, not its absolute level -- which is what keeps the reward model-
    agnostic: nothing here is fit to a particular estimator's output."""
    range_loss = -40.0 * np.log10(distance / SNR_REF_DISTANCE)
    return (SNR_REF_DB
            + range_loss
            + _theta_loss_db(theta_deg)
            + _aspect_loss_db(phi_deg))


def quality(snr):
    """Saturating map from SNR (dB) to "can a heartbeat actually be pulled
    out of this", in [0, 1]. A logistic: flat near 1 once there's margin to
    spare (more SNR stops helping), collapsing toward 0 below threshold.

    The saturation is what keeps the reward BROAD rather than a needle --
    a linear power->quality map was tried and failed badly (the policy went
    back to sprinting away from the person, since Q was ~0 everywhere
    except within a few degrees of boresight, leaving no gradient to
    climb)."""
    return 1.0 / (1.0 + np.exp(-(snr - QUALITY_HALF_DB) / QUALITY_WIDTH_DB))


def predicted_band_snr(distance, theta_deg, phi_deg):
    """Predicted cnn_band_snr for this pose -- THE quantity this project
    optimises and the only one that should cross a module boundary.

    band_snr is the hand-crafted metric: spectral peak-to-median power of the
    estimator's output in the cardiac band. It is NOT received power, and it
    demonstrably does not scale like it -- measured 0.7m -> 0.9m it falls only
    -0.58 dB where R^-4 predicts -4.37 dB, i.e. about 7x more gently. That
    compression is expected of a peak/median ratio with a nonlinear estimator
    in the path, and it is exactly what quality() encodes: the link budget
    supplies the physics, this curve maps it onto the measured metric.

    Calibrated against both real points: returns 24.14 at 0.7m and 21.08 at
    0.9m, versus measured 25.25 and 22.09.

    NOT floored here. band_snr cannot physically read below ~1 (a flat
    spectrum has peak == median), and the radar channel in radar_env.py does
    apply that floor -- but the REWARD deliberately keeps sloping below it.
    A floored reward would go flat everywhere past ~2.1m, which is inside the
    spawn range, and a flat reward region is the exact condition that made
    earlier versions of this policy drive away instead of approaching. The
    reward is a shaping signal; only the observation claims to be a reading.
    """
    return Q_SCALE * quality(link_budget_db(distance, theta_deg, phi_deg))


def reward(distance, bearing, aspect):
    """
    Score how good this pose is for sensing: Q = Q_SCALE x quality(snr_db),
    minus penalties for crossing the collision safe-zone, the FOV gate, or
    phi's side dead-zone (phi near +-90). Higher = better. theta = bearing,
    phi = aspect, both in radians.

    Nothing clamps anywhere -- Q rolls off smoothly along the SNR curve in
    every direction, and instead of a hard FOV cutoff a continuous penalty
    is subtracted for how far outside the gate theta is. Same soft-penalty
    treatment for phi's dead-zone (see PHI_DEADZONE_SCALE).

    The additive penalties are deliberately kept alongside the multiplicative
    Q: where quality saturates toward 0 (deep in the dead zone, far past the
    FOV gate) the multiplicative term stops carrying any gradient at all, and
    the additive terms are what still point the way out.
    """
    theta_deg = np.degrees(bearing)
    phi_deg = np.degrees(aspect)
    q = predicted_band_snr(distance, theta_deg, phi_deg)

    too_close = max(0.0, MIN_CENTER_DISTANCE - distance)
    collision_penalty = COLLISION_PENALTY_SCALE * too_close

    too_wide = max(0.0, abs(bearing) - np.radians(FOV_GATE_DEG))
    fov_penalty = min(FOV_PENALTY_SCALE * too_wide ** 2, FOV_PENALTY_CAP)

    phi_deadzone_penalty = PHI_DEADZONE_SCALE * np.sin(aspect) ** 2

    return q - collision_penalty - fov_penalty - phi_deadzone_penalty


ANGLE_TRIAL_DISTANCE = 0.6   # the angle captures were all at ~0.6m


def check_against_error(csv_path="mmResp/angle_accuracy_all_with_phi.csv"):
    """HELD-OUT VALIDATION ONLY -- never used by the model.

    Ground-truth heart-rate error is deliberately kept out of the reward
    (the whole point of an SNR-based reward is that it stays agnostic to
    which estimator is running). This function exists purely so we can LOOK
    at whether the SNR model ranks poses the same way real error does.

    Prints measured abs_cnn_error against predicted quality per condition,
    and their correlation. A strongly negative correlation is the good
    outcome: high predicted quality should mean low real error.

    Note on filtering: trials are NOT filtered by `is_outlier`. That flag is
    defined as consistency_std > threshold, so filtering on it and then
    correlating against error is circular -- a trap already documented in
    the project notes. Blind-spot (+-90) trials are reported separately
    instead, which is the "fully_clean" convention those notes prescribe.
    """
    import csv
    import os

    if not os.path.exists(csv_path):
        print(f"(no CSV at {csv_path} -- skipping error check)")
        return

    def phi_for(facing, angle_deg):
        if facing == "toward_sensor":
            return 0.0
        if facing == "facing_back":
            return 180.0
        return float(angle_deg)      # "forward": aspect equals the sensor angle

    rows = []
    for r in csv.DictReader(open(csv_path)):
        theta = float(r["angle_deg"])
        phi = phi_for(r["facing"], r["angle_deg"])
        rows.append((
            theta, phi, float(r["abs_cnn_error"]),
            predicted_band_snr(ANGLE_TRIAL_DISTANCE, theta, phi),
        ))

    print(f"{'theta':>7} {'phi':>7} {'n':>4} {'mean_err':>10} {'pred_qual':>10}")
    conditions = sorted({(t, p) for t, p, _, _ in rows})
    for t, p in conditions:
        group = [r for r in rows if r[0] == t and r[1] == p]
        mean_err = sum(g[2] for g in group) / len(group)
        print(f"{t:7.0f} {p:7.0f} {len(group):4d} {mean_err:10.2f} {group[0][3]:10.4f}")

    def corr(pairs):
        n = len(pairs)
        if n < 3:
            return float("nan")
        mx = sum(a for a, _ in pairs) / n
        my = sum(b for _, b in pairs) / n
        cov = sum((a - mx) * (b - my) for a, b in pairs)
        vx = sum((a - mx) ** 2 for a, _ in pairs) ** 0.5
        vy = sum((b - my) ** 2 for _, b in pairs) ** 0.5
        return cov / (vx * vy) if vx and vy else float("nan")

    allp = [(r[3], r[2]) for r in rows]
    within = [(r[3], r[2]) for r in rows if abs(r[0]) < 90]
    print(f"\ncorr(predicted quality, abs error), all {len(allp)} trials : {corr(allp):+.3f}")
    print(f"corr(predicted quality, abs error), within-FOV {len(within)}  : {corr(within):+.3f}")
    print("(negative is the good direction: higher predicted quality -> lower real error)")


if __name__ == "__main__":
    # robot at origin facing +x, person 0.8m ahead, chest turned toward the robot
    d, b = distance_and_angle(0, 0, 0, 0.8, 0)
    a = aspect_angle(0, 0, 0.8, 0, np.pi)
    print("distance:", d, "bearing:", b, "aspect:", a, "reward:", reward(d, b, a))
