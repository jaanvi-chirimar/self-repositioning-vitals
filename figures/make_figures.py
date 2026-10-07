"""Paper figures, generated from saved evaluation records.

Run eval/run_eval.py first (it writes eval/results/*.json), then:

    python figures/make_figures.py

Writes vector PDFs into figures/. Everything is sized for one ACM sigconf
column (3.33in) and checked to stay legible in greyscale: series differ by
line style and marker as well as colour, and every series is labelled
directly rather than only by hue.
"""
import json
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle, Wedge

import geometry as g

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "eval", "results")

COL = 3.33                       # ACM sigconf column width, inches
POLICY = "#2a78d6"               # validated categorical palette, in fixed order
RULES = "#eb6834"
ORACLE = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#52514e"
FAINT = "#d8d7d2"

plt.rcParams.update({
    "font.family": "serif", "font.size": 6.5, "axes.labelsize": 6.5,
    "axes.titlesize": 6.5, "legend.fontsize": 6, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "axes.linewidth": 0.6, "lines.linewidth": 1.3,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": MUTED, "figure.dpi": 200, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01, "pdf.fonttype": 42,
})


def load(name):
    with open(os.path.join(RESULTS, f"{name}.json")) as f:
        return json.load(f)


def tidy(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=FAINT, linewidth=0.5)
        ax.set_axisbelow(True)


# --------------------------------------------------------------------------
# 1. geometry + control architecture
# --------------------------------------------------------------------------
def fig_geometry_architecture():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(COL * 2.06, 1.95),
                                 gridspec_kw={"width_ratios": [1.0, 1.35]})

    # (a) geometry -----------------------------------------------------------
    px, py, facing = 0.0, 0.0, 90.0
    ax.add_patch(Wedge((px, py), 1.5, facing - 45, facing + 45, color=RULES,
                       alpha=0.13, lw=0))
    for side in (0, 180):
        ax.add_patch(Wedge((px, py), 1.5, facing + side + 65, facing + side + 115,
                           color=MUTED, alpha=0.11, lw=0))
    ax.text(0.0, 1.62, "chest sector", ha="center", color=RULES, fontsize=6.5)
    ax.text(-1.30, 0.0, "dead\nzone", ha="center", va="center", color=MUTED, fontsize=6.5)
    ax.text(1.30, 0.0, "dead\nzone", ha="center", va="center", color=MUTED, fontsize=6.5)

    ax.add_patch(Circle((px, py), 0.16, facecolor="white", edgecolor=INK, lw=1.0,
                        zorder=6))
    ax.add_patch(FancyArrowPatch((px, py), (px, py + 0.50), color=INK, lw=1.0,
                                 arrowstyle="-|>", mutation_scale=8, zorder=6))
    ax.text(0.10, 0.52, r"$(x_p, y_p)$", fontsize=6.5, color=INK)
    ax.text(-0.52, 0.40, r"$\psi_p$", fontsize=7.5, color=INK)

    rx, ry, ryaw = 0.80, -0.80, 118.0
    ax.add_patch(Rectangle((rx - 0.12, ry - 0.12), 0.24, 0.24, facecolor=POLICY,
                           edgecolor="none", zorder=6))
    ax.add_patch(FancyArrowPatch(
        (rx, ry), (rx + 0.42 * np.cos(np.radians(ryaw)), ry + 0.42 * np.sin(np.radians(ryaw))),
        color=POLICY, lw=1.0, arrowstyle="-|>", mutation_scale=8, zorder=6))
    ax.text(rx + 0.14, ry - 0.20, r"$(x_r, y_r)$", fontsize=6.5, color=POLICY)
    ax.text(rx - 0.46, ry + 0.02, r"$\psi_r$", fontsize=7.5, color=POLICY)

    ax.plot([rx, px], [ry, py], color=INK, lw=0.7, ls=(0, (4, 2)), zorder=4)
    ax.text(0.30, -0.52, r"$d$", fontsize=8, color=INK)
    los = np.degrees(np.arctan2(py - ry, px - rx))
    ax.add_patch(Wedge((rx, ry), 0.34, min(los, ryaw), max(los, ryaw), facecolor="none",
                       edgecolor=POLICY, lw=0.8, zorder=5))
    ax.text(rx - 0.30, ry + 0.26, r"$\theta$", fontsize=8, color=POLICY)
    aspect = np.degrees(np.arctan2(ry - py, rx - px))
    ax.add_patch(Wedge((px, py), 0.62, aspect, facing, facecolor="none",
                       edgecolor=RULES, lw=0.8, zorder=5))
    ax.text(0.56, 0.20, r"$\phi$", fontsize=8, color=RULES)

    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.35, 1.80)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) sensing geometry", loc="left", color=INK)

    # (b) architecture -------------------------------------------------------
    bx.set_xlim(0, 10.6); bx.set_ylim(0, 5.6); bx.axis("off")
    bx.set_title("(b) control architecture", loc="left", color=INK)

    def box(x, y, w, h, label, sub, color, fill):
        bx.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=color, lw=0.9))
        bx.text(x + w / 2, y + h * 0.63, label, ha="center", va="center", fontsize=6.8,
                color=INK)
        bx.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center", fontsize=5.8,
                color=MUTED)

    def arrow(x0, y0, x1, y1, color=INK, ls="-"):
        bx.add_patch(FancyArrowPatch((x0, y0), (x1, y1), color=color, lw=0.8,
                                     arrowstyle="-|>", mutation_scale=7, ls=ls))

    #   sensors -> policy -> shield -> hold -> wheels, with a feedback loop
    box(0.1, 3.5, 2.3, 1.2, "sensors", "lidar + radar", MUTED, "#f4f3f0")
    box(3.6, 3.5, 2.7, 1.2, "policy (PPO)", "proposes velocity", POLICY, "#eaf1fb")
    box(7.5, 3.5, 3.0, 1.2, "shield", "not chest-on?", RULES, "#fdeee7")
    box(7.5, 1.3, 3.0, 1.2, "hold", "pose good?", RULES, "#fdeee7")
    box(3.6, 1.3, 2.7, 1.2, "wheels", "rate-limited", MUTED, "#f4f3f0")

    arrow(2.4, 4.1, 3.6, 4.1)
    bx.text(3.0, 4.32, "obs", fontsize=5.8, color=MUTED, ha="center")
    arrow(6.3, 4.1, 7.5, 4.1)
    bx.text(6.9, 4.32, "action", fontsize=5.8, color=MUTED, ha="center")
    arrow(9.0, 3.5, 9.0, 2.5, color=RULES)
    bx.text(9.15, 3.0, "override", fontsize=5.8, color=RULES, ha="left", va="center")
    arrow(7.5, 1.9, 6.3, 1.9)
    arrow(3.6, 1.9, 1.25, 1.9, color=MUTED, ls=(0, (3, 2)))
    arrow(1.25, 1.9, 1.25, 3.5, color=MUTED, ls=(0, (3, 2)))
    bx.text(2.45, 1.45, "robot moves", fontsize=5.8, color=MUTED, ha="center")

    fig.savefig(os.path.join(HERE, "geometry_architecture.pdf"))
    plt.close(fig)


def fig_geometry():
    """The sensing geometry alone, at single-column width: the symbols the
    formulation defines, on one picture."""
    fig, ax = plt.subplots(figsize=(COL, 1.75))
    px, py, facing = 0.0, 0.0, 90.0

    # sectors
    ax.add_patch(Wedge((px, py), 1.02, facing - 45, facing + 45, color=RULES, alpha=0.13, lw=0))
    for side in (0, 180):
        ax.add_patch(Wedge((px, py), 2.80, facing + side + 70, facing + side + 110,
                           color=MUTED, alpha=0.11, lw=0))
    ax.text(0.0, 0.88, "chest sector", ha="center", color=RULES, fontsize=6)
    ax.text(-2.05, 0.14, "dead\nzone", ha="center", va="center", color=MUTED, fontsize=6)
    ax.text(2.10, 0.14, "dead\nzone", ha="center", va="center", color=MUTED, fontsize=6)

    # person, facing up
    ax.add_patch(Circle((px, py), 0.15, facecolor="white", edgecolor=INK, lw=1.0, zorder=6))
    ax.add_patch(FancyArrowPatch((px, py), (px, py + 0.55), color=INK, lw=1.0,
                                 arrowstyle="-|>", mutation_scale=8, zorder=6))
    ax.text(-0.22, 0.02, r"$(x_p, y_p)$", fontsize=6, color=INK, ha="right", va="center")
    ax.text(-0.10, 0.52, r"$\psi_p$", fontsize=7.5, color=INK, ha="right", va="center")

    # robot, out in the chest sector, aimed 30deg off the person
    bearing_from_person = -22.0
    r_dist = 1.22
    rx = px + r_dist * np.cos(np.radians(bearing_from_person))
    ry = py + r_dist * np.sin(np.radians(bearing_from_person))
    los = np.degrees(np.arctan2(py - ry, px - rx))        # robot -> person
    ryaw = los + 38.0
    ax.add_patch(Rectangle((rx - 0.12, ry - 0.12), 0.24, 0.24, facecolor=POLICY,
                           edgecolor="none", zorder=6))
    tip = (rx + 0.50 * np.cos(np.radians(ryaw)), ry + 0.50 * np.sin(np.radians(ryaw)))
    ax.add_patch(FancyArrowPatch((rx, ry), tip, color=POLICY, lw=1.0, arrowstyle="-|>",
                                 mutation_scale=8, zorder=6))
    ax.text(rx + 0.18, ry - 0.16, r"$(x_r, y_r)$", fontsize=6, color=POLICY, va="center")
    ax.text(tip[0] - 0.04, tip[1] - 0.20, r"$\psi_r$", fontsize=7.5, color=POLICY,
            ha="center", va="center")

    # d along the line of sight, label offset off the line
    ax.plot([rx, px], [ry, py], color=INK, lw=0.7, ls=(0, (4, 2)), zorder=4)
    mid = ((rx + px) / 2, (ry + py) / 2)
    # offset perpendicular to the sight line, on the side away from the phi arc
    perp = np.radians(np.degrees(np.arctan2(ry - py, rx - px)) + 90.0)
    ax.text(mid[0] + 0.26 * np.cos(perp), mid[1] + 0.26 * np.sin(perp), r"$d$",
            fontsize=7.5, color=INK, ha="center", va="center")

    # theta between the robot's heading and the line of sight
    ax.add_patch(Arc((rx, ry), 0.72, 0.72, theta1=min(los, ryaw), theta2=max(los, ryaw),
                     edgecolor=POLICY, lw=0.8, zorder=5))
    tlab = np.radians((los + ryaw) / 2)
    ax.text(rx + 0.60 * np.cos(tlab), ry + 0.60 * np.sin(tlab), r"$\theta$",
            fontsize=7.5, color=POLICY, ha="center", va="center")

    # phi between the person's facing and the direction to the robot
    ax.add_patch(Arc((px, py), 1.02, 1.02, theta1=bearing_from_person, theta2=facing,
                     edgecolor=RULES, lw=0.8, zorder=5))
    plab = np.radians((bearing_from_person + facing) / 2)
    ax.text(0.68 * np.cos(plab), 0.68 * np.sin(plab), r"$\phi$", fontsize=7.5,
            color=RULES, ha="center", va="center")

    ax.set_xlim(-2.80, 2.95); ax.set_ylim(-1.00, 1.05)
    ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(os.path.join(HERE, "geometry.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 2. reward model against the measurements it is anchored to
# --------------------------------------------------------------------------
def fig_reward_model():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(COL, 1.28))

    d = np.linspace(0.55, 3.0, 300)
    ax.plot(d, [g.predicted_band_snr(x, 0, 0) for x in d], color=POLICY,
            label="model")
    ax.axhline(g.CARDIAC_NOISE_FLOOR, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.text(2.0, g.CARDIAC_NOISE_FLOOR - 0.45, "noise floor", fontsize=6, color=MUTED,
            ha="center", va="top")
    ax.plot([0.7, 1.2], [13.088, 10.267], "o", ms=4.5, color=INK, zorder=5)
    ax.plot([0.9], [12.543], "^", ms=5, mfc="white", mec=RULES, mew=1.1, zorder=5)
    ax.annotate("anchors", (0.7, 13.088), (0.95, 14.3), fontsize=6, color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    ax.annotate("held out", (0.9, 12.543), (1.45, 12.9), fontsize=6, color=RULES,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=RULES))
    ax.set_xlabel("range (m)"); ax.set_ylabel("cardiac-band SNR")
    ax.set_ylim(4, 15.5)
    tidy(ax)
    ax.set_title("(a) range", loc="left", color=INK)

    phi = np.linspace(-180, 180, 400)
    for dist, style, lab in ((0.7, "-", "0.7 m"), (1.2, (0, (4, 2)), "1.2 m"),
                             (2.0, (0, (1.5, 1.5)), "2.0 m")):
        bx.plot(phi, [g.predicted_band_snr(dist, 0, p) for p in phi],
                color=POLICY if dist == 0.7 else (RULES if dist == 1.2 else MUTED),
                ls=style, label=lab)
    for lo, hi in ((-115, -65), (65, 115)):
        bx.axvspan(lo, hi, color=MUTED, alpha=0.10, lw=0)
    bx.text(-90, 14.6, "dead\nzone", ha="center", va="top", fontsize=6, color=MUTED)
    # direct labels beat a legend box: identity survives greyscale, and the box
    # was landing on the 2.0 m curve.
    for dist, col in ((0.7, POLICY), (1.2, RULES), (2.0, MUTED)):
        bx.text(0, g.predicted_band_snr(dist, 0, 0) + 0.50, f"{dist} m",
                ha="center", va="bottom", fontsize=6, color=col)
    bx.set_xlabel(r"aspect $\phi$ (deg)")
    bx.set_xticks([-180, -90, 0, 90, 180])
    bx.set_ylim(4, 15.5)
    tidy(bx)
    bx.set_title("(b) body aspect", loc="left", color=INK)

    fig.savefig(os.path.join(HERE, "reward_model.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. trajectories
# --------------------------------------------------------------------------
def _draw_obstacles(ax, obstacles):
    for o in obstacles:
        if o[0] == "circle":
            ax.add_patch(Circle((o[1], o[2]), o[3], facecolor="#e7e6e1",
                                edgecolor=MUTED, lw=0.5))
        else:
            _, cx, cy, hw, hh, yaw = o
            r = Rectangle((-hw, -hh), 2 * hw, 2 * hh, facecolor="#e7e6e1",
                          edgecolor=MUTED, lw=0.5)
            tr = (matplotlib.transforms.Affine2D().rotate(yaw).translate(cx, cy)
                  + ax.transData)
            r.set_transform(tr)
            ax.add_patch(r)


def _draw_person(ax, person):
    px, py, phi = person
    ax.add_patch(Circle((px, py), 0.15, facecolor="white", edgecolor=INK, lw=0.9,
                        zorder=6))
    ax.add_patch(FancyArrowPatch((px, py), (px + 0.36 * np.cos(phi), py + 0.36 * np.sin(phi)),
                                 color=INK, lw=0.9, arrowstyle="-|>", mutation_scale=7,
                                 zorder=6))


def _draw_path(ax, path, color, label, ls="-"):
    p = np.array(path)
    ax.plot(p[:, 0], p[:, 1], color=color, lw=1.3, ls=ls, label=label, zorder=5)
    ax.plot(p[0, 0], p[0, 1], "s", ms=4, color=color, mec="white", mew=0.6, zorder=6)
    ax.plot(p[-1, 0], p[-1, 1], "o", ms=5.5, color=color, mec="white", mew=0.8, zorder=7)


def fig_trajectories():
    alone = {e["seed"]: e for e in load("empty_none")["episodes"] if "path" in e}
    rules = {e["seed"]: e for e in load("empty_shield-hold")["episodes"] if "path" in e}
    room = [e for e in load("bedroom_shield-hold-search")["episodes"] if "path" in e]

    # pick the clearest contrast: same start, policy alone ends side-on or
    # behind while the rules bring the robot to the chest
    shared = [k for k in alone if k in rules]
    seed = max(shared, key=lambda k: alone[k]["aspect_deg"] - rules[k]["aspect_deg"])
    ea, er = alone[seed], rules[seed]

    # and a room episode where the shield actually had to route around the bed
    eb = max(room, key=lambda e: (e["shield_frac"], e["occluded_frac"]))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(COL, 1.6))

    _draw_person(ax, ea["person"])
    _draw_path(ax, ea["path"], POLICY, "policy alone", ls=(0, (4, 2)))
    _draw_path(ax, er["path"], RULES, "policy + rules")
    p = np.array(ea["path"])
    ax.annotate("start", (p[0, 0], p[0, 1]), (p[0, 0] + 0.12, p[0, 1] - 0.30),
                fontsize=6, color=MUTED,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
    # Both runs hug the left of the panel, so the labels live in a gutter on the
    # right. The gutter is reserved as an invisible DATA point: with equal aspect
    # a manual set_xlim is refitted away, and axes-fraction text then lands back
    # on top of the paths.
    pts = np.vstack([np.array(ea["path"]), np.array(er["path"])])
    gx0, gx1 = pts[:, 0].min(), pts[:, 0].max()
    gy0, gy1 = pts[:, 1].min(), pts[:, 1].max()
    span = max(gx1 - gx0, gy1 - gy0)
    ax.plot([gx1 + 0.62 * span], [(gy0 + gy1) / 2], marker="o", ms=0, alpha=0)
    for ytop, color, text, ep in ((gy1, POLICY, "policy alone", ea),
                                  (gy1 - 0.38 * (gy1 - gy0), RULES, "policy + rules", er)):
        ax.text(gx1 + 0.10 * span, ytop, f"{text}\n" +
                r"$|\phi|$ = " + f"{ep['aspect_deg']:.0f}" + r"$^\circ$",
                fontsize=6, color=color, ha="left", va="top")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) open room, same start", loc="left", color=INK)

    _draw_obstacles(bx, [tuple(o) for o in eb["obstacles"]])
    _draw_person(bx, eb["person"])
    _draw_path(bx, eb["path"], RULES, None)
    bx.text(1.1, 0.85, "bed", ha="center", va="center", fontsize=6, color=MUTED)
    bx.set_aspect("equal"); bx.axis("off")
    bx.set_title("(b) held-out furnished room", loc="left", color=INK)

    fig.savefig(os.path.join(HERE, "trajectories.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. where episodes end up
# --------------------------------------------------------------------------
def fig_outcomes():
    alone = load("empty_none")["episodes"]
    rules = load("empty_shield-hold")["episodes"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(COL, 1.45),
                                 gridspec_kw={"width_ratios": [1.35, 1]})

    bins = np.arange(0, 181, 15)
    for eps, color, label, ls in ((alone, POLICY, "policy alone", (0, (4, 2))),
                                  (rules, RULES, "policy + rules", "-")):
        counts, _ = np.histogram([e["aspect_deg"] for e in eps], bins=bins)
        ax.step(bins, np.append(counts, counts[-1]), where="post", color=color,
                lw=1.3, ls=ls, label=label)
    ax.axvspan(0, 45, color=RULES, alpha=0.10, lw=0)
    ax.text(22, ax.get_ylim()[1] * 0.92, "usable", ha="center", fontsize=6, color=RULES)
    ax.set_xlabel(r"final aspect $|\phi|$ (deg)"); ax.set_ylabel("episodes")
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.legend(frameon=False, loc="upper right", handlelength=1.8, borderpad=0.1)
    tidy(ax)
    ax.set_title("(a) where episodes end", loc="left", color=INK)

    qa = np.array([e["quality"] for e in alone])
    qr = np.array([e["quality"] for e in rules])
    parts = bx.violinplot([qa, qr], showextrema=False, widths=0.7)
    for body, color in zip(parts["bodies"], (POLICY, RULES)):
        body.set_facecolor(color); body.set_alpha(0.35); body.set_edgecolor(color)
        body.set_linewidth(0.8)
    for i, (vals, color) in enumerate(((qa, POLICY), (qr, RULES)), start=1):
        bx.plot([i], [np.median(vals)], "o", ms=4, color=color, zorder=5)
        bx.text(i, np.median(vals) + 0.5, f"{np.mean(vals):.1f}", ha="center",
                fontsize=6, color=color)
    bx.axhline(g.CARDIAC_NOISE_FLOOR, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    bx.text(2.42, g.CARDIAC_NOISE_FLOOR + 0.25, "floor", fontsize=6, color=MUTED,
            ha="right")
    bx.set_xticks([1, 2]); bx.set_xticklabels(["policy", "+ rules"])
    bx.set_ylabel("cardiac-band SNR")
    tidy(bx)
    bx.set_title("(b) signal quality at the final pose", loc="left", color=INK)

    fig.savefig(os.path.join(HERE, "outcomes.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. teaser: why pose matters, and what each approach achieves
# --------------------------------------------------------------------------
def fig_teaser():
    """Three panels: a fixed sensor, a robot that only closes distance, and
    ours, which also controls which side of the person it measures from.
    The bar under each panel is the quality the reward model predicts for
    that geometry, on the same scale as everything else in the paper."""
    fig, axes = plt.subplots(1, 3, figsize=(COL * 2.0, 1.10))
    px, py, facing = 0.0, 0.0, 90.0

    def person(ax, x=px, y=py, phi=facing):
        ax.add_patch(Circle((x, y), 0.17, facecolor="white", edgecolor=INK, lw=0.9, zorder=6))
        ax.add_patch(FancyArrowPatch((x, y), (x + 0.45 * np.cos(np.radians(phi)),
                                              y + 0.45 * np.sin(np.radians(phi))),
                                     color=INK, lw=0.9, arrowstyle="-|>",
                                     mutation_scale=7, zorder=6))

    def sensor(ax, x, y, color, boxed=False):
        if boxed:   # wall-mounted
            ax.add_patch(Rectangle((x - 0.12, y - 0.3), 0.1, 0.6, facecolor=MUTED,
                                   edgecolor="none"))
        ax.add_patch(Rectangle((x - 0.13, y - 0.13), 0.26, 0.26, facecolor=color,
                               edgecolor="none", zorder=6))

    def beam(ax, x, y, tx, ty, strong):
        ax.add_patch(FancyArrowPatch((x, y), (tx, ty), color=RULES if strong else MUTED,
                                     lw=1.4 if strong else 0.8,
                                     ls="-" if strong else (0, (2, 2)),
                                     arrowstyle="-", alpha=0.95 if strong else 0.6,
                                     zorder=3))

    panels = [
        ("(a) fixed sensor", 5.5, "person faces away"),
        ("(b) robot, distance only", 8.3, "close, but side-on"),
        ("(c) ours", 11.8, "chest-on, stopped"),
    ]
    for ax, (title, q, sub) in zip(axes, panels):
        person(ax)
        if title.startswith("(a)"):
            sensor(ax, -1.25, 0.0, MUTED, boxed=True)
            beam(ax, -1.1, 0.0, -0.17, 0.0, strong=False)
        elif title.startswith("(b)"):
            sensor(ax, 0.75, -0.1, POLICY)
            beam(ax, 0.62, -0.1, 0.16, -0.04, strong=False)
        else:
            sensor(ax, 0.0, 0.85, RULES)
            beam(ax, 0.0, 0.72, 0.0, 0.2, strong=True)
        ax.text(0, -0.95, sub, ha="center", fontsize=6.8, color=MUTED)
        ax.set_xlim(-1.6, 1.45); ax.set_ylim(-1.5, 1.15)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, loc="center", fontsize=7.5, color=INK, pad=3,
                     fontweight="bold" if title.startswith("(c)") else "normal")

        # quality bar, floor to peak
        lo, hi = g.CARDIAC_NOISE_FLOOR, g.CARDIAC_PEAK
        frac = (q - lo) / (hi - lo)
        y0, h = -1.42, 0.17
        ax.add_patch(Rectangle((-1.0, y0), 2.0, h, facecolor=FAINT, edgecolor="none"))
        ax.add_patch(Rectangle((-1.0, y0), 2.0 * frac, h,
                               facecolor=RULES if frac > 0.75 else MUTED, edgecolor="none"))
        ax.text(1.05, y0 + h / 2, f"{q:.1f}", fontsize=6.8, va="center",
                color=INK if frac > 0.75 else MUTED)

    fig.subplots_adjust(top=0.88, bottom=0.10, wspace=0.45)
    fig.text(0.5, 0.005, "predicted cardiac-band SNR at the final pose "
             "(noise floor 5.3, best achievable 13.3)",
             ha="center", fontsize=6.8, color=MUTED)
    fig.savefig(os.path.join(HERE, "teaser.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_teaser()
    print("wrote teaser.pdf")
    fig_geometry()
    print("wrote geometry.pdf")
    fig_geometry_architecture()
    print("wrote geometry_architecture.pdf (full width; unused if geometry.pdf is used)")
    fig_reward_model()
    print("wrote reward_model.pdf")
    if os.path.isdir(RESULTS) and os.listdir(RESULTS):
        fig_trajectories()
        print("wrote trajectories.pdf")
        fig_outcomes()
        print("wrote outcomes.pdf")
    else:
        print("no eval/results/*.json yet -- run eval/run_eval.py for the data figures")
