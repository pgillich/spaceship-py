"""Real-time 2D solar system visualisation using Matplotlib animation.

Each frame advances the simulation by STEPS_PER_FRAME * DT seconds and
redraws all planet positions.  Two subplots are shown side-by-side:
  - Left  : inner solar system (out to Mars + Ceres)
  - Right : full solar system  (out to Eris)

Run with:
    python -m solar_system.visualize
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List

import matplotlib
from mypy.util import T
matplotlib.use("Qt5Agg")  # interactive backend required for plt.show() in WSL
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from solar_system.planets import Planet, Sun

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

DT              = 3600 * 6        # time step: 6 hours
STEPS_PER_FRAME = 20              # simulation steps advanced per animation frame
TRAIL_LENGTH    = 10             # number of past positions kept for trail
INTERVAL_MS     = 100             # milliseconds between frames (~5 fps)

AU = 1.496e11  # metres per AU

# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------

COLOURS = [
    "#b5b5b5",  # Mercury
    "#e8c97a",  # Venus
    "#4fa3e0",  # Earth
    "#c1440e",  # Mars
    "#a0785a",  # Ceres
    "#c88b3a",  # Jupiter
    "#e4d191",  # Saturn
    "#7de8e8",  # Uranus
    "#3f54ba",  # Neptune
    "#a0522d",  # Pluto
    "#cc88cc",  # Haumea
    "#dd9944",  # Makemake
    "#eeeeee",  # Eris
]


def build_planets(sun: Sun) -> List[Planet]:
    return [
        # Inner planets
        Planet("Mercury",  3.301e23, np.array([5.791e10, 0.0]), np.array([0.0, 47_870.0]), sun),
        Planet("Venus",    4.867e24, np.array([1.082e11, 0.0]), np.array([0.0, 35_020.0]), sun),
        Planet("Earth",    5.972e24, np.array([1.496e11, 0.0]), np.array([0.0, 29_780.0]), sun),
        Planet("Mars",     6.390e23, np.array([2.279e11, 0.0]), np.array([0.0, 24_130.0]), sun),
        # Dwarf planet (asteroid belt)
        Planet("Ceres",    9.393e20, np.array([4.139e11, 0.0]), np.array([0.0, 17_900.0]), sun),
        # Outer planets
        Planet("Jupiter",  1.898e27, np.array([7.783e11, 0.0]), np.array([0.0, 13_060.0]), sun),
        Planet("Saturn",   5.683e26, np.array([1.427e12, 0.0]), np.array([0.0,  9_640.0]), sun),
        Planet("Uranus",   8.681e25, np.array([2.871e12, 0.0]), np.array([0.0,  6_800.0]), sun),
        Planet("Neptune",  1.024e26, np.array([4.495e12, 0.0]), np.array([0.0,  5_430.0]), sun),
        # Dwarf planets (trans-Neptunian)
        Planet("Pluto",    1.303e22, np.array([5.906e12, 0.0]), np.array([0.0,  4_740.0]), sun),
        Planet("Haumea",   4.006e21, np.array([6.452e12, 0.0]), np.array([0.0,  4_530.0]), sun),
        Planet("Makemake", 3.100e21, np.array([6.796e12, 0.0]), np.array([0.0,  4_420.0]), sun),
        Planet("Eris",     1.660e22, np.array([1.015e13, 0.0]), np.array([0.0,  3_616.0]), sun),
    ]


def main() -> None:
    sun = Sun(mass=1.989e30, Position=np.array([0.0, 0.0]))
    planets = build_planets(sun)
    colours = COLOURS[: len(planets)]

    # Per-planet trail buffers
    trails: List[Deque[np.ndarray]] = [
        deque(maxlen=TRAIL_LENGTH) for _ in planets
    ]

    # ---------------------------------------------------------------------------
    # Figure layout: two subplots
    # ---------------------------------------------------------------------------
    fig, (ax_inner, ax_outer) = plt.subplots(1, 2, figsize=(14, 7))
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.92, wspace=0.25)
    fig.patch.set_facecolor("#0d0d1a")
    for ax in (ax_inner, ax_outer):
        ax.set_facecolor("#0d0d1a")
        ax.set_aspect("equal")
        ax.tick_params(colors="#666666")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    # Axis limits in AU
    inner_lim = 3.5   # AU  – shows up to Ceres
    outer_lim = 70.0  # AU  – shows up to Eris

    for ax, lim in ((ax_inner, inner_lim), (ax_outer, outer_lim)):
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("x (AU)", color="#888888")
        ax.set_ylabel("y (AU)", color="#888888")
        # Sun marker
        ax.plot(0, 0, "o", color="#ffe066", markersize=10 if lim < 10 else 6)

    ax_inner.set_title("Inner solar system", color="#cccccc")
    ax_outer.set_title("Full solar system", color="#cccccc")

    # ---------------------------------------------------------------------------
    # Create artists for each planet (dot + trail line + label) in both axes
    # ---------------------------------------------------------------------------
    inner_dots, outer_dots = [], []
    inner_trails, outer_trails = [], []
    inner_labels, outer_labels = [], []

    for planet, colour in zip(planets, colours):
        ms = 6  # marker size

        d_i, = ax_inner.plot([], [], "o", color=colour, markersize=ms, zorder=3)
        d_o, = ax_outer.plot([], [], "o", color=colour, markersize=ms, zorder=3)
        t_i, = ax_inner.plot([], [], "-", color=colour, linewidth=0.6, alpha=0.5, zorder=2)
        t_o, = ax_outer.plot([], [], "-", color=colour, linewidth=0.6, alpha=0.5, zorder=2)
        l_i  = ax_inner.text(0, 0, planet.name, color=colour, fontsize=6, zorder=4)
        l_o  = ax_outer.text(0, 0, planet.name, color=colour, fontsize=6, zorder=4)

        inner_dots.append(d_i);     outer_dots.append(d_o)
        inner_trails.append(t_i);   outer_trails.append(t_o)
        inner_labels.append(l_i);   outer_labels.append(l_o)

    # Elapsed-time text (placed in the outer subplot title, avoids blit=True crash)
    elapsed = [0.0]  # mutable container so the closure can update it

    # ---------------------------------------------------------------------------
    # Animation update function
    # ---------------------------------------------------------------------------
    def update(_frame: int):
        # Advance simulation
        for _ in range(STEPS_PER_FRAME):
            for planet in planets:
                planet.step(DT)
        elapsed[0] += STEPS_PER_FRAME * DT

        years = elapsed[0] / (365.25 * 24 * 3600)
        ax_outer.set_title(f"Full solar system — {years:.2f} years", color="#cccccc")

        artists = [ax_outer.title]  # title artist must be included for blit=True
        for i, planet in enumerate(planets):
            trails[i].append(planet.Position.copy())
            x_au = planet.Position[0] / AU
            y_au = planet.Position[1] / AU

            trail_x = [p[0] / AU for p in trails[i]]
            trail_y = [p[1] / AU for p in trails[i]]

            for dot, trail, label in (
                (inner_dots[i], inner_trails[i], inner_labels[i]),
                (outer_dots[i], outer_trails[i], outer_labels[i]),
            ):
                dot.set_data([x_au], [y_au])
                trail.set_data(trail_x, trail_y)
                label.set_position((x_au + 0.05, y_au + 0.05))
                artists += [dot, trail, label]

        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        interval=INTERVAL_MS,
        blit=True,
        cache_frame_data=False,
    )

    # Stop the Qt timer cleanly before Qt tears down the event loop,
    # preventing the "TimerQT.__del__ NoneType not callable" error on exit.
    fig.canvas.mpl_connect("close_event", lambda _: ani.event_source.stop())

    plt.show()


if __name__ == "__main__":
    main()
