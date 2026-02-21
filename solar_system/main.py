"""Entry point for the 2D solar system simulation.

Simulates Mercury, Venus, Earth, and Mars orbiting the Sun for one Earth year
and prints the final [x, y] position of each planet.
"""
from __future__ import annotations

import numpy as np

from solar_system.planets import Planet, Sun, simulate

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

YEAR = 365.25 * 24 * 3600   # one Earth year in seconds
DT   = 3600                  # time step: 1 hour

# ---------------------------------------------------------------------------
# Solar system bodies
# Real initial conditions: circular orbit approximation
#   position = [semi-major axis, 0]  (m)
#   velocity = [0, orbital speed]    (m/s)  → counter-clockwise orbit
# ---------------------------------------------------------------------------

def main() -> None:
    sun = Sun(mass=1.989e30, Position=np.array([0.0, 0.0]))

    planets = [
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

    print(f"Simulating {len(planets)} planets for 1 Earth year (dt={DT} s) …\n")

    coords = simulate(planets, duration=YEAR, dt=DT)

    # coords shape: (n_planets, n_steps, 2)
    n_steps = coords.shape[1]
    print(f"Steps recorded: {n_steps}\n")

    au = 1.496e11  # 1 AU in metres

    print(f"{'Planet':<10}  {'Final x (AU)':>14}  {'Final y (AU)':>14}  {'Radius (AU)':>12}")
    print("-" * 58)
    for i, planet in enumerate(planets):
        x, y = coords[i, -1]
        r = np.linalg.norm([x, y])
        print(f"{planet.name:<10}  {x/au:>14.4f}  {y/au:>14.4f}  {r/au:>12.4f}")


if __name__ == "__main__":
    main()
