"""2D solar system planet trajectory calculation.

Each Planet orbits a fixed Sun using simple Euler integration:

    a = G * sun_mass / r²  (directed toward the Sun)
    v += a * dt
    pos += v * dt

Coordinate system
-----------------
- Origin: centre of the Sun.
- Axes: x pointing right, y pointing up (both in metres).

Units (SI)
----------
- Distance : metres (m)
- Mass     : kilograms (kg)
- Time     : seconds (s)
- Velocity : m/s
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

G: float = 6.674_30e-11  # gravitational constant, m^3 kg^-1 s^-2


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Sun:
    """Represents the central star (fixed at the origin).

    Parameters
    ----------
    mass : float
        Mass in kilograms.
    position : array-like of shape (2,)
        Position in metres (typically [0, 0]).
    """

    mass: float
    Position: np.ndarray


@dataclass
class Planet:
    """Represents a planet orbiting a Sun in 2-D.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. "Earth").
    mass : float
        Mass in kilograms.
    position : array-like of shape (2,)
        Initial [x, y] position in metres.
    velocity : array-like of shape (2,)
        Initial [vx, vy] velocity in m/s.
    sun : Sun
        The central body this planet orbits.
    """

    name: str
    mass: float
    Position: np.ndarray
    Velocity: np.ndarray
    sun: Sun


    def acceleration(self) -> np.ndarray:
        """Compute gravitational acceleration toward the Sun.

        Returns
        -------
        acceleration : ndarray of shape (2,)
            [ax, ay] in m/s².
        """
        Delta = self.sun.Position - self.Position
        dist = np.linalg.norm(Delta)
        return G * self.sun.mass / dist**2 * (Delta / dist)

    def step(self, dt: float) -> None:
        """Advance position and velocity by one Euler step.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        """
        acc = self.acceleration()
        self.Velocity += acc * dt
        self.Position += self.Velocity * dt


# ---------------------------------------------------------------------------
# High-level simulation interface
# ---------------------------------------------------------------------------

def simulate(
    planets: List[Planet],
    duration: float,
    dt: float,
) -> np.ndarray:
    """Simulate planet trajectories using simple Euler integration.

    Each planet orbits its own Sun independently.

    Parameters
    ----------
    planets : list of Planet
        Initial conditions for every planet.
    duration : float
        Total simulation time in seconds.
    dt : float
        Time step in seconds.

    Returns
    -------
    positions : ndarray of shape (n_planets, n_steps, 2)
        [x, y] coordinates for each planet at every time step.

    Examples
    --------
    >>> from solar_system.planets import Sun, Planet, simulate
    >>> import numpy as np
    >>> sun   = Sun(1.989e30, [0.0, 0.0])
    >>> earth = Planet("Earth", 5.972e24, [1.496e11, 0.0], [0.0, 29_780.0], sun)
    >>> coords = simulate([earth], duration=365.25*24*3600, dt=3600)
    >>> coords.shape
    (1, 8766, 2)
    """
    n_steps = int(duration / dt)
    pos_history = np.empty((len(planets), n_steps, 2), dtype=float)

    for step in range(n_steps):
        for i, planet in enumerate(planets):
            pos_history[i, step] = planet.Position
            planet.step(dt)

    return pos_history
