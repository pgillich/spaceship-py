"""Simplified chemical rocket ascent trajectory optimization.

This module sets up a toy optimization problem: choose pitch profile parameters
for a single-stage rocket to reach a target circular orbit altitude with minimal
propellant (maximize final mass) subject to simple dynamics.

NOTE: This is a pedagogical example, not flight-grade code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6371e3  # m
G0 = 9.80665  # m/s^2

@dataclass
class RocketParams:
    thrust: float  # N (assumed constant)
    isp: float  # s
    dry_mass: float  # kg
    prop_mass: float  # kg
    Cd: float = 0.0  # drag coeff (ignored in toy model)

@dataclass
class TargetOrbit:
    altitude: float  # m above Earth's surface

@dataclass
class OptimizationResult:
    success: bool
    message: str
    final_mass: float
    delta_v: float
    time_of_flight: float
    pitch_params: np.ndarray
    final_altitude: float


def pitch_profile(t: float, tf: float, params: np.ndarray) -> float:
    """Return pitch angle (radians from local horizontal) as a smooth function.

    We model pitch as a cubic Hermite from vertical (90 deg) to near horizontal.
    params = [theta_final_deg, shape_factor]
    """
    theta_final = math.radians(params[0])
    shape = params[1]
    s = t / tf
    # Hermite blend h(s) = 2s^3 - 3s^2 + 1 (goes 1->0). Modify with shape to control curvature.
    h = (2 * s**3 - 3 * s**2 + 1) ** shape
    return h * math.pi/2 + (1 - h) * theta_final


def dynamics(t: float, y: np.ndarray, rp: RocketParams, tf: float, pitch_params: np.ndarray) -> np.ndarray:
    r, v_r, v_t, m = y  # radial distance from Earth's center, radial vel, tangential vel, mass
    altitude = r - R_EARTH
    # Simple gravity only
    g = MU_EARTH / r**2

    # Thrust acceleration
    if m <= rp.dry_mass:
        e_T = 0.0
    else:
        e_T = rp.thrust

    # Pitch angle from horizontal: gamma
    gamma = pitch_profile(t, tf, pitch_params)
    # Decompose thrust along radial & tangential directions (horizontal tangent plane approx)
    a_thrust = e_T / m
    a_r = a_thrust * math.sin(gamma) - g
    a_t = a_thrust * math.cos(gamma)

    # Mass flow
    mdot = -e_T / (rp.isp * G0) if e_T > 0 else 0.0

    return np.array([v_r, a_r, a_t / (1 + altitude / R_EARTH), mdot])


def simulate_ascent(rp: RocketParams, pitch_params: np.ndarray, tf: float, n_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    # Initial state: on surface, vertical velocity 0, tiny horizontal
    r0 = R_EARTH
    v_r0 = 0.0
    v_t0 = 10.0  # small eastward component to avoid singularity
    m0 = rp.dry_mass + rp.prop_mass
    y0 = np.array([r0, v_r0, v_t0, m0])

    sol = solve_ivp(
        lambda t, y: dynamics(t, y, rp, tf, pitch_params),
        (0.0, tf), y0, rtol=1e-6, atol=1e-8, dense_output=True
    )
    if (not sol.success) or (sol.sol is None):
        raise RuntimeError(
            f"Ascent integration failed: success={sol.success}, status={sol.status}, message={sol.message}"
        )
    t = np.linspace(0.0, tf, n_points)
    y = sol.sol(t)
    return t, y


def objective(pitch_params: np.ndarray, rp: RocketParams, target: TargetOrbit) -> float:
    # Decision vars: [theta_final_deg, shape]
    theta_final_deg = pitch_params[0]
    shape = pitch_params[1]
    # Guard domain
    if not (0 < theta_final_deg < 10) or not (0.3 < shape < 5):
        return 1e6

    tf = 300.0  # fixed ascent duration seconds (toy)
    t, y = simulate_ascent(rp, pitch_params, tf)
    r = y[0]
    v_r = y[1]
    v_t = y[2]
    m = y[3]

    r_final = r[-1]
    v_r_final = v_r[-1]
    v_t_final = v_t[-1]
    m_final = m[-1]

    altitude = r_final - R_EARTH
    g_final = MU_EARTH / r_final**2
    v_circ = math.sqrt(MU_EARTH / r_final)

    # Penalties
    alt_err = abs(altitude - target.altitude) / 100.0  # scale
    vert_vel_pen = abs(v_r_final) / 1.0
    tangential_speed_pen = abs(v_t_final - v_circ) / 10.0

    # Objective: minimize negative final mass + penalties
    return -(m_final - rp.dry_mass) + alt_err + vert_vel_pen + tangential_speed_pen


def optimize_pitch(rp: RocketParams, target: TargetOrbit) -> OptimizationResult:
    x0 = np.array([5.0, 1.5])  # initial guess final pitch ~5 deg, shape factor

    res = minimize(
        lambda x: objective(x, rp, target),
        x0,
        method="Nelder-Mead",
        options={"xatol": 1e-2, "fatol": 1e-2, "maxfev": 200}
    )

    tf = 300.0
    t, y = simulate_ascent(rp, res.x, tf)
    v_r = y[1][-1]
    v_t = y[2][-1]
    r_final = y[0][-1]
    m_final = y[3][-1]
    v_circ = math.sqrt(MU_EARTH / r_final)

    # Rough delta-v used (Tsiolkovsky) ignoring gravity losses nuance
    m0 = rp.dry_mass + rp.prop_mass
    if m_final > 0:
        delta_v = rp.isp * G0 * math.log(m0 / m_final)
    else:
        delta_v = float('nan')

    return OptimizationResult(
        success=res.success,
        message=res.message,
        final_mass=m_final,
        delta_v=delta_v,
        time_of_flight=tf,
        pitch_params=res.x,
        final_altitude=r_final - R_EARTH,
    )


if __name__ == "__main__":
    rp = RocketParams(thrust=2.5e6, isp=300.0, dry_mass=30000.0, prop_mass=200000.0)
    target = TargetOrbit(altitude=200e3)
    result = optimize_pitch(rp, target)
    print(result)
