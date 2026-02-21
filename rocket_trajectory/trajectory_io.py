"""Export utilities for rocket trajectory data."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .ascent_opt import R_EARTH

@dataclass
class TrajectoryRecord:
    time: float
    radius: float
    radial_velocity: float
    tangential_velocity: float
    mass: float


def trajectory_to_rows(t: np.ndarray, y: np.ndarray) -> Iterable[TrajectoryRecord]:
    for i in range(t.size):
        yield TrajectoryRecord(
            time=float(t[i]),
            radius=float(y[0, i]),
            radial_velocity=float(y[1, i]),
            tangential_velocity=float(y[2, i]),
            mass=float(y[3, i]),
        )


def write_csv(path: str | Path, t: np.ndarray, y: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "radius_m", "altitude_m", "v_rad_m_s", "v_tan_m_s", "mass_kg"])
        for rec in trajectory_to_rows(t, y):
            altitude = rec.radius - R_EARTH
            w.writerow([
                rec.time,
                rec.radius,
                altitude,
                rec.radial_velocity,
                rec.tangential_velocity,
                rec.mass,
            ])
    return path


def write_geojson(path: str | Path, t: np.ndarray, y: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Very naive: treat trajectory as LineString in an abstract 3D Earth-centered frame (ECEF-like without rotation).
    coords = []
    for i in range(t.size):
        # Convert spherical to Cartesian (assuming r is radial distance, and using cumulative angle as simple proxy)
        r = y[0, i]
        # We don't track actual longitude/latitude in this toy model; embed radial along z for placeholder.
        coords.append([0.0, 0.0, r - R_EARTH])
    geo = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"description": "Toy ascent trajectory"},
                "geometry": {"type": "LineString", "coordinates": coords},
            }
        ],
    }
    with path.open("w") as f:
        json.dump(geo, f)
    return path
