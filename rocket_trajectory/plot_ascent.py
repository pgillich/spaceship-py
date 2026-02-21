"""Demo script: optimize ascent and generate Plotly figure + optional exports.

Usage (after installing requirements):
    python -m rocket_trajectory.plot_ascent --show --csv out/ascent.csv --geojson out/ascent.geojson
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rocket_trajectory.ascent_opt import RocketParams, TargetOrbit, optimize_pitch, simulate_ascent
from rocket_trajectory.visualization import altitude_time_figure
from rocket_trajectory.trajectory_io import write_csv, write_geojson


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimize toy ascent and visualize")
    p.add_argument("--altitude", type=float, default=200e3, help="Target orbit altitude (m)")
    p.add_argument("--csv", type=Path, help="Path to write CSV export", default=None)
    p.add_argument("--geojson", type=Path, help="Path to write GeoJSON export", default=None)
    p.add_argument("--show", action="store_true", help="Show interactive Plotly figure in browser")
    return p.parse_args()


def main():
    args = parse_args()
    rp = RocketParams(thrust=2.5e6, isp=300.0, dry_mass=30000.0, prop_mass=200000.0)
    target = TargetOrbit(altitude=args.altitude)
    result = optimize_pitch(rp, target)
    t, y = simulate_ascent(rp, result.pitch_params, result.time_of_flight)

    fig = altitude_time_figure(t, y)

    if args.csv:
        write_csv(args.csv, t, y)
        print(f"Wrote CSV: {args.csv}")
    if args.geojson:
        write_geojson(args.geojson, t, y)
        print(f"Wrote GeoJSON: {args.geojson}")

    print("Optimization result:", result)

    if args.show:
        # Use default renderer; user can configure environment variable if needed.
        fig.show()


if __name__ == "__main__":
    main()
