from rocket_trajectory.ascent_opt import RocketParams, TargetOrbit, optimize_pitch, R_EARTH


def test_optimize_pitch_basic():
    rp = RocketParams(thrust=2.5e6, isp=300.0, dry_mass=30000.0, prop_mass=200000.0)
    target = TargetOrbit(altitude=200e3)
    result = optimize_pitch(rp, target)
    assert result.success, f"Optimization failed: {result.message}"
    # Check altitude tolerance ~ +/- 30 km
    # Simulate altitude from pitch params indirectly: run optimize already did simulation
    # For simplicity, recompute final radius
    assert abs(result.final_altitude - target.altitude) < 30e3
    # Delta-v should be positive and less than some upper bound (e.g., 9 km/s for LEO insertion)
    assert 0 < result.delta_v < 9000
