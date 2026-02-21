"""Plotly-based visualization helpers for rocket trajectory results."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .ascent_opt import R_EARTH

def altitude_time_figure(t: np.ndarray, y: np.ndarray):
    """Return a Plotly figure of altitude vs. time.

    Parameters
    ----------
    t : np.ndarray
        1D array of time seconds.
    y : np.ndarray
        State matrix from simulate_ascent (shape (4, N)).
    """
    r = y[0]
    altitude = r - R_EARTH
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=altitude/1000.0, mode="lines", name="Altitude (km)"))
    fig.update_layout(
        title="Ascent Altitude Profile",
        xaxis_title="Time (s)",
        yaxis_title="Altitude (km)",
        template="plotly_white",
    )
    return fig
