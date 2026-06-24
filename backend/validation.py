"""
HaemoSim Validation Module
Academic/Educational Demonstration Tool. Not for clinical use.

This module holds analytic comparisons (e.g., Poiseuille flow)
to validate CFD and PINN results.
"""

import numpy as np

def poiseuille_analytic(r: np.ndarray, R: float, dp_dx: float, mu: float) -> np.ndarray:
    """
    Compute the analytical parabolic velocity profile for steady planar Poiseuille flow
    (flow between two parallel plates separated by distance 2R).

    Parameters:
        r (np.ndarray): Coordinate(s) measured from the channel centerline (range [-R, R]) (m).
        R (float): Channel half-width / vessel radius (m).
        dp_dx (float): Pressure gradient (Pa/m).
        mu (float): Dynamic viscosity of the fluid (Pa.s).

    Returns:
        np.ndarray: Analytical velocity profile (m/s).
    """
    # u(r) = - (dp/dx / (2 * mu)) * (R^2 - r^2)
    # Since flow goes in +x direction, dp_dx must be negative.
    return -(dp_dx / (2.0 * mu)) * (R**2 - r**2)
