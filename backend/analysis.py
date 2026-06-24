"""
HaemoSim Analysis Module
Academic/Educational Demonstration Tool. Not for clinical use.

Calculates haemodynamic metric calculations:
- Wall Shear Stress (WSS) timeseries over cardiac cycles
- Oscillatory Shear Index (OSI)
- Recirculation zone detection and area fraction
- Pressure drop
"""

import numpy as np

def compute_wss_timeseries(solver, radius_map: np.ndarray, n_cycles: int = 3) -> dict:
    """
    Run the solver for n_cycles full heartbeats, and record the wall shear stress (WSS)
    at every timestep for the last full cycle.
    
    Parameters:
        solver (BloodFlowSolver): The initialized solver.
        radius_map (np.ndarray): The geometry radius map.
        n_cycles (int): Total number of cardiac cycles to run.
        
    Returns:
        dict: Containing 'time' (Nt,), 'top' (Nt, Nx) and 'bottom' (Nt, Nx) WSS values.
    """
    # Initialize the solver with the given geometry
    solver.initialize(radius_map)
    
    # Calculate time step and steps per cycle based on heart rate
    T_cycle = 60.0 / solver.heart_beat.heart_rate
    steps_per_cycle = 100
    dt = T_cycle / steps_per_cycle
    
    # 1. Run for (n_cycles - 1) cycles to reach periodic steady state
    warmup_steps = (n_cycles - 1) * steps_per_cycle
    for _ in range(warmup_steps):
        solver.step(dt)
        
    # 2. Record WSS for the last full cycle
    wss_top = []
    wss_bottom = []
    time_points = []
    
    for _ in range(steps_per_cycle):
        solver.step(dt)
        wss = solver.compute_wall_shear_stress()
        wss_top.append(wss['top'])
        wss_bottom.append(wss['bottom'])
        time_points.append(solver.time)
        
    return {
        "time": np.array(time_points),
        "top": np.array(wss_top),
        "bottom": np.array(wss_bottom)
    }

def compute_OSI(wss_timeseries: np.ndarray) -> np.ndarray:
    """
    Compute the Oscillatory Shear Index (OSI) over one cardiac cycle:
    OSI = 0.5 * (1 - |integral(WSS dt)| / integral(|WSS| dt))
    
    Parameters:
        wss_timeseries (np.ndarray): 2D array of shape (Nt, Nx) containing signed WSS.
        
    Returns:
        np.ndarray: 1D array of shape (Nx,) containing the OSI.
    """
    # Use np.trapezoid if available (NumPy >= 2.0.0), otherwise fallback to np.trapz
    trap_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    if trap_func is not None:
        numerator = np.abs(trap_func(wss_timeseries, axis=0))
        denominator = trap_func(np.abs(wss_timeseries), axis=0)
    else:
        numerator = np.abs(np.sum(wss_timeseries, axis=0))
        denominator = np.sum(np.abs(wss_timeseries), axis=0)
        
    # Compute OSI (add small epsilon to avoid division by zero)
    osi = 0.5 * (1.0 - numerator / (denominator + 1e-8))
    return osi

def compute_recirculation_zone(u: np.ndarray, is_fluid: np.ndarray) -> tuple:
    """
    Detect the reversed flow region downstream of a narrowing where u < 0.
    
    Parameters:
        u (np.ndarray): Horizontal velocity field of shape (Nx, Ny).
        is_fluid (np.ndarray): Fluid domain mask of shape (Nx, Ny).
        
    Returns:
        tuple: (recirculation_mask, area_fraction)
            recirculation_mask (np.ndarray): Boolean mask of shape (Nx, Ny).
            area_fraction (float): Percentage of the fluid domain occupied by recirculation.
    """
    recirculation_mask = (u < 0) & is_fluid
    fluid_area = np.sum(is_fluid)
    
    if fluid_area > 0:
        area_fraction = np.sum(recirculation_mask) / fluid_area
    else:
        area_fraction = 0.0
        
    return recirculation_mask, area_fraction

def compute_pressure_drop(pressure_field: np.ndarray, is_fluid: np.ndarray) -> float:
    """
    Compute the pressure drop across the vessel: delta_P = P_inlet - P_outlet.
    
    Parameters:
        pressure_field (np.ndarray): 2D array of shape (Nx, Ny) containing pressure.
        is_fluid (np.ndarray): Boolean mask of shape (Nx, Ny).
        
    Returns:
        float: Pressure drop in Pascals.
    """
    inlet_mask = is_fluid[0]
    outlet_mask = is_fluid[-1]
    
    p_inlet = np.mean(pressure_field[0, inlet_mask]) if np.any(inlet_mask) else 0.0
    p_outlet = np.mean(pressure_field[-1, outlet_mask]) if np.any(outlet_mask) else 0.0
    
    return float(p_inlet - p_outlet)
