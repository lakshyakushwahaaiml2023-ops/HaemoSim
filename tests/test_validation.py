"""
HaemoSim Validation Module Tests
Academic/Educational Demonstration Tool. Not for clinical use.

Tests the BloodFlowSolver against the closed-form analytical Poiseuille flow profile
and reports the normalized L2 error.
"""

import numpy as np
from backend.solver import BloodFlowSolver
from backend.validation import poiseuille_analytic

def test_poiseuille_validation():
    """
    Run the solver on a straight vessel with a parabolic inlet velocity profile
    until steady state is reached. Compare the simulated velocity profile against
    the poiseuille_analytic solution and verify the L2 error is under 10%.
    """
    Nx = 192
    Ny = 96
    solver = BloodFlowSolver(Nx=Nx, Ny=Ny)
    
    # Healthy straight vessel (constant radius)
    R = 0.004  # m
    radius_map = np.ones(Nx) * R
    
    # Parabolic inlet profile (average velocity = 0.1 m/s, max velocity = 0.15 m/s)
    u_avg = 0.1
    mu = solver.viscosity
    
    y_coords = solver.y_coords
    u_inlet = np.zeros(Ny)
    for j in range(Ny):
        y = y_coords[j]
        if abs(y) <= R:
            # Planar Poiseuille: u(y) = 1.5 * u_avg * (1 - y^2/R^2)
            u_inlet[j] = 1.5 * u_avg * (1.0 - (y / R)**2)
            
    # Initialize the solver
    solver.initialize(radius_map, inlet_velocity_profile=u_inlet)
    
    # Time stepping loop
    dt = 0.002  # Time step in seconds
    max_steps = 1500
    converged = False
    
    u_old = solver.u.copy()
    
    for step_idx in range(1, max_steps + 1):
        solver.step(dt)
        
        # Check convergence: velocity change between steps < 1e-5
        u_new = solver.u
        diff = np.max(np.abs(u_new - u_old))
        u_old = u_new.copy()
        
        # Require at least 100 steps to let pressure projection stabilize
        if diff < 1e-5 and step_idx >= 100:
            print(f"\nSolver converged at step {step_idx}")
            converged = True
            break
            
    # Extract profile in the middle of the channel to avoid inlet/outlet boundary boundary effects
    i_mid = Nx // 2
    fluid_mask = solver.is_fluid[i_mid]
    y_fluid = y_coords[fluid_mask]
    u_sim = solver.u[i_mid, fluid_mask]
    
    # Analytical solution
    u_max = 1.5 * u_avg
    dp_dx = -(2.0 * mu * u_max) / (R**2)
    u_anal = poiseuille_analytic(y_fluid, R, dp_dx, mu)
    
    # Compute normalized L2 error
    l2_error = np.sqrt(np.sum((u_sim - u_anal)**2)) / np.sqrt(np.sum(u_anal**2))
    
    print(f"\nValidation Results:")
    print(f"  Steady state converged: {converged}")
    print(f"  Final step diff: {diff:.2e}")
    print(f"  Simulated u_max (center): {np.max(u_sim):.5f} m/s")
    print(f"  Analytical u_max (center): {np.max(u_anal):.5f} m/s")
    print(f"  Normalized L2 Error: {l2_error:.4%}")
    
    # Report if L2 error is above target
    assert converged, f"Solver did not reach steady state within {max_steps} steps. Last step difference: {diff:.2e}"
    assert l2_error < 0.10, f"L2 Error {l2_error:.4%} exceeds the 10% threshold. The solver has a bug or needs grid refinement."

def test_pulsatile_inlet_periodicity():
    """
    Run the solver with a pulsatile heartbeat inlet for 3 full cardiac cycles
    and confirm that the inlet velocity profile is perfectly periodic (cycle N+1 matches cycle N).
    """
    from backend.solver import HeartBeatController
    
    Nx = 192
    Ny = 96
    solver = BloodFlowSolver(Nx=Nx, Ny=Ny)
    
    # Healthy straight vessel (constant radius)
    R = 0.004  # m
    radius_map = np.ones(Nx) * R
    
    # Heartbeat parameters
    heart_rate = 72.0  # BPM
    pulsatility_index = 0.6
    mean_velocity = 0.1
    
    heart_beat = HeartBeatController(
        heart_rate=heart_rate,
        pulsatility_index=pulsatility_index,
        mean_velocity=mean_velocity
    )
    
    # Initialize the solver
    solver.initialize(radius_map, heart_beat=heart_beat)
    
    # Calculate Womersley number and print it
    alpha = solver.compute_womersley_number()
    print(f"\nCalculated Womersley number alpha: {alpha:.5f}")
    
    # We want T_cycle / dt to be an exact integer.
    # Period T_cycle = 60 / 72 = 5/6 seconds.
    # If dt = 1/120 seconds, steps_per_cycle = 100.
    dt = 1.0 / 120.0
    steps_per_cycle = 100
    total_steps = 3 * steps_per_cycle  # 3 cycles
    
    # Record inlet velocity u profile at each step
    inlet_history = []
    
    for step_idx in range(total_steps):
        solver.step(dt)
        # Record the inlet velocity profile (column 0)
        inlet_history.append(solver.u[0, :].copy())
        
    # Compare cycle 2 with cycle 1, and cycle 3 with cycle 2
    for step_in_cycle in range(steps_per_cycle):
        c1 = inlet_history[step_in_cycle]
        c2 = inlet_history[step_in_cycle + steps_per_cycle]
        c3 = inlet_history[step_in_cycle + 2 * steps_per_cycle]
        
        # Assert that they match within numerical tolerance
        np.testing.assert_allclose(c2, c1, rtol=1e-6, atol=1e-8,
            err_msg=f"Mismatch between Cycle 2 and Cycle 1 at step {step_in_cycle}")
        np.testing.assert_allclose(c3, c2, rtol=1e-6, atol=1e-8,
            err_msg=f"Mismatch between Cycle 3 and Cycle 2 at step {step_in_cycle}")
            
    print("Periodicity check passed: all 3 cycles are identical within numerical tolerance.")

