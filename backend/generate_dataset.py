"""
HaemoSim Dataset Generator
Academic/Educational Demonstration Tool. Not for clinical use.

Generates a training dataset by running BloodFlowSolver across a grid of
stenosis severities and heart rates, saving results to dataset.npz.
Uses parallel process execution to accelerate generation.
"""

import os
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.solver import BloodFlowSolver, HeartBeatController
from backend.geometry import VesselGeometry

def simulate_case(args):
    """
    Run a single case simulation for a specific severity and heart rate.
    """
    severity, heart_rate, Nx, Ny = args
    solver = BloodFlowSolver(Nx=Nx, Ny=Ny)
    geom = VesselGeometry(Nx=Nx, length=0.08, radius_max=0.004)
    
    # Artery with stenosis narrowing
    radius_map = geom.stenosis(severity=severity)
    
    # Dynamic pulsatile heart beat controller
    heart_beat = HeartBeatController(
        heart_rate=heart_rate,
        pulsatility_index=0.6,
        mean_velocity=0.1
    )
    solver.initialize(radius_map, heart_beat=heart_beat)
    
    # 100 timesteps per cycle
    T_cycle = 60.0 / heart_rate
    steps_per_cycle = 100
    dt = T_cycle / steps_per_cycle
    
    # Warmup for 2 cycles to reach periodic steady state
    warmup_steps = 2 * steps_per_cycle
    for _ in range(warmup_steps):
        solver.step(dt)
        
    # Record the 3rd cycle
    u_history = np.zeros((steps_per_cycle, Nx, Ny), dtype=np.float32)
    v_history = np.zeros((steps_per_cycle, Nx, Ny), dtype=np.float32)
    p_history = np.zeros((steps_per_cycle, Nx, Ny), dtype=np.float32)
    wss_top_history = np.zeros((steps_per_cycle, Nx), dtype=np.float32)
    wss_bottom_history = np.zeros((steps_per_cycle, Nx), dtype=np.float32)
    t_history = np.zeros(steps_per_cycle, dtype=np.float32)
    
    for step in range(steps_per_cycle):
        solver.step(dt)
        state = solver.get_state()
        wss = solver.compute_wall_shear_stress()
        
        # Save relative time in cardiac cycle (scaled [0, T_cycle])
        # Note: solver.time is absolute, we take its remainder of T_cycle
        t_in_cycle = solver.time % T_cycle
        
        u_history[step] = state["u"].astype(np.float32)
        v_history[step] = state["v"].astype(np.float32)
        p_history[step] = state["p"].astype(np.float32)
        wss_top_history[step] = wss["top"].astype(np.float32)
        wss_bottom_history[step] = wss["bottom"].astype(np.float32)
        t_history[step] = t_in_cycle
        
    return {
        "severity": severity,
        "heart_rate": heart_rate,
        "u": u_history,
        "v": v_history,
        "p": p_history,
        "wss_top": wss_top_history,
        "wss_bottom": wss_bottom_history,
        "t": t_history
    }

def main():
    start_time = time.time()
    Nx = 192
    Ny = 96
    
    # Parameter space grid
    severities = np.linspace(0.0, 0.75, 12)  # 12 severities
    heart_rates = np.linspace(50.0, 100.0, 8)  # 8 heart rates (50 to 100 BPM)
    
    cases = []
    for sev in severities:
        for hr in heart_rates:
            cases.append((sev, hr, Nx, Ny))
            
    print(f"Starting dataset generation for {len(cases)} cases...")
    print(f"Grid size: {Nx}x{Ny}, 100 timesteps per case (total {len(cases)*100} states).")
    
    # Run in parallel using all available cores
    results = []
    # Using ProcessPoolExecutor to distribute cases
    max_workers = min(os.cpu_count() or 4, 8)
    print(f"Spawning pool with {max_workers} processes...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(simulate_case, cases))
        
    print("All simulations completed. Organizing data matrices...")
    
    # Setup output matrices
    # Shape: (12 severities, 8 heart rates, 100 timesteps, Nx, Ny)
    u_data = np.zeros((12, 8, 100, Nx, Ny), dtype=np.float32)
    v_data = np.zeros((12, 8, 100, Nx, Ny), dtype=np.float32)
    p_data = np.zeros((12, 8, 100, Nx, Ny), dtype=np.float32)
    wss_top_data = np.zeros((12, 8, 100, Nx), dtype=np.float32)
    wss_bottom_data = np.zeros((12, 8, 100, Nx), dtype=np.float32)
    t_data = np.zeros((12, 8, 100), dtype=np.float32)
    
    # Get index mapping helper
    sev_to_idx = {val: idx for idx, val in enumerate(severities)}
    hr_to_idx = {val: idx for idx, val in enumerate(heart_rates)}
    
    for res in results:
        s_idx = sev_to_idx[res["severity"]]
        h_idx = hr_to_idx[res["heart_rate"]]
        
        u_data[s_idx, h_idx] = res["u"]
        v_data[s_idx, h_idx] = res["v"]
        p_data[s_idx, h_idx] = res["p"]
        wss_top_data[s_idx, h_idx] = res["wss_top"]
        wss_bottom_data[s_idx, h_idx] = res["wss_bottom"]
        t_data[s_idx, h_idx] = res["t"]
        
    # Get geometry grids and metadata
    solver = BloodFlowSolver(Nx=Nx, Ny=Ny)
    geom = VesselGeometry(Nx=Nx, length=0.08, radius_max=0.004)
    
    # Save is_fluid masks for each severity
    is_fluid_data = np.zeros((12, Nx, Ny), dtype=bool)
    for idx, sev in enumerate(severities):
        radius_map = geom.stenosis(severity=sev)
        solver.initialize(radius_map)
        is_fluid_data[idx] = solver.is_fluid.copy()
        
    # Save to disk
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset.npz")
    print(f"Saving dataset compressed npz to {output_path}...")
    np.savez_compressed(
        output_path,
        u=u_data,
        v=v_data,
        p=p_data,
        wss_top=wss_top_data,
        wss_bottom=wss_bottom_data,
        t=t_data,
        severities=severities,
        heart_rates=heart_rates,
        is_fluid=is_fluid_data,
        x=solver.x_coords,
        y=solver.y_coords
    )
    
    elapsed = time.time() - start_time
    print(f"Dataset generated successfully in {elapsed:.1f} seconds (~{elapsed/60:.1f} mins).")
    print(f"Dataset file size: {os.path.getsize(output_path) / (1024*1024):.1f} MB.")

if __name__ == "__main__":
    main()
