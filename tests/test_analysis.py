"""
HaemoSim Analysis Module Tests & Clinical Verification
Academic/Educational Demonstration Tool. Not for clinical use.

Verifies the scientific claim that Oscillatory Shear Index (OSI)
peaks downstream of a stenosis narrowing (throat) rather than at the throat itself.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from backend.solver import BloodFlowSolver, HeartBeatController
from backend.geometry import VesselGeometry
from backend.analysis import compute_wss_timeseries, compute_OSI

def test_stenosis_osi_peak():
    """
    Run a pulsatile simulation in a 60% stenosis geometry, compute the OSI along
    the walls, confirm it peaks downstream of the throat, and save a verification plot.
    """
    Nx = 192
    Ny = 96
    solver = BloodFlowSolver(Nx=Nx, Ny=Ny)
    geom = VesselGeometry(Nx=Nx, length=0.08, radius_max=0.004)
    
    # Generate 60% stenosis geometry
    radius_map = geom.stenosis(severity=0.6)
    
    # Setup heartbeat
    heart_beat = HeartBeatController(heart_rate=72.0, pulsatility_index=0.6, mean_velocity=0.1)
    solver.initialize(radius_map, heart_beat=heart_beat)
    
    # Run and compute WSS timeseries over 3 cycles (record last cycle)
    print("\nSimulating pulsatile flow in stenosis geometry...")
    wss = compute_wss_timeseries(solver, radius_map, n_cycles=3)
    
    # Compute OSI along top and bottom walls
    osi_top = compute_OSI(wss['top'])
    osi_bottom = compute_OSI(wss['bottom'])
    
    # Throat center is at L/2 = 40mm (index 96 of 192)
    # Exclude boundary regions near inlet/outlet to avoid numerical artifacts
    active_range = slice(10, Nx - 10)
    x_active = solver.x_coords[active_range]
    osi_top_active = osi_top[active_range]
    osi_bottom_active = osi_bottom[active_range]
    
    # Find peak OSI positions
    peak_idx_top = np.argmax(osi_top_active)
    peak_idx_bottom = np.argmax(osi_bottom_active)
    
    x_peak_top = x_active[peak_idx_top]
    x_peak_bottom = x_active[peak_idx_bottom]
    
    print("\nClinical Analysis Results:")
    print(f"  Stenosis throat location: 0.040 m (40.0 mm)")
    print(f"  Peak OSI Top location:    {x_peak_top:.3f} m ({x_peak_top*1000:.1f} mm), OSI = {osi_top_active[peak_idx_top]:.4f}")
    print(f"  Peak OSI Bottom location: {x_peak_bottom:.3f} m ({x_peak_bottom*1000:.1f} mm), OSI = {osi_bottom_active[peak_idx_bottom]:.4f}")
    
    # Plotting OSI(x) along vessel length
    plt.figure(figsize=(10, 6))
    plt.plot(solver.x_coords * 1000, osi_top, 'r-', label='Top Wall OSI', linewidth=2)
    plt.plot(solver.x_coords * 1000, osi_bottom, 'b--', label='Bottom Wall OSI', linewidth=2)
    
    # Add vertical lines for throat and peak OSI locations
    plt.axvline(x=40.0, color='gray', linestyle=':', label='Stenosis Throat (40mm)', linewidth=1.5)
    plt.axvline(x=x_peak_top * 1000, color='red', linestyle='-.', alpha=0.6, label='Peak OSI Top')
    plt.axvline(x=x_peak_bottom * 1000, color='blue', linestyle='-.', alpha=0.6, label='Peak OSI Bottom')
    
    # Color-code the stenosis throat region
    plt.axvspan(30.0, 50.0, color='orange', alpha=0.08, label='Narrowing Zone')
    
    plt.title('Oscillatory Shear Index (OSI) along Stenosed Vessel', fontsize=14, fontweight='bold')
    plt.xlabel('Vessel Axial Location x (mm)', fontsize=12)
    plt.ylabel('OSI', fontsize=12)
    plt.ylim(-0.02, 0.52)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    
    # Save the plot in the artifact directory
    output_path = r"C:\Users\PREDATOR\.gemini\antigravity\brain\dacff554-137b-471d-bd68-184eaa959a43\osi_plot.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"OSI Plot saved successfully to {output_path}")
    
    # ASSERTIONS: Peak OSI must occur downstream of the throat (x > 0.040m)
    assert x_peak_top > 0.040, f"Peak OSI Top ({x_peak_top*1000:.1f}mm) is not downstream of throat (40.0mm)!"
    assert x_peak_bottom > 0.040, f"Peak OSI Bottom ({x_peak_bottom*1000:.1f}mm) is not downstream of throat (40.0mm)!"
