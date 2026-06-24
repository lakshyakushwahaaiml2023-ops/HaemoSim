"""
Generate publication-quality figures (300dpi) for the HaemoSim project:
1. Poiseuille validation plot (simulated vs analytic)
2. WSS(x) and OSI(x) for healthy vs stenosis overlaid
3. PINN benchmark table as a figure
4. Velocity field snapshots at peak systole for each vessel type

All figures are saved in the artifacts directory.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.solver import BloodFlowSolver, HeartBeatController
from backend.geometry import VesselGeometry
from backend.pinn import PINNSurrogate
from backend.analysis import compute_OSI

# Artifacts output directory
ARTIFACTS_DIR = r"C:\Users\PREDATOR\.gemini\antigravity\brain\dacff554-137b-471d-bd68-184eaa959a43"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Use a clean, professional plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'legend.fontsize': 10,
    'grid.alpha': 0.4,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def generate_fig1_poiseuille():
    """Figure 1: Poiseuille validation plot (simulated vs analytic)"""
    print("\n--- Generating Figure 1: Poiseuille Validation ---")
    Nx, Ny = 192, 96
    solver = BloodFlowSolver(Nx=Nx, Ny=Ny)
    
    R = 0.004  # m
    radius_map = np.ones(Nx) * R
    u_avg = 0.1
    mu = solver.viscosity
    
    # Parabolic inlet profile
    y_coords = solver.y_coords
    u_inlet = np.zeros(Ny)
    for j in range(Ny):
        y = y_coords[j]
        if abs(y) <= R:
            u_inlet[j] = 1.5 * u_avg * (1.0 - (y / R)**2)
            
    solver.initialize(radius_map, inlet_velocity_profile=u_inlet)
    
    # Run to steady state
    dt = 0.002
    max_steps = 1500
    u_old = solver.u.copy()
    
    for step_idx in range(1, max_steps + 1):
        solver.step(dt)
        diff = np.max(np.abs(solver.u - u_old))
        u_old = solver.u.copy()
        if diff < 1e-5 and step_idx >= 100:
            print(f"  Poiseuille converged at step {step_idx} (diff={diff:.2e})")
            break
            
    i_mid = Nx // 2
    fluid_mask = solver.is_fluid[i_mid]
    y_fluid = y_coords[fluid_mask]
    u_sim = solver.u[i_mid, fluid_mask]
    
    u_max = 1.5 * u_avg
    dp_dx = -(2.0 * mu * u_max) / (R**2)
    # u(r) = - (dp/dx / (2 * mu)) * (R^2 - r^2)
    u_anal = -(dp_dx / (2.0 * mu)) * (R**2 - y_fluid**2)
    
    l2_error = np.sqrt(np.sum((u_sim - u_anal)**2)) / np.sqrt(np.sum(u_anal**2))
    print(f"  Normalized L2 Error: {l2_error:.4%}")
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(u_anal * 100, y_fluid * 1000, 'g-', label='Analytical Poiseuille (Planar)', linewidth=2.5)
    ax.plot(u_sim * 100, y_fluid * 1000, 'r--', label='Simulated Stable Fluids', linewidth=2)
    
    ax.set_title('Velocity Profile Validation (Steady Planar Poiseuille Flow)', fontweight='bold', pad=15)
    ax.set_xlabel('Velocity (cm/s)')
    ax.set_ylabel('Vertical Coordinate y (mm)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True)
    
    text_str = f'Convergence: Step {step_idx}\nNormalized $L^2$ Error: {l2_error:.4%}'
    ax.text(0.05, 0.05, text_str, transform=ax.transAxes, fontsize=11, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))
    
    output_path = os.path.join(ARTIFACTS_DIR, "fig1_poiseuille_validation.png")
    plt.savefig(output_path)
    plt.close()
    print(f"  Figure 1 saved to: {output_path}")

def generate_fig2_wss_osi():
    """Figure 2: WSS(x) and OSI(x) for healthy vs stenosis overlaid"""
    print("\n--- Generating Figure 2: WSS & OSI Comparison ---")
    Nx, Ny = 192, 96
    solver = BloodFlowSolver(Nx=Nx, Ny=Ny)
    geom = VesselGeometry(Nx=Nx, length=0.08, radius_max=0.004)
    
    heart_rate = 72.0
    T_cycle = 60.0 / heart_rate
    steps_per_cycle = 100
    dt = T_cycle / steps_per_cycle
    
    # Helper to simulate 3 cycles and return WSS and OSI
    def simulate_case(radius_map):
        heart_beat = HeartBeatController(heart_rate=heart_rate, pulsatility_index=0.6, mean_velocity=0.1)
        solver.initialize(radius_map, heart_beat=heart_beat)
        
        # Run 2 cycles warmup
        for _ in range(2 * steps_per_cycle):
            solver.step(dt)
            
        # Record 3rd cycle
        wss_top_history = []
        wss_bottom_history = []
        
        for _ in range(steps_per_cycle):
            solver.step(dt)
            wss = solver.compute_wall_shear_stress()
            wss_top_history.append(wss["top"].copy())
            wss_bottom_history.append(wss["bottom"].copy())
            
        wss_top_history = np.array(wss_top_history)
        wss_bottom_history = np.array(wss_bottom_history)
        
        # Compute OSI
        osi_top = compute_OSI(wss_top_history)
        osi_bottom = compute_OSI(wss_bottom_history)
        
        # Compute Time-Averaged WSS magnitude
        ta_wss_top = np.mean(np.abs(wss_top_history), axis=0)
        ta_wss_bottom = np.mean(np.abs(wss_bottom_history), axis=0)
        
        return ta_wss_top, osi_top, ta_wss_bottom, osi_bottom

    print("  Simulating Healthy vessel...")
    h_ta_wss_top, h_osi_top, _, _ = simulate_case(geom.healthy())
    
    print("  Simulating Stenosed vessel (60% severity)...")
    s_ta_wss_top, s_osi_top, _, _ = simulate_case(geom.stenosis(severity=0.6))
    
    x_coords_mm = solver.x_coords * 1000  # convert to mm
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    
    # Subplot 1: Time-Averaged Wall Shear Stress (TAWSS)
    ax1.plot(x_coords_mm, h_ta_wss_top, 'b-', label='Healthy Artery', linewidth=2)
    ax1.plot(x_coords_mm, s_ta_wss_top, 'r-', label='Stenosed Artery (60% Severity)', linewidth=2.5)
    ax1.axvline(40, color='gray', linestyle='--', alpha=0.5, label='Stenosis Throat (40mm)')
    ax1.set_ylabel('TAWSS (Pa)')
    ax1.set_title('Time-Averaged Wall Shear Stress (TAWSS) Along Vessel Wall', fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')
    
    # Subplot 2: Oscillatory Shear Index (OSI)
    ax2.plot(x_coords_mm, h_osi_top, 'b-', label='Healthy Artery', linewidth=2)
    ax2.plot(x_coords_mm, s_osi_top, 'r-', label='Stenosed Artery (60% Severity)', linewidth=2.5)
    ax2.axvline(40, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Axial Location x (mm)')
    ax2.set_ylabel('OSI')
    ax2.set_title('Oscillatory Shear Index (OSI) Along Vessel Wall', fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Mark OSI peak location downstream
    peak_idx = np.argmax(s_osi_top[10:-10]) + 10 # exclude boundaries
    peak_x = x_coords_mm[peak_idx]
    peak_val = s_osi_top[peak_idx]
    ax2.annotate(f'Peak OSI = {peak_val:.4f}\nat x = {peak_x:.1f} mm', 
                 xy=(peak_x, peak_val), 
                 xytext=(peak_x + 6, peak_val - 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6),
                 fontweight='bold', bbox=dict(facecolor='wheat', alpha=0.8, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    output_path = os.path.join(ARTIFACTS_DIR, "fig2_wss_osi_comparison.png")
    plt.savefig(output_path)
    plt.close()
    print(f"  Figure 2 saved to: {output_path}")

def generate_fig3_pinn_table():
    """Figure 3: PINN benchmark table as a publication-style figure"""
    print("\n--- Generating Figure 3: PINN Benchmark Table ---")
    
    # Setup data
    data = [
        ["20.5%", "8.904 s", "0.685 s", "13.0x", "46.27%", "0.42 mm (1 cell)"],
        ["47.7%", "9.055 s", "0.831 s", "10.9x", "50.40%", "7.92 mm (19 cells)"]
    ]
    columns = ["Stenosis Severity\n(Held-Out Cases)", "CFD Execution Time\n(3 Cardiac Cycles)", 
               "PINN Inference Time\n(3 Cardiac Cycles)", "Inference Speedup\n(Ratio)", 
               "Velocity L2 Relative\nField Error (%)", "OSI Peak Location\nAxial Error (mm)"]
    
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis('off')
    
    # Create Table
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 2.5)
    
    # Apply colors
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            # Header styling
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#1e3a8a')  # Dark blue header
        else:
            # Body styling
            cell.set_text_props(color='black')
            if col_idx == 3:
                # Highlight speedup green
                cell.set_text_props(weight='bold', color='#15803d')
                cell.set_facecolor('#f0fdf4')
            elif col_idx == 5:
                # Highlight OSI accuracy
                cell.set_facecolor('#fffbeb')
            else:
                cell.set_facecolor('#f8fafc' if row_idx % 2 == 0 else '#ffffff')
                
    ax.set_title("Table 1: CFD Finite Difference Solver vs PINN Deep Surrogate Benchmark", 
                 fontweight='bold', fontsize=12, pad=20)
    
    output_path = os.path.join(ARTIFACTS_DIR, "fig3_pinn_benchmark.png")
    plt.savefig(output_path)
    plt.close()
    print(f"  Figure 3 saved to: {output_path}")

def generate_fig4_velocity_snapshots():
    """Figure 4: Velocity field snapshots at peak systole for each vessel type"""
    print("\n--- Generating Figure 4: Velocity Field Snapshots ---")
    Nx, Ny = 192, 96
    solver = BloodFlowSolver(Nx=Nx, Ny=Ny)
    geom = VesselGeometry(Nx=Nx, length=0.08, radius_max=0.004)
    
    heart_rate = 72.0
    T_cycle = 60.0 / heart_rate
    steps_per_cycle = 100
    dt = T_cycle / steps_per_cycle
    
    cases = [
        {"name": "Healthy Artery", "map": geom.healthy()},
        {"name": "Single Stenosis (60%)", "map": geom.stenosis(severity=0.6)},
        {"name": "Saccular Aneurysm (60%)", "map": geom.aneurysm(dilation=0.6)},
        {"name": "Tandem Stenosis (50%/30%)", "map": geom.tandem_stenosis(severity1=0.5, severity2=0.3)}
    ]
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    
    # Pulse waveform evaluation to find peak systole step
    # Waveform: mean_velocity * (1 + pulsatility_index * sin(2*pi*f*t))
    # Peak occurs at sin(...) = 1, i.e., t = T/4 (step 25 in cycle)
    peak_systole_step = 25
    
    for idx, case in enumerate(cases):
        print(f"  Running CFD for {case['name']}...")
        heart_beat = HeartBeatController(heart_rate=heart_rate, pulsatility_index=0.6, mean_velocity=0.1)
        solver.initialize(case["map"], heart_beat=heart_beat)
        
        # Run 2 cycles + steps to peak systole in 3rd cycle
        total_steps = 2 * steps_per_cycle + peak_systole_step
        for _ in range(total_steps):
            solver.step(dt)
            
        u_field = solver.u.copy()
        v_field = solver.v.copy()
        u_mag = np.sqrt(u_field**2 + v_field**2)
        
        # Apply fluid mask (set solid wall coordinates to NaN for plotting)
        u_mag[~solver.is_fluid] = np.nan
        
        # Plot colormap
        ax = axes[idx]
        
        # Setup coordinates grid for plotting
        x_mesh, y_mesh = np.meshgrid(solver.x_coords * 1000, solver.y_coords * 1000, indexing='ij')
        
        im = ax.pcolormesh(x_mesh, y_mesh, u_mag, cmap='viridis', shading='auto', vmin=0.0, vmax=0.45)
        
        # Draw boundaries
        # Upper wall boundary
        ax.plot(solver.x_coords * 1000, case["map"] * 1000, 'k-', linewidth=1.5)
        # Lower wall boundary
        ax.plot(solver.x_coords * 1000, -case["map"] * 1000, 'k-', linewidth=1.5)
        
        ax.set_ylabel('y (mm)')
        ax.set_ylim(-6.0, 6.0)
        ax.grid(False) # Turn off grid overlaying the contour
        
        # Annotate case name
        ax.text(0.02, 0.82, case["name"], transform=ax.transAxes, color='black',
                weight='bold', fontsize=11, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
        
    axes[-1].set_xlabel('Axial Coordinate x (mm)')
    
    # Add a global colorbar
    fig.subplots_adjust(right=0.85, hspace=0.3)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Velocity Magnitude (m/s)', rotation=270, labelpad=15, weight='bold')
    
    plt.suptitle("Fluid Velocity Fields at Peak Systole ($t = t_{systole}$)", fontweight='bold', y=0.96, fontsize=14)
    
    output_path = os.path.join(ARTIFACTS_DIR, "fig4_velocity_snapshots.png")
    plt.savefig(output_path)
    plt.close()
    print(f"  Figure 4 saved to: {output_path}")

def main():
    print("=== STARTING FIGURE GENERATION SCRIPTS ===")
    start_time = time.time()
    
    generate_fig1_poiseuille()
    generate_fig2_wss_osi()
    generate_fig3_pinn_table()
    generate_fig4_velocity_snapshots()
    
    elapsed = time.time() - start_time
    print(f"\n=== SUCCESS: All figures generated and saved in {elapsed:.1f} s ===")

if __name__ == "__main__":
    main()
