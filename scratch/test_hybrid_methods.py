import os
import sys
import time
import numpy as np
import torch

# Ensure backend can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.solver import BloodFlowSolver, HeartBeatController
from backend.geometry import VesselGeometry
from backend.pinn import PINNSurrogate

def get_pinn_predictions(pinn_model, solver, device, severity, heart_rate, T_cycle, fluid_indices):
    """
    Utility to get PINN predictions at solver's current time.
    """
    solver.u_inlet = solver.heart_beat.get_inlet_velocity(solver.time, solver.y_coords, solver.radius_map[0])
    
    # Coordinate inputs
    x_fluid = solver.x_coords[fluid_indices[0]]
    y_fluid = solver.y_coords[fluid_indices[1]]
    t_fluid = np.ones_like(x_fluid) * (solver.time % T_cycle)
    sev_fluid = np.ones_like(x_fluid) * severity
    hr_fluid = np.ones_like(x_fluid) * heart_rate
    
    x_t = torch.tensor(x_fluid, dtype=torch.float32, device=device).view(-1, 1)
    y_t = torch.tensor(y_fluid, dtype=torch.float32, device=device).view(-1, 1)
    t_t = torch.tensor(t_fluid, dtype=torch.float32, device=device).view(-1, 1)
    sev_t = torch.tensor(sev_fluid, dtype=torch.float32, device=device).view(-1, 1)
    hr_t = torch.tensor(hr_fluid, dtype=torch.float32, device=device).view(-1, 1)
    
    with torch.no_grad():
        up_t, vp_t, pp_t = pinn_model.predict_physical(x_t, y_t, t_t, sev_t, hr_t)
        
    u_grid = np.zeros_like(solver.u)
    v_grid = np.zeros_like(solver.v)
    p_grid = np.zeros_like(solver.p)
    
    u_grid[fluid_indices[0], fluid_indices[1]] = up_t.cpu().numpy().flatten()
    v_grid[fluid_indices[0], fluid_indices[1]] = vp_t.cpu().numpy().flatten()
    p_grid[fluid_indices[0], fluid_indices[1]] = pp_t.cpu().numpy().flatten()
    
    solver.enforce_velocity_bc(u_grid, v_grid)
    return u_grid, v_grid, p_grid

def run_hybrid_simulation(pinn_model, solver, device, severity, heart_rate, T_cycle, steps_per_cycle, dt, fluid_indices, k_interval, strategy="base", beta=0.5):
    """
    Runs a hybrid cycle with a specific error reduction strategy:
    - 'base': standard alternating step, no special AI-step corrections
    - 'project': run pressure projection on AI steps to enforce divergence-free flow
    - 'blend': temporal blending between previous velocity and current PINN prediction
    - 'blend_project': both blending and pressure projection on AI steps
    - 'advect_project': semi-lagrangian advection of previous velocity + pressure projection
    """
    solver.initialize(solver.radius_map, heart_beat=solver.heart_beat)
    solver.time = T_cycle * 2.0  # start of 3rd cycle
    
    states = []
    t_start = time.time()
    
    for step in range(1, steps_per_cycle + 1):
        if step % k_interval == 0:
            # Physics step: standard CFD solver step starting from the current solver state
            solver.step(dt)
            u_grid = solver.u.copy()
            v_grid = solver.v.copy()
            p_grid = solver.p.copy()
        else:
            # AI step: predict using PINN surrogate
            solver.time += dt
            u_pinn, v_pinn, p_pinn = get_pinn_predictions(pinn_model, solver, device, severity, heart_rate, T_cycle, fluid_indices)
            
            if strategy == "base":
                u_grid, v_grid, p_grid = u_pinn, v_pinn, p_pinn
                solver.u = u_grid.copy()
                solver.v = v_grid.copy()
                solver.p = p_grid.copy()
            elif strategy == "project":
                solver.u = u_pinn.copy()
                solver.v = v_pinn.copy()
                solver.p = p_pinn.copy()
                # Run pressure projection to enforce incompressibility
                solver.project_pressure(dt)
                u_grid = solver.u.copy()
                v_grid = solver.v.copy()
                p_grid = solver.p.copy()
            elif strategy == "blend":
                # u^n = beta * u^{n-1} + (1 - beta) * u^{pinn}
                u_grid = beta * solver.u + (1.0 - beta) * u_pinn
                v_grid = beta * solver.v + (1.0 - beta) * v_pinn
                p_grid = beta * solver.p + (1.0 - beta) * p_pinn
                solver.enforce_velocity_bc(u_grid, v_grid)
                solver.u = u_grid.copy()
                solver.v = v_grid.copy()
                solver.p = p_grid.copy()
            elif strategy == "blend_project":
                u_grid = beta * solver.u + (1.0 - beta) * u_pinn
                v_grid = beta * solver.v + (1.0 - beta) * v_pinn
                p_grid = beta * solver.p + (1.0 - beta) * p_pinn
                solver.enforce_velocity_bc(u_grid, v_grid)
                solver.u = u_grid.copy()
                solver.v = v_grid.copy()
                solver.p = p_grid.copy()
                solver.project_pressure(dt)
                u_grid = solver.u.copy()
                v_grid = solver.v.copy()
                p_grid = solver.p.copy()
            elif strategy == "advect_project":
                # Run advection step first using solver's previous velocity
                u_adv = solver.advect(solver.u, dt)
                v_adv = solver.advect(solver.v, dt)
                solver.enforce_velocity_bc(u_adv, v_adv)
                # Then blend with PINN prediction to correct spatial errors
                u_grid = beta * u_adv + (1.0 - beta) * u_pinn
                v_grid = beta * v_adv + (1.0 - beta) * v_pinn
                solver.enforce_velocity_bc(u_grid, v_grid)
                solver.u = u_grid.copy()
                solver.v = v_grid.copy()
                # Run pressure projection to correct divergence
                solver.project_pressure(dt)
                u_grid = solver.u.copy()
                v_grid = solver.v.copy()
                p_grid = solver.p.copy()
                
        states.append({
            "u": u_grid,
            "v": v_grid,
            "p": p_grid
        })
        
    t_end = time.time()
    elapsed = t_end - t_start
    return elapsed, states

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Hybrid Strategies Evaluation on device: {device}")
    
    # Setup simulation parameters
    heart_rate = 72.0  # BPM
    T_cycle = 60.0 / heart_rate
    steps_per_cycle = 100
    dt = T_cycle / steps_per_cycle
    severity = 0.60  # 60% Stenosis
    
    Nx, Ny = 192, 96
    solver = BloodFlowSolver(Nx=Nx, Ny=Ny)
    geom = VesselGeometry(Nx=Nx, length=0.08, radius_max=0.004)
    radius_map = geom.stenosis(severity=severity)
    
    # Load PINN model
    pinn_model = PINNSurrogate().to(device)
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pinn_checkpoint.pth")
    pinn_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    pinn_model.eval()
    
    # --- CFD Ground Truth ---
    heart_beat = HeartBeatController(heart_rate=heart_rate, pulsatility_index=0.6, mean_velocity=0.1)
    solver.initialize(radius_map, heart_beat=heart_beat)
    for _ in range(2 * steps_per_cycle):
        solver.step(dt)
        
    cfd_states = []
    for _ in range(steps_per_cycle):
        solver.step(dt)
        cfd_states.append({
            "u": solver.u.copy(),
            "v": solver.v.copy(),
            "p": solver.p.copy()
        })
        
    fluid_indices = np.where(solver.is_fluid)
    
    # Helper to calculate errors
    def calc_error(states):
        l2_errors = []
        for step in range(steps_per_cycle):
            cfd_u = cfd_states[step]["u"][solver.is_fluid]
            cfd_v = cfd_states[step]["v"][solver.is_fluid]
            test_u = states[step]["u"][solver.is_fluid]
            test_v = states[step]["v"][solver.is_fluid]
            
            diff = np.sum((test_u - cfd_u)**2 + (test_v - cfd_v)**2)
            norm = np.sum(cfd_u**2 + cfd_v**2)
            l2_errors.append(np.sqrt(diff / norm))
        return np.mean(l2_errors)
        
    # Evaluate strategies
    strategies = [
        ("base", 5, "Hybrid 4:1 (Base)"),
        ("project", 5, "Hybrid 4:1 (+Proj)"),
        ("blend", 5, "Hybrid 4:1 (+Blend beta=0.8)"),
        ("blend_project", 5, "Hybrid 4:1 (+Blend +Proj beta=0.8)"),
        ("blend_project", 5, "Hybrid 4:1 (+Blend +Proj beta=0.5)"),
        ("blend_project", 5, "Hybrid 4:1 (+Blend +Proj beta=0.2)"),
        ("advect_project", 5, "Hybrid 4:1 (+Adv +Proj beta=0.8)"),
        ("advect_project", 5, "Hybrid 4:1 (+Adv +Proj beta=0.5)"),
        ("advect_project", 5, "Hybrid 4:1 (+Adv +Proj beta=0.2)"),
        
        ("base", 2, "Hybrid 1:1 (Base)"),
        ("project", 2, "Hybrid 1:1 (+Proj)"),
        ("blend_project", 2, "Hybrid 1:1 (+Blend +Proj beta=0.5)"),
        ("advect_project", 2, "Hybrid 1:1 (+Adv +Proj beta=0.5)"),
    ]
    
    print("\nEvaluating strategies...")
    print(f"{"Strategy Name":35} | {"FPS":6} | {"Time/Frame (ms)":15} | {"Velocity Error":14}")
    print("-"*76)
    
    for strategy, k_interval, name in strategies:
        # Extract beta if applicable
        beta = 0.5
        if "beta=0.8" in name:
            beta = 0.8
        elif "beta=0.2" in name:
            beta = 0.2
            
        elapsed, states = run_hybrid_simulation(
            pinn_model, solver, device, severity, heart_rate, 
            T_cycle, steps_per_cycle, dt, fluid_indices, k_interval,
            strategy=strategy, beta=beta
        )
        fps = steps_per_cycle / elapsed
        ms_per_frame = (elapsed / steps_per_cycle) * 1000
        err = calc_error(states)
        print(f"{name:35} | {fps:6.1f} | {ms_per_frame:15.2f} | {err:13.2%}")

if __name__ == "__main__":
    main()
