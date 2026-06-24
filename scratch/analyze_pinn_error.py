import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.solver import BloodFlowSolver, HeartBeatController
from backend.geometry import VesselGeometry
from backend.pinn import PINNSurrogate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    heart_rate = 72.0
    T_cycle = 60.0 / heart_rate
    steps_per_cycle = 100
    dt = T_cycle / steps_per_cycle
    severity = 0.60
    
    Nx, Ny = 192, 96
    solver = BloodFlowSolver(Nx=Nx, Ny=Ny)
    geom = VesselGeometry(Nx=Nx, length=0.08, radius_max=0.004)
    radius_map = geom.stenosis(severity=severity)
    
    pinn_model = PINNSurrogate().to(device)
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pinn_checkpoint.pth")
    pinn_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    pinn_model.eval()
    
    # Run CFD ground truth
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
            "p": solver.p.copy(),
            "time": solver.time
        })
        
    fluid_indices = np.where(solver.is_fluid)
    
    # We will look at step 25 (approx peak flow, as sin(2*pi*f*t) is peak at T/4)
    step = 25
    cfd_u = cfd_states[step]["u"]
    cfd_v = cfd_states[step]["v"]
    cfd_p = cfd_states[step]["p"]
    t_val = cfd_states[step]["time"]
    
    # Predict PINN
    x_fluid = solver.x_coords[fluid_indices[0]]
    y_fluid = solver.y_coords[fluid_indices[1]]
    t_fluid = np.ones_like(x_fluid) * (t_val % T_cycle)
    sev_fluid = np.ones_like(x_fluid) * severity
    hr_fluid = np.ones_like(x_fluid) * heart_rate
    
    x_t = torch.tensor(x_fluid, dtype=torch.float32, device=device).view(-1, 1)
    y_t = torch.tensor(y_fluid, dtype=torch.float32, device=device).view(-1, 1)
    t_t = torch.tensor(t_fluid, dtype=torch.float32, device=device).view(-1, 1)
    sev_t = torch.tensor(sev_fluid, dtype=torch.float32, device=device).view(-1, 1)
    hr_t = torch.tensor(hr_fluid, dtype=torch.float32, device=device).view(-1, 1)
    
    with torch.no_grad():
        up_t, vp_t, pp_t = pinn_model.predict_physical(x_t, y_t, t_t, sev_t, hr_t)
        
    pinn_u = np.zeros_like(cfd_u)
    pinn_v = np.zeros_like(cfd_v)
    pinn_p = np.zeros_like(cfd_p)
    
    pinn_u[fluid_indices[0], fluid_indices[1]] = up_t.cpu().numpy().flatten()
    pinn_v[fluid_indices[0], fluid_indices[1]] = vp_t.cpu().numpy().flatten()
    pinn_p[fluid_indices[0], fluid_indices[1]] = pp_t.cpu().numpy().flatten()
    
    solver.enforce_velocity_bc(pinn_u, pinn_v)
    
    # Print stats
    print(f"Step {step} (t = {t_val % T_cycle:.3f} s):")
    print(f"  CFD u: mean = {cfd_u[solver.is_fluid].mean():.5f}, max = {cfd_u[solver.is_fluid].max():.5f}, min = {cfd_u[solver.is_fluid].min():.5f}")
    print(f"  PINN u: mean = {pinn_u[solver.is_fluid].mean():.5f}, max = {pinn_u[solver.is_fluid].max():.5f}, min = {pinn_u[solver.is_fluid].min():.5f}")
    
    print(f"  CFD v: mean = {cfd_v[solver.is_fluid].mean():.5f}, max = {cfd_v[solver.is_fluid].max():.5f}, min = {cfd_v[solver.is_fluid].min():.5f}")
    print(f"  PINN v: mean = {pinn_v[solver.is_fluid].mean():.5f}, max = {pinn_v[solver.is_fluid].max():.5f}, min = {pinn_v[solver.is_fluid].min():.5f}")
    
    print(f"  CFD p: mean = {cfd_p[solver.is_fluid].mean():.5f}, max = {cfd_p[solver.is_fluid].max():.5f}, min = {cfd_p[solver.is_fluid].min():.5f}")
    print(f"  PINN p: mean = {pinn_p[solver.is_fluid].mean():.5f}, max = {pinn_p[solver.is_fluid].max():.5f}, min = {pinn_p[solver.is_fluid].min():.5f}")
    
    # Calculate error at this step
    diff = np.sum((pinn_u[solver.is_fluid] - cfd_u[solver.is_fluid])**2 + (pinn_v[solver.is_fluid] - cfd_v[solver.is_fluid])**2)
    norm = np.sum(cfd_u[solver.is_fluid]**2 + cfd_v[solver.is_fluid]**2)
    err = np.sqrt(diff / norm)
    print(f"  Velocity L2 Relative Error at this step = {err:.2%}")
    
    # Let's check scaling factor
    # True inlet mean velocity scaling
    inlet_scale = solver.heart_beat.get_inlet_velocity_scaling(t_val)
    print(f"  True inlet scale = {inlet_scale:.5f}")
    
    # Measure inlet velocity from PINN u-velocity (column 1)
    pinn_inlet_scale = pinn_u[1, solver.is_fluid[1]].mean()
    print(f"  PINN inlet scale (column 1) = {pinn_inlet_scale:.5f}")
    
    # What if we scale the PINN u and v in the interior by true_inlet_scale / PINN_inlet_scale?
    scaling_ratio = inlet_scale / (pinn_inlet_scale + 1e-8)
    print(f"  Suggested scaling ratio = {scaling_ratio:.5f}")
    
    scaled_pinn_u = pinn_u * scaling_ratio
    scaled_pinn_v = pinn_v * scaling_ratio
    solver.enforce_velocity_bc(scaled_pinn_u, scaled_pinn_v)
    
    diff_scaled = np.sum((scaled_pinn_u[solver.is_fluid] - cfd_u[solver.is_fluid])**2 + (scaled_pinn_v[solver.is_fluid] - cfd_v[solver.is_fluid])**2)
    err_scaled = np.sqrt(diff_scaled / norm)
    print(f"  Velocity L2 Relative Error after dynamic scaling = {err_scaled:.2%}")

if __name__ == "__main__":
    main()
