import os
import sys
import time
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.solver import BloodFlowSolver, HeartBeatController
from backend.geometry import VesselGeometry
from backend.pinn import PINNSurrogate

class BloodFlowSolverJacobi(BloodFlowSolver):
    def project_pressure_custom(self, dt: float, jacobi_iters: int) -> None:
        if jacobi_iters == 0:
            return
            
        u_left = self.u[:-2, 1:-1]
        u_right = self.u[2:, 1:-1]
        v_down = self.v[1:-1, :-2]
        v_up = self.v[1:-1, 2:]
        
        div = np.zeros((self.Nx, self.Ny))
        div[1:-1, 1:-1] = (u_right - u_left) / (2.0 * self.dx) + (v_up - v_down) / (2.0 * self.dy)
        div[~self.is_fluid] = 0.0
        
        f = (self.density / dt) * div
        c_p = 2.0 / self.dx**2 + 2.0 / self.dy**2
        active_interior = self.is_fluid[1:-1, 1:-1]
        
        p_guess = self.p.copy()
        
        for _ in range(jacobi_iters):
            self.enforce_pressure_bc(p_guess)
            
            p_left = p_guess[:-2, 1:-1]
            p_right = p_guess[2:, 1:-1]
            p_down = p_guess[1:-1, :-2]
            p_up = p_guess[1:-1, 2:]
            
            p_next_interior = ((p_left + p_right) / self.dx**2 + (p_down + p_up) / self.dy**2 - f[1:-1, 1:-1]) / c_p
            p_guess[1:-1, 1:-1] = np.where(active_interior, p_next_interior, p_guess[1:-1, 1:-1])
            
        self.enforce_pressure_bc(p_guess)
        self.p = p_guess
        
        p_left = self.p[:-2, 1:-1]
        p_right = self.p[2:, 1:-1]
        p_down = self.p[1:-1, :-2]
        p_up = self.p[1:-1, 2:]
        
        grad_p_x = (p_right - p_left) / (2.0 * self.dx)
        grad_p_y = (p_up - p_down) / (2.0 * self.dy)
        
        self.u[1:-1, 1:-1] -= (dt / self.density) * grad_p_x * active_interior
        self.v[1:-1, 1:-1] -= (dt / self.density) * grad_p_y * active_interior
        
        self.enforce_velocity_bc(self.u, self.v)

def get_pinn_predictions(pinn_model, solver, device, severity, heart_rate, T_cycle, fluid_indices):
    solver.u_inlet = solver.heart_beat.get_inlet_velocity(solver.time, solver.y_coords, solver.radius_map[0])
    
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

def run_hybrid_simulation(pinn_model, solver, device, severity, heart_rate, T_cycle, steps_per_cycle, dt, fluid_indices, k_interval, beta, jacobi_iters):
    solver.initialize(solver.radius_map, heart_beat=solver.heart_beat)
    solver.time = T_cycle * 2.0  # start of 3rd cycle
    
    states = []
    t_start = time.time()
    
    for step in range(1, steps_per_cycle + 1):
        if step % k_interval == 0:
            solver.step(dt)
            u_grid = solver.u.copy()
            v_grid = solver.v.copy()
            p_grid = solver.p.copy()
        else:
            solver.time += dt
            u_pinn, v_pinn, p_pinn = get_pinn_predictions(pinn_model, solver, device, severity, heart_rate, T_cycle, fluid_indices)
            
            # Advect previous state
            u_adv = solver.advect(solver.u, dt)
            v_adv = solver.advect(solver.v, dt)
            solver.enforce_velocity_bc(u_adv, v_adv)
            
            # Blend
            u_grid = beta * u_adv + (1.0 - beta) * u_pinn
            v_grid = beta * v_adv + (1.0 - beta) * v_pinn
            solver.enforce_velocity_bc(u_grid, v_grid)
            solver.u = u_grid.copy()
            solver.v = v_grid.copy()
            
            # Custom pressure project
            solver.project_pressure_custom(dt, jacobi_iters)
            u_grid = solver.u.copy()
            v_grid = solver.v.copy()
            p_grid = solver.p.copy()
            
        states.append({
            "u": u_grid,
            "v": v_grid,
            "p": p_grid
        })
        
    t_end = time.time()
    return t_end - t_start, states

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    heart_rate = 72.0
    T_cycle = 60.0 / heart_rate
    steps_per_cycle = 100
    dt = T_cycle / steps_per_cycle
    severity = 0.60
    
    Nx, Ny = 192, 96
    solver = BloodFlowSolverJacobi(Nx=Nx, Ny=Ny)
    geom = VesselGeometry(Nx=Nx, length=0.08, radius_max=0.004)
    radius_map = geom.stenosis(severity=severity)
    
    pinn_model = PINNSurrogate().to(device)
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pinn_checkpoint.pth")
    pinn_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    pinn_model.eval()
    
    # Ground Truth
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

    betas = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99]
    
    print("Sweeping Beta values with Jacobi Iters = 5:")
    print(f"{"Beta":5} | {"FPS":6} | {"Time/Frame (ms)":15} | {"Velocity Error":14}")
    print("-"*48)
    for beta in betas:
        elapsed, states = run_hybrid_simulation(pinn_model, solver, device, severity, heart_rate, T_cycle, steps_per_cycle, dt, fluid_indices, 5, beta, 5)
        fps = steps_per_cycle / elapsed
        ms_per_frame = (elapsed / steps_per_cycle) * 1000
        err = calc_error(states)
        print(f"{beta:5.2f} | {fps:6.1f} | {ms_per_frame:15.2f} | {err:13.2%}")

    print("\nSweeping Beta values with Jacobi Iters = 10:")
    print(f"{"Beta":5} | {"FPS":6} | {"Time/Frame (ms)":15} | {"Velocity Error":14}")
    print("-"*48)
    for beta in betas:
        elapsed, states = run_hybrid_simulation(pinn_model, solver, device, severity, heart_rate, T_cycle, steps_per_cycle, dt, fluid_indices, 5, beta, 10)
        fps = steps_per_cycle / elapsed
        ms_per_frame = (elapsed / steps_per_cycle) * 1000
        err = calc_error(states)
        print(f"{beta:5.2f} | {fps:6.1f} | {ms_per_frame:15.2f} | {err:13.2%}")

if __name__ == "__main__":
    main()
