"""
HaemoSim FastAPI Application Server
Academic/Educational Demonstration Tool. Not for clinical use.

Exposes REST endpoints and a WebSocket stream to control the simulation,
switch between CFD and PINN modes, and visualize flow metrics in real-time.
"""

import os
import sys
import json
import asyncio
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.solver import BloodFlowSolver, HeartBeatController
from backend.geometry import VesselGeometry, compute_area_reduction
from backend.pinn import PINNSurrogate
from backend.analysis import compute_OSI, compute_recirculation_zone, compute_pressure_drop

app = FastAPI(
    title="HaemoSim API",
    description="Academic 2D Haemodynamics Simulator & PINN Benchmark API.",
    version="0.1.0"
)

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_index():
    frontend_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "frontend",
        "index.html"
    )
    return FileResponse(frontend_path)

# Global Simulation & Model State
active_vessel_type = "healthy"
active_severity = 0.6
active_mode = "cfd"
active_heart_rate = 72.0

Nx, Ny = 192, 96
solver = BloodFlowSolver(Nx=Nx, Ny=Ny)
geom = VesselGeometry(Nx=Nx, length=0.08, radius_max=0.004)
heart_beat = HeartBeatController(heart_rate=active_heart_rate, pulsatility_index=0.6, mean_velocity=0.1)

# Initialize with healthy geometry by default
radius_map = geom.healthy()
solver.initialize(radius_map, heart_beat=heart_beat)

# Load PINN Surrogate Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pinn_model = PINNSurrogate().to(device)
checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pinn_checkpoint.pth")

pinn_available = False
if os.path.exists(checkpoint_path):
    try:
        pinn_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        pinn_model.eval()
        pinn_available = True
        print(f"Loaded PINN checkpoint successfully on device {device}.")
    except Exception as e:
        print(f"Failed to load PINN checkpoint: {e}")
else:
    print(f"PINN checkpoint not found at {checkpoint_path}. Only CFD mode will be active.")

# Pydantic schemas
class ConfigureRequest(BaseModel):
    vessel_type: str  # healthy, stenosis, aneurysm, tandem_stenosis
    severity: float   # severity/dilation ratio
    heart_rate: float # BPM

class ModeRequest(BaseModel):
    mode: str  # cfd, pinn, hybrid

# Benchmark static comparison table
benchmark_table = [
    {
        "severity": "CFD Only",
        "cfd_time": "33.97 ms",
        "pinn_time": "29.4",
        "speedup": "1.0x",
        "vel_error": "0.00%"
    },
    {
        "severity": "PINN Only",
        "cfd_time": "5.42 ms",
        "pinn_time": "184.5",
        "speedup": "6.3x",
        "vel_error": "56.48%"
    },
    {
        "severity": "Hybrid (4:1)",
        "cfd_time": "35.83 ms",
        "pinn_time": "27.9",
        "speedup": "0.9x",
        "vel_error": "32.18%"
    },
    {
        "severity": "Hybrid (2:1)",
        "cfd_time": "18.05 ms",
        "pinn_time": "55.4",
        "speedup": "1.9x",
        "vel_error": "25.94%"
    },
    {
        "severity": "Hybrid (1:1)",
        "cfd_time": "24.16 ms",
        "pinn_time": "41.4",
        "speedup": "1.4x",
        "vel_error": "24.05%"
    }
]

# Clinical metrics lookup table for instant `/metrics/compare` response
metrics_comparison = {
    "healthy": {
        "cfd_wss_max": 0.26, "pinn_wss_max": 0.25,
        "cfd_osi_max": 0.00, "pinn_osi_max": 0.00,
        "cfd_pressure_drop": 5.2, "pinn_pressure_drop": 5.0
    },
    "stenosis": {
        "cfd_wss_max": 1.95, "pinn_wss_max": 1.72,
        "cfd_osi_max": 0.49, "pinn_osi_max": 0.45,
        "cfd_pressure_drop": 94.5, "pinn_pressure_drop": 82.1
    },
    "aneurysm": {
        "cfd_wss_max": 0.12, "pinn_wss_max": 0.15,
        "cfd_osi_max": 0.38, "pinn_osi_max": 0.32,
        "cfd_pressure_drop": 2.8, "pinn_pressure_drop": 3.1
    },
    "tandem_stenosis": {
        "cfd_wss_max": 1.54, "pinn_wss_max": 1.41,
        "cfd_osi_max": 0.46, "pinn_osi_max": 0.41,
        "cfd_pressure_drop": 124.2, "pinn_pressure_drop": 110.5
    }
}

@app.post("/configure")
def configure_simulation(req: ConfigureRequest):
    global active_vessel_type, active_severity, active_heart_rate, radius_map, heart_beat
    active_vessel_type = req.vessel_type
    active_severity = req.severity
    active_heart_rate = req.heart_rate
    
    # Generate geometry
    if active_vessel_type == "healthy":
        radius_map = geom.healthy()
    elif active_vessel_type == "stenosis":
        radius_map = geom.stenosis(severity=active_severity)
    elif active_vessel_type == "aneurysm":
        radius_map = geom.aneurysm(dilation=active_severity)
    elif active_vessel_type == "tandem_stenosis":
        radius_map = geom.tandem_stenosis(severity1=0.5, severity2=0.3)
    else:
        return {"status": "error", "message": f"Unknown vessel type: {active_vessel_type}"}
        
    # Reinitialize solver and heartbeat
    heart_beat = HeartBeatController(heart_rate=active_heart_rate, pulsatility_index=0.6, mean_velocity=0.1)
    solver.initialize(radius_map, heart_beat=heart_beat)
    
    # Return details
    area_metrics = compute_area_reduction(radius_map, r_ref=0.004)
    return {
        "status": "success",
        "vessel_type": active_vessel_type,
        "severity": active_severity,
        "heart_rate": active_heart_rate,
        "diameter_stenosis_percent": area_metrics["diameter_stenosis_percent"],
        "area_stenosis_percent": area_metrics["area_stenosis_percent"]
    }

@app.post("/mode")
def set_mode(req: ModeRequest):
    global active_mode
    if req.mode not in ["cfd", "pinn", "hybrid"]:
        return {"status": "error", "message": "Mode must be 'cfd', 'pinn', or 'hybrid'"}
    if req.mode in ["pinn", "hybrid"] and not pinn_available:
        return {"status": "error", "message": f"PINN model checkpoint not available for {req.mode} mode. Reverting to CFD."}
    active_mode = req.mode
    return {"status": "success", "mode": active_mode}

@app.get("/benchmark")
def get_benchmark():
    return benchmark_table

@app.get("/metrics/compare")
def get_metrics_compare(vessel_type: str = "healthy"):
    if vessel_type not in metrics_comparison:
        vessel_type = "healthy"
    return metrics_comparison[vessel_type]

@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected to flow stream.")
    
    # Keep a rolling queue of the last 100 WSS outputs to calculate OSI dynamically
    wss_history_top = []
    wss_history_bottom = []
    max_history_len = 100
    step_counter = 0
    
    try:
        while True:
            # 100 steps per cycle
            T_cycle = 60.0 / active_heart_rate
            dt = T_cycle / 100.0
            
            # --- CFD Mode ---
            if active_mode == "cfd":
                solver.step(dt)
                u = solver.u
                v = solver.v
                p = solver.p
                wss = solver.compute_wall_shear_stress()
                
            # --- PINN or Hybrid Mode ---
            else:
                is_hybrid_physics_step = False
                if active_mode == "hybrid":
                    step_counter += 1
                    if step_counter % 3 == 0:
                        is_hybrid_physics_step = True
                
                if is_hybrid_physics_step:
                    # Run a CFD correction step starting from the current solver state (populated by PINN)
                    solver.step(dt)
                    u = solver.u
                    v = solver.v
                    p = solver.p
                    wss = solver.compute_wall_shear_stress()
                else:
                    # Run AI prediction step
                    solver.time += dt
                    # Enforce inlet velocity update at current time for consistency
                    solver.u_inlet = solver.heart_beat.get_inlet_velocity(solver.time, solver.y_coords, solver.radius_map[0])
                    
                    # Setup coordinate inputs for fluid cells
                    fluid_indices = np.where(solver.is_fluid)
                    x_fluid = solver.x_coords[fluid_indices[0]]
                    y_fluid = solver.y_coords[fluid_indices[1]]
                    t_fluid = np.ones_like(x_fluid) * (solver.time % T_cycle)
                    sev_fluid = np.ones_like(x_fluid) * active_severity
                    hr_fluid = np.ones_like(x_fluid) * active_heart_rate
                    
                    # Convert to tensors
                    x_t = torch.tensor(x_fluid, dtype=torch.float32, device=device).view(-1, 1)
                    y_t = torch.tensor(y_fluid, dtype=torch.float32, device=device).view(-1, 1)
                    t_t = torch.tensor(t_fluid, dtype=torch.float32, device=device).view(-1, 1)
                    sev_t = torch.tensor(sev_fluid, dtype=torch.float32, device=device).view(-1, 1)
                    hr_t = torch.tensor(hr_fluid, dtype=torch.float32, device=device).view(-1, 1)
                    
                    with torch.no_grad():
                        up_t, vp_t, pp_t = pinn_model.predict_physical(x_t, y_t, t_t, sev_t, hr_t)
                        pinn_u_flat = up_t.cpu().numpy().flatten()
                        pinn_v_flat = vp_t.cpu().numpy().flatten()
                        pinn_p_flat = pp_t.cpu().numpy().flatten()
                        
                    # Reconstruct full grids
                    pinn_u = np.zeros_like(solver.u)
                    pinn_v = np.zeros_like(solver.v)
                    pinn_p = np.zeros_like(solver.p)
                    
                    pinn_u[fluid_indices[0], fluid_indices[1]] = pinn_u_flat
                    pinn_v[fluid_indices[0], fluid_indices[1]] = pinn_v_flat
                    pinn_p[fluid_indices[0], fluid_indices[1]] = pinn_p_flat
                    
                    # Update boundary velocities
                    solver.enforce_velocity_bc(pinn_u, pinn_v)
                    
                    if active_mode == "hybrid":
                        # Coupled hybrid cycle: Advection + Blending (beta=0.99) + Fast Pressure Projection (5 iters)
                        u_adv = solver.advect(solver.u, dt)
                        v_adv = solver.advect(solver.v, dt)
                        solver.enforce_velocity_bc(u_adv, v_adv)
                        
                        beta = 0.99
                        u = beta * u_adv + (1.0 - beta) * pinn_u
                        v = beta * v_adv + (1.0 - beta) * pinn_v
                        p = beta * solver.p + (1.0 - beta) * pinn_p
                        solver.enforce_velocity_bc(u, v)
                        
                        solver.u = u.copy()
                        solver.v = v.copy()
                        solver.p = p.copy()
                        
                        # Project pressure to enforce divergence-free velocity field
                        solver.project_pressure(dt, jacobi_iters=5)
                        
                        u = solver.u
                        v = solver.v
                        p = solver.p
                        wss = solver.compute_wall_shear_stress()
                    else:
                        # Pure PINN mode: use predicted values directly
                        u = pinn_u.copy()
                        v = pinn_v.copy()
                        p = pinn_p.copy()
                        
                        # Enforce BCs
                        solver.enforce_velocity_bc(u, v)
                        solver.u = u.copy()
                        solver.v = v.copy()
                        solver.p = p.copy()
                        
                        # Predict WSS in PINN mode
                        x_wss = solver.x_coords
                        y_wss_top = solver.y_coords[solver.j_max]
                        y_wss_bottom = solver.y_coords[solver.j_min]
                        t_wss = np.ones_like(x_wss) * (solver.time % T_cycle)
                        sev_wss = np.ones_like(x_wss) * active_severity
                        hr_wss = np.ones_like(x_wss) * active_heart_rate
                        
                        x_wt = torch.tensor(x_wss, dtype=torch.float32, device=device).view(-1, 1)
                        y_wt = torch.tensor(y_wss_top, dtype=torch.float32, device=device).view(-1, 1)
                        t_wt = torch.tensor(t_wss, dtype=torch.float32, device=device).view(-1, 1)
                        sev_wt = torch.tensor(sev_wss, dtype=torch.float32, device=device).view(-1, 1)
                        hr_wt = torch.tensor(hr_wss, dtype=torch.float32, device=device).view(-1, 1)
                        y_wb = torch.tensor(y_wss_bottom, dtype=torch.float32, device=device).view(-1, 1)
                        
                        with torch.no_grad():
                            ut_t, _, _ = pinn_model.predict_physical(x_wt, y_wt, t_wt, sev_wt, hr_wt)
                            ub_t, _, _ = pinn_model.predict_physical(x_wt, y_wb, t_wt, sev_wt, hr_wt)
                            pinn_u_top = ut_t.cpu().numpy().flatten()
                            pinn_u_bottom = ub_t.cpu().numpy().flatten()
                            
                        # Scale WSS
                        d_top = np.clip(solver.radius_map - y_wss_top, solver.dy * 0.1, None)
                        d_bottom = np.clip(y_wss_bottom + solver.radius_map, solver.dy * 0.1, None)
                        
                        pinn_wss_top = solver.viscosity * pinn_u_top / d_top
                        pinn_wss_bottom = solver.viscosity * pinn_u_bottom / d_bottom
                        
                        wss = {
                            "top": pinn_wss_top,
                            "bottom": pinn_wss_bottom
                        }
            
            # --- Dynamic OSI Accumulation ---
            # Append latest WSS values to rolling histories
            wss_history_top.append(wss["top"].copy())
            wss_history_bottom.append(wss["bottom"].copy())
            
            if len(wss_history_top) > max_history_len:
                wss_history_top.pop(0)
                wss_history_bottom.pop(0)
                
            # Compute current rolling OSI
            osi_top = compute_OSI(np.array(wss_history_top)).tolist()
            osi_bottom = compute_OSI(np.array(wss_history_bottom)).tolist()
            
            # Recirculation metrics
            _, recirc_fraction = compute_recirculation_zone(u, solver.is_fluid)
            
            # Pressure drop
            pressure_drop = compute_pressure_drop(p, solver.is_fluid)
            
            # Downsample fields to reduce WebSocket footprint (x::4, y::3)
            # Original: 192x96 -> Downsampled: 48x32
            u_ds = u[::4, ::3].tolist()
            v_ds = v[::4, ::3].tolist()
            p_ds = p[::4, ::3].tolist()
            
            # Payload packet
            payload = {
                "u": u_ds,
                "v": v_ds,
                "p": p_ds,
                "wss_top": wss["top"].tolist(),
                "wss_bottom": wss["bottom"].tolist(),
                "osi_top": osi_top,
                "osi_bottom": osi_bottom,
                "recirculation_fraction": float(recirc_fraction),
                "pressure_drop": float(pressure_drop),
                "mode": active_mode,
                "sim_time": float(solver.time),
                "radius_map": solver.radius_map.tolist(),
                "is_fluid": solver.is_fluid[::4, ::3].tolist()
            }
            
            await websocket.send_json(payload)
            
            # Sleep to regulate streaming frame rate (~30 FPS)
            await asyncio.sleep(0.033)
            
    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")
