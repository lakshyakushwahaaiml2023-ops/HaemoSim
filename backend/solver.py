"""
HaemoSim Blood Flow Solver
Academic/Educational Demonstration Tool. Not for clinical use or diagnostic decision making.

Implements 2D incompressible Navier-Stokes solver using the Stable Fluids method:
- Semi-Lagrangian advection
- Implicit viscous diffusion (Jacobi iteration)
- Pressure Poisson projection (Jacobi iteration)
"""

import numpy as np

class HeartBeatController:
    """
    HeartBeatController manages the transient pulsatile inlet flow conditions
    using a Womersley-like heartbeat waveform.
    
    Academic/Educational Demonstration Tool. Not for clinical use.
    """
    def __init__(
        self,
        heart_rate: float = 72.0,            # BPM
        pulsatility_index: float = 0.6,      # index (0 to 1) scaling pulse amplitude
        mean_velocity: float = 0.1,          # m/s average inflow velocity
        phase: float = 0.0                   # phase offset in radians
    ) -> None:
        self.heart_rate = heart_rate
        self.pulsatility_index = pulsatility_index
        self.mean_velocity = mean_velocity
        self.phase = phase

    def get_inlet_velocity_scaling(self, t: float) -> float:
        """
        Compute the average inlet velocity scale at time t.
        Formula: mean_velocity * (1 + pulsatility_index * sin(2 * pi * f * t - phase))
        """
        f = self.heart_rate / 60.0
        omega = 2.0 * np.pi * f
        return self.mean_velocity * (1.0 + self.pulsatility_index * np.sin(omega * t - self.phase))

    def get_inlet_velocity(self, t: float, y_coords: np.ndarray, radius_inlet: float) -> np.ndarray:
        """
        Compute the 1D spatial profile of u velocity at time t (parabolic shape scaled by pulsatility).
        """
        u_mean = self.get_inlet_velocity_scaling(t)
        profile = np.zeros_like(y_coords)
        mask = np.abs(y_coords) <= radius_inlet
        profile[mask] = 1.5 * u_mean * (1.0 - (y_coords[mask] / radius_inlet)**2)
        return profile

class BloodFlowSolver:
    """
    2D incompressible Navier-Stokes solver for blood flow simulation in vessels
    using the Stable Fluids method.
    
    Designed for academic demonstration and PINN benchmarking.
    """
    def __init__(self, Nx: int = 192, Ny: int = 96) -> None:
        # Spatial grid sizes
        self.Nx = Nx
        self.Ny = Ny
        
        # Physical domain properties (default healthy vessel)
        self.length = 0.08          # m (80 mm)
        self.radius_max = 0.004      # m (4 mm)
        
        # Grid parameters
        self.dx = self.length / self.Nx
        # Pad grid height by 20% to represent walls inside grid
        self.y_max = 1.2 * self.radius_max
        self.dy = (2.0 * self.y_max) / self.Ny
        
        # Grid coordinates (cell centers)
        self.x_coords = (np.arange(self.Nx) + 0.5) * self.dx
        self.y_coords = -self.y_max + (np.arange(self.Ny) + 0.5) * self.dy
        
        # Physical fluid properties (blood)
        self.density = 1060.0       # kg/m^3
        self.viscosity = 0.0035     # Pa.s (dynamic viscosity)
        
        # Fields
        self.u = np.zeros((self.Nx, self.Ny))
        self.v = np.zeros((self.Nx, self.Ny))
        self.p = np.zeros((self.Nx, self.Ny))
        
        # Time and pulsatile control
        self.time = 0.0
        self.use_pulsatile = True
        self.heart_beat = HeartBeatController()
        
        # Geometry masking
        self.is_fluid = np.zeros((self.Nx, self.Ny), dtype=bool)
        self.radius_map = np.zeros(self.Nx)
        self.u_inlet = np.zeros(self.Ny)
        
        # Indices of boundary fluid cells
        self.j_min = np.zeros(self.Nx, dtype=int)
        self.j_max = np.zeros(self.Nx, dtype=int)

    def initialize(
        self, 
        radius_map: np.ndarray, 
        heart_beat: HeartBeatController = None,
        inlet_velocity_profile: np.ndarray = None
    ) -> None:
        """
        Initialize the solver with a custom vessel geometry (radius_map), optional heartbeat controller,
        and optional static inlet velocity profile.
        
        Parameters:
            radius_map (np.ndarray): Array of shape (Nx,) defining the radius of the vessel at each x column.
            heart_beat (HeartBeatController): Optional custom heartbeat controller.
            inlet_velocity_profile (np.ndarray): Optional array for overriding with a static inlet u-velocity.
        """
        self.radius_map = radius_map.copy()
        self.time = 0.0
        
        if heart_beat is not None:
            self.heart_beat = heart_beat
            
        if inlet_velocity_profile is not None:
            self.use_pulsatile = False
            self.u_inlet = inlet_velocity_profile.copy()
        else:
            self.use_pulsatile = True
            # Set initial inlet profile using heartbeat at t=0
            self.u_inlet = self.heart_beat.get_inlet_velocity(0.0, self.y_coords, self.radius_map[0])
        
        # Determine fluid vs solid cells
        self.is_fluid = np.zeros((self.Nx, self.Ny), dtype=bool)
        for i in range(self.Nx):
            self.is_fluid[i, :] = np.abs(self.y_coords) <= self.radius_map[i]
            
        # Find the top and bottom boundary fluid cells for each column
        for i in range(self.Nx):
            fluid_indices = np.where(self.is_fluid[i])[0]
            if len(fluid_indices) > 0:
                self.j_min[i] = fluid_indices[0]
                self.j_max[i] = fluid_indices[-1]
            else:
                self.j_min[i] = self.Ny // 2
                self.j_max[i] = self.Ny // 2
            
        # Reset fields and seed flow in fluid cells
        self.u = np.zeros((self.Nx, self.Ny))
        self.v = np.zeros((self.Nx, self.Ny))
        self.p = np.zeros((self.Nx, self.Ny))
        
        for i in range(self.Nx):
            self.u[i, :] = self.u_inlet
            
        # Apply fluid mask (enforce no-slip immediately)
        self.enforce_velocity_bc(self.u, self.v)

    def enforce_velocity_bc(self, u: np.ndarray, v: np.ndarray) -> None:
        """
        Enforce boundary conditions for velocity field:
        - No-slip (u = v = 0) at solid cells (outside radius_map)
        - Prescribed inlet velocity at x=0
        - Outflow (zero-gradient) boundary condition at x=L
        """
        # Solid walls (masked out cells)
        u[~self.is_fluid] = 0.0
        v[~self.is_fluid] = 0.0
        
        # Prescribed inlet at i = 0 (fluid cells only)
        inlet_mask = self.is_fluid[0]
        u[0, inlet_mask] = self.u_inlet[inlet_mask]
        u[0, ~inlet_mask] = 0.0
        v[0, :] = 0.0
        
        # Outflow at i = -1 (zero-gradient)
        u[-1, :] = u[-2, :]
        v[-1, :] = v[-2, :]

    def enforce_pressure_bc(self, p: np.ndarray) -> None:
        """
        Enforce boundary conditions for pressure field:
        - Neumann (dp/dn = 0) at solid boundaries (copy from adjacent fluid cell)
        - Neumann (dp/dx = 0) at the inlet (p[0, j] = p[1, j])
        - Dirichlet (p = 0) at the outlet (p[-1, j] = 0)
        """
        # Copy fluid pressure to adjacent solid cells in each column
        for i in range(self.Nx):
            j_min_idx = self.j_min[i]
            j_max_idx = self.j_max[i]
            
            p[i, :j_min_idx] = p[i, j_min_idx]
            p[i, j_max_idx + 1:] = p[i, j_max_idx]
            
        # Inlet: dp/dx = 0
        p[0, :] = p[1, :]
        
        # Outlet: p = 0
        p[-1, :] = 0.0

    def _bilinear_interpolate(self, field: np.ndarray, i_float: np.ndarray, j_float: np.ndarray) -> np.ndarray:
        """
        Helper to perform vectorized bilinear interpolation on the grid.
        """
        i0 = np.floor(i_float).astype(int)
        i1 = i0 + 1
        j0 = np.floor(j_float).astype(int)
        j1 = j0 + 1
        
        # Clip to boundary indices
        np.clip(i0, 0, self.Nx - 1, out=i0)
        np.clip(i1, 0, self.Nx - 1, out=i1)
        np.clip(j0, 0, self.Ny - 1, out=j0)
        np.clip(j1, 0, self.Ny - 1, out=j1)
        
        wx = i_float - i0
        wy = j_float - j0
        
        # Interpolated values
        val = (1.0 - wx) * (1.0 - wy) * field[i0, j0] + \
              wx * (1.0 - wy) * field[i1, j0] + \
              (1.0 - wx) * wy * field[i0, j1] + \
              wx * wy * field[i1, j1]
              
        return val

    def advect(self, field: np.ndarray, dt: float) -> np.ndarray:
        """
        Semi-Lagrangian advection step: trace velocity backwards in time
        and interpolate the field value.
        """
        # Create coordinates meshgrid
        X, Y = np.meshgrid(self.x_coords, self.y_coords, indexing='ij')
        
        # Trace backtrace trajectory
        X_prev = X - dt * self.u
        Y_prev = Y - dt * self.v
        
        # Convert to fractional index coordinates
        i_float = X_prev / self.dx - 0.5
        j_float = (Y_prev + self.y_max) / self.dy - 0.5
        
        return self._bilinear_interpolate(field, i_float, j_float)

    def diffuse(self, u_advected: np.ndarray, v_advected: np.ndarray, dt: float) -> None:
        """
        Solve implicit viscous diffusion equation using Jacobi iterations:
        (I - nu * dt * Laplacian) u_new = u_advected
        """
        nu = self.viscosity / self.density
        
        # Coefficient factor for denominator in Jacobi
        c = 1.0 + 2.0 * nu * dt / self.dx**2 + 2.0 * nu * dt / self.dy**2
        
        # Only solve in fluid interior (exclude boundary columns x=0 and x=L)
        active_interior = self.is_fluid[1:-1, 1:-1]
        
        # Jacobi iterations (40 is sufficient for standard time-steps)
        for _ in range(40):
            u_left = self.u[:-2, 1:-1]
            u_right = self.u[2:, 1:-1]
            u_down = self.u[1:-1, :-2]
            u_up = self.u[1:-1, 2:]
            
            u_next_interior = (u_advected[1:-1, 1:-1] + nu * dt * (
                (u_left + u_right) / self.dx**2 + (u_down + u_up) / self.dy**2
            )) / c
            
            self.u[1:-1, 1:-1] = np.where(active_interior, u_next_interior, self.u[1:-1, 1:-1])
            
            v_left = self.v[:-2, 1:-1]
            v_right = self.v[2:, 1:-1]
            v_down = self.v[1:-1, :-2]
            v_up = self.v[1:-1, 2:]
            
            v_next_interior = (v_advected[1:-1, 1:-1] + nu * dt * (
                (v_left + v_right) / self.dx**2 + (v_down + v_up) / self.dy**2
            )) / c
            
            self.v[1:-1, 1:-1] = np.where(active_interior, v_next_interior, self.v[1:-1, 1:-1])
            
            # Update boundaries during iterations
            self.enforce_velocity_bc(self.u, self.v)

    def project_pressure(self, dt: float, jacobi_iters: int = 80) -> None:
        """
        Pressure Poisson projection step: solve Poisson equation for pressure
        and subtract the pressure gradient to make the velocity field divergence-free.
        """
        # 1. Compute velocity divergence
        u_left = self.u[:-2, 1:-1]
        u_right = self.u[2:, 1:-1]
        v_down = self.v[1:-1, :-2]
        v_up = self.v[1:-1, 2:]
        
        div = np.zeros((self.Nx, self.Ny))
        div[1:-1, 1:-1] = (u_right - u_left) / (2.0 * self.dx) + (v_up - v_down) / (2.0 * self.dy)
        div[~self.is_fluid] = 0.0
        
        # 2. Solve pressure Poisson equation: Laplacian(p) = (rho / dt) * div
        f = (self.density / dt) * div
        
        c_p = 2.0 / self.dx**2 + 2.0 / self.dy**2
        active_interior = self.is_fluid[1:-1, 1:-1]
        
        # Use a new zero pressure guess or start from previous pressure field to accelerate
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
        
        # 3. Correct velocity field: u_new = u - (dt / rho) * grad(p)
        p_left = self.p[:-2, 1:-1]
        p_right = self.p[2:, 1:-1]
        p_down = self.p[1:-1, :-2]
        p_up = self.p[1:-1, 2:]
        
        grad_p_x = (p_right - p_left) / (2.0 * self.dx)
        grad_p_y = (p_up - p_down) / (2.0 * self.dy)
        
        self.u[1:-1, 1:-1] -= (dt / self.density) * grad_p_x * active_interior
        self.v[1:-1, 1:-1] -= (dt / self.density) * grad_p_y * active_interior
        
        # Finally re-enforce boundaries
        self.enforce_velocity_bc(self.u, self.v)

    def step(self, dt: float) -> None:
        """
        Advance simulator by time-step dt.
        """
        self.time += dt
        
        if self.use_pulsatile:
            # Update dynamic inlet velocity based on current simulation time
            self.u_inlet = self.heart_beat.get_inlet_velocity(self.time, self.y_coords, self.radius_map[0])
            
        # 1. Advect velocity field
        u_advected = self.advect(self.u, dt)
        v_advected = self.advect(self.v, dt)
        
        self.enforce_velocity_bc(u_advected, v_advected)
        
        # 2. Viscous diffusion
        self.diffuse(u_advected, v_advected, dt)
        
        # 3. Pressure projection
        self.project_pressure(dt)

    def get_state(self) -> dict:
        """
        Get the current simulation fields and configuration.
        """
        return {
            "x": self.x_coords,
            "y": self.y_coords,
            "u": self.u,
            "v": self.v,
            "p": self.p,
            "is_fluid": self.is_fluid
        }

    def compute_wall_shear_stress(self) -> dict:
        """
        Compute Wall Shear Stress (WSS) magnitude along top and bottom vessel walls.
        
        Returns:
            dict: Containing arrays 'top' and 'bottom' of shape (Nx,) representing WSS in Pascals.
        """
        # Calculate wall distance from boundary fluid cell to actual wall boundary
        # top wall boundary is at +radius_map
        d_top = np.clip(self.radius_map - self.y_coords[self.j_max], self.dy * 0.1, None)
        # bottom wall boundary is at -radius_map
        d_bottom = np.clip(self.y_coords[self.j_min] + self.radius_map, self.dy * 0.1, None)
        
        # Extract velocity values at boundary cells (signed)
        u_top = self.u[np.arange(self.Nx), self.j_max]
        u_bottom = self.u[np.arange(self.Nx), self.j_min]
        
        # WSS = mu * du/dn
        wss_top = self.viscosity * u_top / d_top
        wss_bottom = self.viscosity * u_bottom / d_bottom
        
        return {
            "top": wss_top,
            "bottom": wss_bottom
        }

    def compute_womersley_number(self) -> float:
        """
        Calculate the Womersley number alpha for the vessel.
        alpha = R * sqrt(omega * density / viscosity)
        
        Physically, the Womersley number represents the ratio of transient inertial forces
        to viscous forces:
        - alpha < 1: viscous forces dominate, flow is quasi-steady (in-phase with pressure gradient).
        - alpha > 1: inertial forces dominate, flow is transient (phase lag, flatter profile).
        """
        f = self.heart_beat.heart_rate / 60.0
        omega = 2.0 * np.pi * f
        R = self.radius_max
        alpha = R * np.sqrt(omega * self.density / self.viscosity)
        return alpha
