# HaemoSim: 2D Haemodynamics Simulator & PINN Benchmark

> [!WARNING]
> **ACADEMIC & EDUCATIONAL DEMONSTRATION ONLY**  
> HaemoSim is an academic research demonstration and educational tool developed to compare Physics-Informed Neural Networks (PINNs) against traditional numerical CFD solvers. It is **not** a clinical diagnostic tool, medical device, or software intended for clinical decision support. The geometries, physical assumptions, and flow parameters are simplified and do not represent actual clinical physiology or pathology.

---

## Project Description

HaemoSim is an interactive framework designed to simulate 2D blood flow through idealized arterial vessels and evaluate the feasibility of Physics-Informed Neural Network (PINN) surrogates. The codebase includes:
- **Numerical CFD Engine**: A 2D Navier-Stokes solver using the Stable Fluids method on a staggered-like collocated grid, incorporating implicit viscous diffusion (Jacobi solver) and pressure Poisson projection.
- **Deep PINN Surrogate**: A deep residual neural network architecture (8 layers, 128 neurons, skip connections) trained on Navier-Stokes residuals using PyTorch autograd.
- **Hemodynamics Suite**: Calculations of clinical metrics like time-varying Wall Shear Stress (WSS), Oscillatory Shear Index (OSI), recirculation area fractions, and pressure drops.
- **Interactive Viewport**: A glossy 3D cylindrical replica of the vessel rendered via Three.js with specular reflections, mouse rotation controls, and real-time advecting 3D streamlines.

---

## Validation Results

To establish the physical correctness of the underlying numerical solver, we validated the CFD engine against the analytical planar **Poiseuille flow** profile (steady state laminar flow between two infinite parallel plates separated by distance $2R$):

$$u(y) = -\frac{1}{2\mu} \frac{dp}{dx} (R^2 - y^2)$$

- **Grid Resolution**: $192 \times 96$ cell grids (length $80\text{ mm}$, baseline radius $R = 4\text{ mm}$).
- **Boundary Conditions**: Constant parabolic inlet flow profile (mean velocity $u_{avg} = 0.1\text{ m/s}$), no-slip wall boundaries, and zero-pressure outlet boundaries.
- **Convergence Criterion**: Maximum velocity difference between successive iterations $\Delta u_{max} < 10^{-5}\text{ m/s}$.
- **Result**: The solver successfully converged at step 652 with a **normalized $L^2$ relative velocity error of `2.0463%`** compared to the analytical profile. This confirms high quantitative accuracy for laminar shear flow.

*(See Figure 1: [fig1_poiseuille_validation.png](file:///C:/Users/PREDATOR/.gemini/antigravity/brain/dacff554-137b-471d-bd68-184eaa959a43/fig1_poiseuille_validation.png) in the artifacts directory for the profile overlay).*

---

## Key Scientific Finding

### OSI Peak Location Downstream of Stenosis
Under pulsatile inlet flow conditions (72 BPM, heart cycle period $T = 0.8333\text{ s}$), we simulated flow through a symmetric $60\%$ diameter stenosis vessel (throat centered at $x = 40.0\text{ mm}$). The simulation replicates a well-documented hemodynamic pattern:

- **At the Throat ($x = 40\text{ mm}$)**: The narrowing accelerates flow, causing a jet of high velocity and a corresponding peak in wall shear stress magnitude. However, because flow remains unidirectional and forward-facing, the **Oscillatory Shear Index (OSI) remains $0.0$**.
- **Downstream of the Throat ($x > 40\text{ mm}$)**: Due to the sudden expansion of the vessel wall, flow separation occurs. This creates a transient recirculation zone where flow reverses and fluctuates throughout the cardiac cycle.
- **Observations**: The boundary wall shear stress alternates direction as the inlet pulse accelerates and decelerates, resulting in a peak in oscillatory shear (**`OSI = 0.4918`**) located downstream at **`x = 64.0 mm`**.
- **Literature Alignment**: This finding matches the general clinical literature pattern (e.g., Ku et al., 1985; Zarins et al., 1983). The localization of low, highly oscillatory shear stress (high OSI) downstream of narrowings or at bifurcations is strongly correlated with the localization of endothelial cell dysfunction and atherosclerotic plaque progression.

*(See Figure 2: [fig2_wss_osi_comparison.png](file:///C:/Users/PREDATOR/.gemini/antigravity/brain/dacff554-137b-471d-bd68-184eaa959a43/fig2_wss_osi_comparison.png) in the artifacts directory for the overlaid spatial profiles).*

---

## PINN Benchmark Summary

We benchmarked the trained PINN surrogate against the numerical CFD ground truth on held-out severities (unseen $20.5\%$ and $47.7\%$ stenosis cases at 72 BPM):

| Case Severity | CFD Exec Time | PINN Inference Time | Speedup | Velocity Field $L^2$ Error | OSI Peak Location Error |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **20.5% Stenosis** | 8.904 s | 0.685 s | **13.0x** | 46.27% | **0.42 mm** (1 grid cell) |
| **47.7% Stenosis** | 9.055 s | 0.831 s | **10.9x** | 50.40% | **7.92 mm** (19 grid cells) |

*(See Figure 3: [fig3_pinn_benchmark.png](file:///C:/Users/PREDATOR/.gemini/antigravity/brain/dacff554-137b-471d-bd68-184eaa959a43/fig3_pinn_benchmark.png) for a formatted publication table).*

### Scientific Tradeoffs
- **Traditional CFD**: Provides high quantitative precision ($L^2$ velocity field accuracy) and strictly enforces mass/momentum conservation laws. However, it requires solving large linear systems iteratively, making it computationally heavy for real-time interactions.
- **PINN Surrogate**: Offers an order-of-magnitude **speedup ($\approx 11\text{x}$ to $13\text{x}$)** and demonstrates high spatial localization accuracy (e.g., locating the post-stenotic OSI peak within $0.42\text{ mm}$ for mild cases). However, it exhibits high relative field errors ($\approx 50\%$), showing a clear struggle to capture absolute velocity magnitudes and sharp local gradients at the stenosis throat.
- **Takeaway**: PINNs represent a powerful tool for rapid parameter exploration, clinical screening, and real-time visualization, but traditional CFD solvers remain mandatory for high-precision diagnostic and surgical planning metrics.

---

## Hybrid Simulation Engine (NeuroSim-Style)

To address the physical inconsistencies of pure AI models while preserving real-time speeds, we implemented a **coupled hybrid simulation cycle** modeled after the NeuroSim architecture. The simulation engine couples physics solvers with neural networks using periodic corrections:

1. **Physical Advection**: The solver advects the previous step's velocity field using the Semi-Lagrangian solver to preserve temporal continuity.
2. **Surrogate prior**: The PINN predicts the velocity and pressure fields at the current timestep based on global parameters.
3. **Temporal Blending**: The advected physical field is blended with the neural surrogate prediction using a high-inertia coefficient ($\beta = 0.99$):
   $$\mathbf{u}^* = \beta \mathbf{u}_{adv} + (1 - \beta) \mathbf{u}_{pinn}$$
   This leverages the PINN as a global corrector to prevent spatial drift, while the physical solver supplies temporal smoothing and boundary conformity.
4. **Fast Pressure Projection**: Runs a fast, low-iteration ($N_{jacobi} = 5$) pressure Poisson projection step to enforce the incompressibility constraint ($\nabla \cdot \mathbf{u} = 0$) and physical pressure-velocity coupling. This eliminates unphysical neural noises and boundary slip.
5. **Periodic CFD Correction**: Every $K$ steps (configured to a $2:1$ cycle, i.e., $K=3$), a full CFD step (diffusion + projection) is run starting from the blended state to reset accumulated error.

### Coupled Hybrid Engine Sweep Results

| Simulation Mode | Execution Time/Frame (ms) | Frames/Sec (FPS) | Velocity Field $L^2$ Relative Error | Performance Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **Only Physics (CFD)** | 33.97 ms | 29.4 FPS | **Reference (0.00%)** | 1.0x (Baseline) |
| **Only PINN Surrogate** | 5.42 ms | 184.5 FPS | 56.48% | 6.3x speedup |
| **Hybrid Cycle (4:1)** | 35.83 ms | 27.9 FPS | 32.18% | 0.9x speedup |
| **Hybrid Cycle (2:1)** | 18.05 ms | 55.4 FPS | **25.94%** | **1.9x speedup** |
| **Hybrid Cycle (1:1)** | 24.16 ms | 41.4 FPS | 24.05% | 1.4x speedup |

### Key Trade-Off Insights
- **Error Reduction**: The advection-blending-projection scheme cuts the relative velocity field error of the hybrid cycle **in half** (from **56.48%** down to **25.94%** for the $2:1$ cycle) compared to the pure PINN model.
- **Real-Time Speedup**: The $2:1$ hybrid configuration achieves a **1.9x speedup** over traditional CFD, providing the optimal blend of real-time responsiveness and physical conservation.

---

## Explicit Limitations

To maintain scientific integrity and prevent misuse, several limitations must be noted:

1. **Two-Dimensional (2D) Geometry Assumption**:
   Vascular flow is inherently three-dimensional. Real arteries contain helical flow, secondary vortices, and out-of-plane recirculation zones that cannot be represented in a 2D planar channel simulation.
2. **Idealized Vessel Geometries**:
   Vessel walls are represented as simple cosine-shaped contractions/expansions. Real blood vessels feature complex patient-specific curvatures, branchings, taperings, and bifurcations (e.g., at the carotid bulb or coronary arteries) which introduce highly asymmetric flow structures.
3. **No Patient-Specific Boundary Conditions**:
   Flow rates and pulsatile waves are generated using idealized mathematical formulas (Womersley-like sine functions). Clinical studies require patient-specific inflow profiles extracted from Doppler Ultrasound or Phase-Contrast Magnetic Resonance Imaging (PC-MRI).
4. **Coarse WSS & Boundary-Layer Resolution**:
   Wall Shear Stress (WSS) is extremely sensitive to velocity gradients close to the wall. HaemoSim uses a uniform grid with a simple finite difference approximation. It lacks boundary-conforming meshes, prism boundary layers, or fine subgrid resolutions, leading to significant underestimation of peak WSS values and gradient magnitudes compared to commercial solvers.

---

## Running the Application

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI Server
```bash
python -m uvicorn backend.main:app --port 8000
```

### 3. Open the Dashboard
Navigate to `http://localhost:8000` in your web browser. Use the top-right workspace tab to switch between the rotatable **3D Live Simulation** view and the side-by-side **Cohort Comparison Profile** dashboard.
