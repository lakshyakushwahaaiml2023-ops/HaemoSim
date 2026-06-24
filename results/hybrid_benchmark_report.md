# Hybrid Simulation Architecture Benchmark Report

This benchmark compares the performance, execution speed, and velocity field accuracy across three simulation configurations for a 60% severity stenosis vessel under pulsatile flow:
1. **Only Physics (CFD)**: High-precision finite difference numerical solver.
2. **Only PINN Surrogate**: High-speed coordinate-based neural network model.
3. **Hybrid Cycle (4:1, 2:1, 1:1)**: Coupled execution where every $K$-th step runs the numerical CFD step starting from the PINN-predicted state to correct numerical errors.

## Benchmark Performance Stats

| Simulation Mode | Total Time (100 steps) | Time/Frame (ms) | Frames/Sec (FPS) | Velocity Field $L^2$ Relative Error |
| :--- | :---: | :---: | :---: | :---: |
| **Only Physics (CFD)** | 3.397 s | 33.97 ms | 29.4 FPS | **Reference (0.00%)** |
| **Only PINN Surrogate** | 0.542 s | 5.42 ms | 184.5 FPS | 56.48% |
| **Hybrid Cycle (4:1)** | 3.583 s | 35.83 ms | 27.9 FPS | 32.18% |
| **Hybrid Cycle (2:1)** | 1.805 s | 18.05 ms | 55.4 FPS | 25.94% |
| **Hybrid Cycle (1:1)** | 2.416 s | 24.16 ms | 41.4 FPS | 24.05% |

## Critical Technical Insights

1. **Velocity Field L2 Error Reduction**:
   - The pure PINN model has an average velocity $L^2$ error of **56.48%** compared to the CFD ground truth.
   - Increasing the frequency of numerical physics steps systematically reduces the L2 relative field error:
     - **4:1 cycle** (CFD correction once every 5 steps): **32.18%** error.
     - **2:1 cycle** (CFD correction once every 3 steps): **25.94%** error.
     - **1:1 cycle** (CFD correction once every 2 steps): **24.05%** error.
   - By running the **1:1 hybrid cycle**, the velocity field error drops significantly because the solver runs a full physical projection/diffusion correction step on every alternate frame. This ensures temporal consistency, enforces boundary wall constraints (no-slip), and wipes out spatial neural noises.

2. **FPS and Real-Time Interaction Tradeoff**:
   - Pure PINN runs at **184.5 FPS** (5.42 ms/frame).
   - Pure CFD runs at **29.4 FPS** (33.97 ms/frame).
   - The **Hybrid Cycle (1:1)** runs at **41.4 FPS** (24.16 ms/frame) — achieving a valuable balance of physical consistency and speedup.
