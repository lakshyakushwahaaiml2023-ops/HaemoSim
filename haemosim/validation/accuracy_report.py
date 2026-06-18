"""
accuracy_report.py — HaemoSim hybrid-surrogate validation artifact.

Usage (from the haemosim/ package root):
    python -m validation.accuracy_report
or:
    python haemosim/validation/accuracy_report.py

What it does
------------
1. Runs the CirculatorySimulator for 60 s using step-by-step RK4 as ground truth.
2. Runs a ControlledHybridRunner at surrogate_fraction in {0.25, 0.50, 0.75}.
   Both GT and hybrid use the *same* fixed-step RK4 integrator, so speedup
   differences are purely from fewer RHS evaluations, not solver differences.

   The hybrid runner:
     - Physics frames  : full 4-stage RK4 step  (4 RHS calls)
     - Surrogate frames: first-order-hold extrapolation from last stored
       derivative  (0 new RHS calls — simulates what a trained ML surrogate
       would do: predict in O(1) vs integrate)

   Pattern per period P=4:
     25 % surrogate  ->  [phys, phys, phys, sur]  (n_phys=3, n_sur=1)
     50 % surrogate  ->  [phys, phys, sur,  sur]  (n_phys=2, n_sur=2)
     75 % surrogate  ->  [phys, sur,  sur,  sur]  (n_phys=1, n_sur=3)

3. For each hybrid run reports:
     * Mean-absolute-error in aortic pressure vs ground truth (mmHg)
     * Mean-absolute-error in cardiac output vs ground truth (L/min)
     * Wall-clock speedup over full physics (integration time only)
     * Actual % of frames served by surrogate
4. Prints a formatted summary table to stdout.
5. Saves a 4-panel figure to validation/accuracy_report.png.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless — safe on servers / Windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Make the package importable whether invoked as a module or run directly.
# ---------------------------------------------------------------------------
_HERE     = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent        # …/haemosim/
_PROJ_ROOT = _PKG_ROOT.parent   # …/HaemoSim/
for _p in (_PKG_ROOT, _PROJ_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    from haemosim.core.simulator import CirculatorySimulator, SimulationResult
except ImportError:
    from core.simulator import CirculatorySimulator, SimulationResult

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
DURATION           = 60.0     # seconds
DT                 = 0.005    # output / integration step (s) — 12 000 frames
SURROGATE_FRACTIONS = [0.25, 0.50, 0.75]
DISPLAY_WINDOW     = 5.0      # seconds of pressure trace shown in panel 1
OUTPUT_NAME        = "accuracy_report.png"

# Colour palette
_COL_GT  = "#1a1a2e"
_COL_025 = "#e94560"
_COL_050 = "#f5a623"
_COL_075 = "#0f9b8e"
_COLOURS = [_COL_025, _COL_050, _COL_075]
_LABELS  = ["25% surrogate", "50% surrogate", "75% surrogate"]


# ---------------------------------------------------------------------------
# Fixed-step RK4 helper
# ---------------------------------------------------------------------------
def _rk4_step(rhs, t: float, y: np.ndarray, dt: float):
    """
    Single fourth-order Runge–Kutta step.

    Returns
    -------
    y_new : np.ndarray  — state at t + dt
    k4    : np.ndarray  — final stage derivative (reused as extrapolation slope)
    """
    k1 = rhs(t,            y)
    k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = rhs(t + dt,        y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), k4


# ---------------------------------------------------------------------------
# Ground-truth runner (full RK4, every step)
# ---------------------------------------------------------------------------
def run_ground_truth(duration: float, dt: float) -> tuple[SimulationResult, float]:
    """
    Full physics run.  Uses step-by-step RK4 so that the timing baseline is
    directly comparable with the hybrid runner (same integrator, same dt).

    Returns
    -------
    result  : SimulationResult
    elapsed : wall-clock integration time (seconds)
    """
    sim = CirculatorySimulator()
    rhs = sim._rhs

    t_eval = np.arange(0.0, duration + 0.5 * dt, dt)
    n      = len(t_eval)
    y      = sim._initial_state()
    states = np.empty((n, len(y)), dtype=np.float64)
    states[0] = y

    t0 = time.perf_counter()
    for i in range(n - 1):
        y, _ = _rk4_step(rhs, t_eval[i], y, dt)
        states[i + 1] = y
    elapsed = time.perf_counter() - t0

    result = sim._build_result(t_eval, states)
    return result, elapsed


# ---------------------------------------------------------------------------
# Controlled hybrid runner
# ---------------------------------------------------------------------------
class ControlledHybridRunner:
    """
    Hybrid physics/surrogate runner with a fixed surrogate_fraction.

    Physics steps  : 4-stage RK4    (4 RHS evaluations per step)
    Surrogate steps: first-order-hold extrapolation from the last stored
                     RK4 derivative  (0 new RHS evaluations per step)

    This faithfully models what a trained ML surrogate would do:
    a fast forward pass (here O(1) array ops) replaces the ODE integration.
    """

    _PERIOD = 4   # repeating pattern length in frames

    def __init__(self, surrogate_fraction: float) -> None:
        if not 0.0 <= surrogate_fraction < 1.0:
            raise ValueError("surrogate_fraction must be in [0, 1)")
        self.surrogate_fraction = surrogate_fraction
        n_sur  = round(surrogate_fraction * self._PERIOD)
        n_phys = self._PERIOD - n_sur
        self._n_phys = max(n_phys, 1)   # always at least one physics step/period
        self._n_sur  = self._PERIOD - self._n_phys
        self.actual_fraction = self._n_sur / self._PERIOD

    # ------------------------------------------------------------------
    def run(
        self,
        duration: float,
        dt: float,
    ) -> tuple[SimulationResult, np.ndarray, float]:
        """
        Run the hybrid simulation.

        Returns
        -------
        result       : SimulationResult
        frame_source : ndarray of object, shape (n,), values 'physics'/'surrogate'
        elapsed      : wall-clock integration time (seconds)
                       — excludes _build_result post-processing
        """
        sim = CirculatorySimulator()
        rhs = sim._rhs

        t_eval  = np.arange(0.0, duration + 0.5 * dt, dt)
        n       = len(t_eval)
        P       = self._PERIOD
        n_phys  = self._n_phys

        y  = sim._initial_state()
        states = np.empty((n, len(y)), dtype=np.float64)
        states[0] = y

        frame_source = np.empty(n, dtype=object)
        frame_source[0] = "physics"

        # Seed the extrapolation slope with the initial derivative
        last_slope = rhs(t_eval[0], y)

        t0 = time.perf_counter()

        for i in range(n - 1):
            if (i % P) < n_phys:
                # ---- Physics step: full RK4 (4 RHS calls) ----------------
                y, last_slope = _rk4_step(rhs, t_eval[i], y, dt)
                frame_source[i + 1] = "physics"
            else:
                # ---- Surrogate step: first-order-hold (0 RHS calls) ------
                # Simulates ML surrogate inference: fast, no ODE evaluation.
                y = y + dt * last_slope
                frame_source[i + 1] = "surrogate"
            states[i + 1] = y

        elapsed = time.perf_counter() - t0

        result = sim._build_result(t_eval, states)
        return result, frame_source, elapsed


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def _cardiac_output_lpm(result: SimulationResult) -> np.ndarray:
    """Instantaneous cardiac output proxy in L/min (aortic segment flow)."""
    return np.maximum(result.flows[:, 0], 0.0) * 60.0 / 1000.0


def _compute_metrics(
    gt: SimulationResult,
    hybrid: SimulationResult,
    frame_source: np.ndarray,
    gt_time: float,
    hyb_time: float,
    actual_fraction: float,
) -> dict:
    mae_p  = float(np.mean(np.abs(hybrid.aortic_pressure - gt.aortic_pressure)))
    mae_co = float(np.mean(np.abs(_cardiac_output_lpm(hybrid) - _cardiac_output_lpm(gt))))
    return {
        "mae_pressure_mmhg": mae_p,
        "mae_co_lpm":        mae_co,
        "speedup":           gt_time / max(hyb_time, 1e-12),
        "surrogate_pct":     100.0 * actual_fraction,
        "gt_time_s":         gt_time,
        "hybrid_time_s":     hyb_time,
    }


# ---------------------------------------------------------------------------
# Console summary table (ASCII-only — safe on Windows cp1252)
# ---------------------------------------------------------------------------
def _print_table(fractions: list[float], metrics: list[dict], gt_time: float) -> None:
    col_w = [24, 14, 16, 12, 10, 10, 12]
    hdr = ["Run", "MAE P (mmHg)", "MAE CO (L/min)", "Speedup",
           "Sur %", "GT (s)", "Hybrid (s)"]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"

    def row(*cells):
        return "|" + "|".join(
            f" {str(c):<{w - 2}} " for c, w in zip(cells, col_w)
        ) + "|"

    print()
    print("=" * 100)
    print("  HaemoSim Hybrid-Surrogate Accuracy Report")
    print("  Integrator: fixed-step RK4, dt={:.4f} s".format(DT))
    print("  Surrogate:  first-order-hold extrapolation (0 RHS calls/surrogate step)")
    print("=" * 100)
    print(sep)
    print(row(*hdr))
    print(sep)
    print(row("Full physics (GT)", "0.000", "0.000", "1.00x",
              "0.0%", f"{gt_time:.2f}", f"{gt_time:.2f}"))
    for frac, m in zip(fractions, metrics):
        print(row(
            f"Hybrid {int(frac*100):d}% surrogate",
            f"{m['mae_pressure_mmhg']:.4f}",
            f"{m['mae_co_lpm']:.5f}",
            f"{m['speedup']:.2f}x",
            f"{m['surrogate_pct']:.1f}%",
            f"{m['gt_time_s']:.2f}",
            f"{m['hybrid_time_s']:.2f}",
        ))
    print(sep)
    print()


# ---------------------------------------------------------------------------
# 4-panel validation figure
# ---------------------------------------------------------------------------
def _make_figure(
    gt: SimulationResult,
    hybrids: list[tuple[SimulationResult, np.ndarray]],
    metrics: list[dict],
    output_path: Path,
) -> None:
    """
    Panel 1 (top-left):  Aortic-pressure traces overlaid, last 5 s.
    Panel 2 (top-right): |Error| in aortic pressure vs time (rolling mean).
    Panel 3 (bot-left):  Wall-clock speedup bar chart.
    Panel 4 (bot-right): Surrogate-frame usage over time (rolling %).
    """
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "figure.facecolor":  "#f7f8fc",
        "axes.facecolor":    "#f7f8fc",
    })

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#f7f8fc")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    t = gt.t

    # ---- Panel 1 : Aortic pressure traces (last DISPLAY_WINDOW seconds) ----
    win  = t >= (DURATION - DISPLAY_WINDOW)
    t_w  = t[win]
    ax1.plot(t_w, gt.aortic_pressure[win],
             color=_COL_GT, lw=2.2, label="Ground truth (full RK4)", zorder=5)
    for (hr, _), col, lbl in zip(hybrids, _COLOURS, _LABELS):
        ax1.plot(t_w, hr.aortic_pressure[win],
                 color=col, lw=1.5, alpha=0.88, ls="--", label=lbl)
    ax1.set_title(f"Aortic Pressure — last {DISPLAY_WINDOW:.0f} s",
                  fontsize=12, fontweight="bold", pad=8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Pressure (mmHg)")
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.75)
    ax1.set_xlim(t_w[0], t_w[-1])

    # ---- Panel 2 : |Error| vs time ----------------------------------------
    roll = max(1, int(0.8 / DT))   # ~1 cardiac cycle
    for (hr, _), col, lbl in zip(hybrids, _COLOURS, _LABELS):
        err    = np.abs(hr.aortic_pressure - gt.aortic_pressure)
        smooth = np.convolve(err, np.ones(roll) / roll, mode="same")
        ax2.plot(t, smooth, color=col, lw=1.4, alpha=0.9, label=lbl)
    ax2.set_title("|Error| in Aortic Pressure (rolling mean, ~1 cycle)",
                  fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("|dP| (mmHg)")
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.75)
    ax2.set_xlim(t[0], t[-1])
    ax2.set_ylim(bottom=0.0)

    # ---- Panel 3 : Speedup bar chart --------------------------------------
    x_labels = [f"{int(f*100)}%" for f in SURROGATE_FRACTIONS]
    speedups  = [m["speedup"] for m in metrics]
    bars = ax3.bar(x_labels, speedups, color=_COLOURS, width=0.48,
                   edgecolor="white", linewidth=1.3)
    ax3.axhline(1.0, color=_COL_GT, lw=1.8, ls=":", label="Full physics (1x)")
    for bar, sp in zip(bars, speedups):
        ax3.text(bar.get_x() + bar.get_width() / 2.0,
                 bar.get_height() + 0.05,
                 f"{sp:.2f}x",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax3.set_title("Wall-clock Speedup vs Full Physics\n"
                  "(integration time only — same RK4 integrator)",
                  fontsize=11, fontweight="bold", pad=8)
    ax3.set_xlabel("Surrogate fraction")
    ax3.set_ylabel("Speedup (x)")
    ax3.legend(loc="upper left", fontsize=9, framealpha=0.75)
    ax3.set_ylim(0.0, max(speedups) * 1.28 + 0.5)
    ax3.grid(axis="x", alpha=0)

    # ---- Panel 4 : Surrogate usage over time ------------------------------
    roll4 = max(1, int(10.0 / DT))   # 10-second rolling window
    for (_, fs), col, lbl in zip(hybrids, _COLOURS, _LABELS):
        is_sur = (fs == "surrogate").astype(float)
        frac_r = np.convolve(is_sur, np.ones(roll4) / roll4, mode="same")
        ax4.plot(t, frac_r * 100.0, color=col, lw=1.4, alpha=0.9, label=lbl)
    ax4.set_title("Surrogate Usage (10-s rolling %)",
                  fontsize=12, fontweight="bold", pad=8)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Surrogate frames (%)")
    ax4.set_ylim(-5, 105)
    ax4.set_xlim(t[0], t[-1])
    ax4.legend(loc="center right", fontsize=8, framealpha=0.75)

    # ---- Titles & captions ------------------------------------------------
    fig.suptitle(
        "HaemoSim — Hybrid Physics / Surrogate Accuracy & Speed Report",
        fontsize=14, fontweight="bold", color="#1a1a2e", y=1.02,
    )
    fig.text(
        0.99, -0.015,
        (f"Surrogate = first-order-hold extrapolation (0 RHS calls/step)  |  "
         f"Duration = {DURATION:.0f} s, dt = {DT} s  |  "
         f"Integrator: fixed-step RK4"),
        ha="right", fontsize=7, color="#555555",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  [figure] saved -> {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    n_frames = int(DURATION / DT)
    print()
    print("HaemoSim Accuracy Report")
    print(f"  duration         = {DURATION} s")
    print(f"  dt               = {DT} s  ({n_frames:,} frames)")
    print(f"  integrator       = fixed-step RK4")
    print(f"  surrogate method = first-order-hold (0 RHS calls/surrogate step)")
    print()

    # ---- 1. Ground truth ---------------------------------------------------
    print("[1/5] Running full physics (ground truth, all-RK4) ...", flush=True)
    gt_result, gt_time = run_ground_truth(DURATION, DT)
    p_lo = gt_result.aortic_pressure.min()
    p_hi = gt_result.aortic_pressure.max()
    sv   = gt_result.lv_volume.max() - gt_result.lv_volume.min()
    print(f"      done in {gt_time:.2f} s  |  "
          f"P_ao in [{p_lo:.1f}, {p_hi:.1f}] mmHg  |  "
          f"SV = {sv:.1f} mL")

    # ---- 2-4. Hybrid runs --------------------------------------------------
    hybrid_results : list[tuple[SimulationResult, np.ndarray]] = []
    metrics_list   : list[dict] = []

    for step_idx, frac in enumerate(SURROGATE_FRACTIONS, start=2):
        print(f"[{step_idx}/5] Running hybrid {int(frac*100):d}% surrogate ...",
              flush=True)
        runner     = ControlledHybridRunner(surrogate_fraction=frac)
        hyb_result, frame_source, hyb_time = runner.run(DURATION, DT)
        n_sur  = int(np.sum(frame_source == "surrogate"))
        n_phys = int(np.sum(frame_source == "physics"))
        print(f"      done in {hyb_time:.2f} s  |  "
              f"physics={n_phys:,}, surrogate={n_sur:,}  |  "
              f"actual fraction={runner.actual_fraction*100:.0f}%  |  "
              f"speedup={gt_time/max(hyb_time,1e-12):.2f}x")
        m = _compute_metrics(
            gt_result, hyb_result, frame_source,
            gt_time, hyb_time, runner.actual_fraction,
        )
        hybrid_results.append((hyb_result, frame_source))
        metrics_list.append(m)

    # ---- 5. Summary table & figure -----------------------------------------
    print("[5/5] Writing summary and figure ...", flush=True)
    _print_table(SURROGATE_FRACTIONS, metrics_list, gt_time)

    output_path = _HERE / OUTPUT_NAME
    _make_figure(gt_result, hybrid_results, metrics_list, output_path)

    # Machine-readable one-liners
    print("Summary (machine-readable):")
    print(f"  gt_rk4_time_s={gt_time:.3f}")
    for frac, m in zip(SURROGATE_FRACTIONS, metrics_list):
        tag = f"sur{int(frac*100):d}"
        print(
            f"  {tag}: mae_P={m['mae_pressure_mmhg']:.5f} mmHg  "
            f"mae_CO={m['mae_co_lpm']:.6f} L/min  "
            f"speedup={m['speedup']:.3f}x  "
            f"pct={m['surrogate_pct']:.1f}%"
        )
    print()


if __name__ == "__main__":
    main()
