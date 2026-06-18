# visualiser/server/sim_bridge.py
"""Simulation streamer for HaemoSim.
Runs a short HaemoSim simulation (e.g., 5 s) in a background thread and
yields a JSON‑serialisable dict for each time step. This is a minimal
implementation – replace with a full‑duration run as needed.
"""

import asyncio
import threading
from typing import AsyncIterator, Dict, Any

# Import the simulator lazily to avoid heavy import at module load time.

def _run_simulation(duration: float = 5.0, dt: float = 0.01) -> list[Dict[str, Any]]:
    """Run a quick HaemoSim simulation and return a list of state dicts.

    Each dict contains the time stamp and a few representative values
    (aortic pressure and a simple particle count placeholder). The real
    project can expose whatever fields are required for visualisation.
    """
    from haemosim.core.simulator import CirculatorySimulator  # type: ignore

    sim = CirculatorySimulator()
    rhs = sim._rhs
    t = 0.0
    y = sim._initial_state()
    results = []
    while t <= duration:
        # Simple explicit Euler for speed – the visualiser only needs a
        # smooth stream, not high‑fidelity physics.
        dy = rhs(t, y)
        y = y + dt * dy
        # Grab a few scalar outputs from the result object helpers.
        # Here we mimic the aortic pressure stored in the simulator
        # result’s aortic_pressure array after the first step.
        # The full simulator builds a rich result at the end, but for a
        # live stream we compute the pressure on‑the‑fly.
        # Use the first state variable as a dummy pressure if available.
        aortic_pressure = float(y[0]) if hasattr(y, '__len__') and len(y) > 0 else 0.0
        results.append({"time": t, "aortic_pressure": aortic_pressure, "particle_count": len(y)})
        t += dt
    return results


class SimulationBridge:
    """Async iterator that yields simulation frames.

    The simulation runs in a separate thread so that the FastAPI event loop
    remains responsive. ``stream()`` yields dictionaries that are JSON‑
    serialisable and can be consumed directly by the client via ``ws.send_json``.
    """

    def __init__(self, duration: float = 5.0, dt: float = 0.01):
        self.duration = duration
        self.dt = dt
        self._queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        # Run the simulation and push frames onto the asyncio queue.
        frames = _run_simulation(self.duration, self.dt)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async def push():
            for frame in frames:
                if self._stop_event.is_set():
                    break
                await self._queue.put(frame)
            # Signal end of stream with a sentinel.
            await self._queue.put({"__end__": True})
        loop.run_until_complete(push())
        loop.close()

    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        while True:
            msg = await self._queue.get()
            if msg.get("__end__"):
                break
            yield msg

    async def stop(self) -> None:
        self._stop_event.set()
        # Wake any pending ``await queue.get()``.
        await self._queue.put({"__end__": True})
