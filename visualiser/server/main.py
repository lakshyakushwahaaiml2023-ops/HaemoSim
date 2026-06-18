# visualiser/server/main.py
"""FastAPI server that launches the HaemoSim simulation and streams its state over WebSocket.
Only a skeleton – the actual simulation bridge is in sim_bridge.py.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from pathlib import Path
from .sim_bridge import SimulationBridge

app = FastAPI()

@app.get("/")
async def get_root():
    # Simple placeholder page – real client is served separately.
    return HTMLResponse(content="<h1>HaemoSim Visualiser Server</h1>")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    streamer = SimulationStreamer()
    try:
        async for message in streamer.stream():
            await ws.send_json(message)
    except WebSocketDisconnect:
        pass
    finally:
        await streamer.stop()
