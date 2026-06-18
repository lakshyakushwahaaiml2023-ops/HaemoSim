#!/usr/bin/env bash
# start.sh – launch HaemoSim visualiser server and client dev server in parallel.
# Assumes the Python virtual environment for the project is at haemosim/.venv
# and Node (npm) is installed on the system.

# Activate the Python environment
source "$(pwd)/haemosim/.venv/Scripts/activate"

# Start FastAPI server (WebSocket backend) on port 8000
# Run in background so the script can continue.
uvicorn visualiser.server.main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

echo "[visualiser] FastAPI server started (PID $SERVER_PID)"

# Change to client directory, install deps, and start Vite dev server.
cd visualiser/client || exit 1
npm install
npm run dev &
CLIENT_PID=$!

echo "[visualiser] Vite dev server started (PID $CLIENT_PID)"

# Wait for both processes; when either exits, propagate exit code.
wait -n
STATUS=$?

# Clean up background jobs if still running.
kill $SERVER_PID $CLIENT_PID 2>/dev/null || true
exit $STATUS
