// visualiser/client/src/ws_client.js
// Simple WebSocket wrapper that invokes a user‑provided callback for each message.

export function startWebSocket(onMessage) {
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  const host = location.hostname;
  const port = 8000; // FastAPI server port (hard‑coded for dev)
  const ws = new WebSocket(`${protocol}://${host}:${port}/ws`);

  ws.onopen = () => {
    console.log('WebSocket connected to simulation server');
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (e) {
      console.error('Failed to parse WS message', e);
    }
  };

  ws.onclose = () => {
    console.log('WebSocket closed');
  };

  ws.onerror = (err) => {
    console.error('WebSocket error', err);
  };

  return ws;
}
