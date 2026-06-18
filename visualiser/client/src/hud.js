// visualiser/client/src/hud.js
// Simple HUD that displays latest aortic pressure.

export function initHUD() {
  const hud = document.createElement('div');
  hud.style.position = 'absolute';
  hud.style.top = '10px';
  hud.style.left = '10px';
  hud.style.padding = '6px 12px';
  hud.style.background = 'rgba(0,0,0,0.5)';
  hud.style.color = '#fff';
  hud.style.fontFamily = 'sans-serif';
  hud.style.fontSize = '14px';
  hud.style.borderRadius = '4px';
  hud.innerText = 'Pressure: -- mmHg';
  document.body.appendChild(hud);
  return {
    setPressure(p) {
      hud.innerText = `Pressure: ${p.toFixed(1)} mmHg`;
    },
  };
}
