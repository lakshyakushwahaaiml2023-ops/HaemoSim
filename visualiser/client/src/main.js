// visualiser/client/src/main.js
import * as THREE from 'three';
import { CameraAnimator } from './camera_animator.js';
import { initScene } from './scene.js';
import { createHeart } from './heart.js';
import { createVessels } from './vessels.js';
import { createParticles } from './particles.js';
import { initHUD } from './hud.js';
import { startWebSocket } from './ws_client.js';
import { initControls } from './controls.js';

// Initialize Three.js scene and start render loop
const { renderer, scene, camera, animate } = initScene();
// Camera animator
const cameraAnimator = new CameraAnimator(camera, renderer);
window.cameraAnimator = cameraAnimator;
window.renderer = renderer;
// Key presets
window.addEventListener('keydown', (e) => {
  if (e.key === '1') cameraAnimator.startPreset(1);
  else if (e.key === '2') cameraAnimator.startPreset(2);
  else if (e.key === '3') cameraAnimator.startPreset(3);
});

// Add visual objects
const { heart, update: updateHeart } = createHeart(scene);
createVessels(scene);
createParticles(scene);
const hud = initHUD();
initControls();

// Start receiving simulation data via WebSocket
window.ws = startWebSocket((msg) => {
  // Handle incoming simulation data
  // (existing handling remains)

  if (msg && typeof msg.aortic_pressure === 'number') {
    const p = msg.aortic_pressure;
    hud.setPressure(p);
    updateHeart(p);
  }
});

// Kick off animation loop
animate();
