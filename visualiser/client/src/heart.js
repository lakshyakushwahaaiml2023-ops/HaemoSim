// visualiser/client/src/heart.js
// Simple heart placeholder – a pulsating sphere.
import * as THREE from 'three';

export function createHeart(scene) {
  const geometry = new THREE.SphereGeometry(0.5, 32, 32);
  const material = new THREE.MeshStandardMaterial({ color: 0xff0000 });
  const heart = new THREE.Mesh(geometry, material);
  heart.position.set(0, 0, 0);
  scene.add(heart);

  // Return a function to update the scale based on a pressure value.
  const update = (pressure) => {
    // Normalise pressure to a small scale factor (e.g., 60–100 mmHg).
    const base = 0.5;
    const scale = base + (pressure - 60) / 200; // modest pulsation
    heart.scale.setScalar(scale);
  };
  return { heart, update };
}
