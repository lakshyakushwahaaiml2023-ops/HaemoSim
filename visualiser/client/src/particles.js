// visualiser/client/src/particles.js
// Very simple particle system using Points.
import * as THREE from 'three';

export function createParticles(scene) {
  const count = 200;
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 4;   // x
    positions[i * 3 + 1] = (Math.random() - 0.5) * 4; // y
    positions[i * 3 + 2] = (Math.random() - 0.5) * 4; // z
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({ color: 0x00aaff, size: 0.05 });
  const points = new THREE.Points(geometry, material);
  scene.add(points);
  return points;
}
