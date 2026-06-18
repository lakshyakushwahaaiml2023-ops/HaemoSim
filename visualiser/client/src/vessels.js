// visualiser/client/src/vessels.js
// Placeholder for vessel geometry creation.
// In a full implementation you would generate tube geometries representing
// arteries and veins based on simulation data.

import * as THREE from 'three';

export function createVessels(scene) {
  // Simple cylinder as a stand‑in.
  const geometry = new THREE.CylinderGeometry(0.1, 0.1, 5, 12);
  const material = new THREE.MeshStandardMaterial({ color: 0x8b0000 });
  const vessel = new THREE.Mesh(geometry, material);
  vessel.rotation.z = Math.PI / 2;
  scene.add(vessel);
  return vessel;
}
