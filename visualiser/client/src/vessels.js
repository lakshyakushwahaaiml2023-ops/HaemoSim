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
  // Example segment data – replace with real simulation values
  vessel.userData = {
    segmentName: 'Aorta',
    flow: 5.2, // mL/s
    resistance: 1.8,
    pressureDrop: 12.5,
  };
  vessel.name = 'vesselMesh';
  vessel.rotation.z = Math.PI / 2;
  scene.add(vessel);
  return vessel;
}
