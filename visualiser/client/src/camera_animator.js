// visualiser/client/src/camera_animator.js
// Simple camera animator with three cinematic presets.
// Usage: const animator = new CameraAnimator(camera, renderer);
// animator.startPreset(1|2|3);
// Call animator.update(delta) each frame (e.g., from the main animate loop).

import * as THREE from 'three';

export class CameraAnimator {
  constructor(camera, renderer) {
    this.camera = camera;
    this.renderer = renderer;
    this.active = false;
    this.preset = null;
    this.time = 0;
    // Parameters for presets
    this.orbitRadius = 18;
    this.orbitSpeed = 0.02; // rad/frame for preset 1
    this.zoomTarget = new THREE.Vector3(0, 0, 0);
    this.startPos = this.camera.position.clone();
  }

  startPreset(preset) {
    this.active = true;
    this.preset = preset;
    this.time = 0;
    // Reset any state needed per preset
    if (preset === 1) {
      this.camera.position.set(this.orbitRadius, 5, 0);
      this.camera.lookAt(0, 0, 0);
    } else if (preset === 2) {
      // Close‑up view of the heart (assumed near origin)
      this.camera.position.set(0, 2, 5);
      this.camera.lookAt(0, 0, 0);
    } else if (preset === 3) {
      // Start at a higher point, look downstream
      this.camera.position.set(0, 2, 15);
      this.camera.lookAt(0, 0, 0);
    }
  }

  stop() {
    this.active = false;
    this.preset = null;
  }

  // delta: time since last frame (in seconds)
  update(delta) {
    if (!this.active) return;
    this.time += delta;
    if (this.preset === 1) {
      // Slow orbit around full body
      const angle = this.time * this.orbitSpeed;
      const x = Math.cos(angle) * this.orbitRadius;
      const z = Math.sin(angle) * this.orbitRadius;
      this.camera.position.set(x, 5, z);
      this.camera.lookAt(0, 0, 0);
    } else if (this.preset === 2) {
      // Orbit close up around heart, radius smaller, also slight zoom pulse
      const radius = 3 + Math.sin(this.time * 2) * 0.3;
      const angle = this.time * 1.5;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      this.camera.position.set(x, 1, z);
      this.camera.lookAt(0, 0, 0);
    } else if (this.preset === 3) {
      // Fly downstream along -Z (aorta direction) while looking forward.
      const speed = 4; // units per second
      const dz = -speed * delta;
      this.camera.position.z += dz;
      this.camera.lookAt(this.camera.position.x, this.camera.position.y, this.camera.position.z - 5);
    }
  }
}
