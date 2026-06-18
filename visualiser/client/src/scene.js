// visualiser/client/src/scene.js
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { FXAAShader } from 'three/examples/jsm/shaders/FXAAShader.js';

export function initScene() {
  const container = document.getElementById('app');

  // Renderer
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  container.appendChild(renderer.domElement);

  // Scene
  const scene = new THREE.Scene();
  // Dark near‑black deep blue background
  scene.background = new THREE.Color(0x0a0a1a);

  // Camera
  const camera = new THREE.PerspectiveCamera(
    45,
    window.innerWidth / window.innerHeight,
    0.1,
    1000,
  );
  camera.position.set(0, 5, 18);
  camera.lookAt(0, 0, 0);

  // Ambient light (soft)
  const ambient = new THREE.AmbientLight(0xffffff, 0.3);
  scene.add(ambient);

  // Warm red point light near the heart (origin placeholder)
  const redLight = new THREE.PointLight(0xff4444, 1.2, 30);
  redLight.position.set(0, 0, 0);
  scene.add(redLight);

  // Cool blue fill light from below
  const blueLight = new THREE.PointLight(0x4488ff, 0.8, 30);
  blueLight.position.set(0, -5, 5);
  scene.add(blueLight);

  // Resize handling
  window.addEventListener('resize', () => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  });

  // Animation loop – callers can add objects to the scene and they will be rendered.
  function animate() {
    const delta = clock.getDelta();
    // Update camera animator if present
    if (window.cameraAnimator) {
      window.cameraAnimator.update(delta);
    }
    requestAnimationFrame(animate);
    // Rotate scene slowly for visual interest
    scene.rotation.y += 0.001;
    // Update star field drift
    updateStars();
    // Render with post‑processing composer
    composer.render();
  }

  return { renderer, scene, camera, animate };
}
