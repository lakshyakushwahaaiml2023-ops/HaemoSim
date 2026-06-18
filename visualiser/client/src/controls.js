// visualiser/client/src/controls.js
// Interactive control panel for the simulation visualiser.
// Creates a collapsible right‑hand drawer with sliders, preset buttons, and toggle switches.
// All controls send JSON messages over the global WebSocket (window.ws).

export function initControls() {
  // Create drawer container
  const drawer = document.createElement('div');
  drawer.id = 'control-drawer';
  Object.assign(drawer.style, {
    position: 'fixed',
    top: '0',
    right: '0',
    height: '100vh',
    width: '320px',
    maxWidth: '90vw',
    background: 'rgba(0,0,0,0.6)',
    backdropFilter: 'blur(12px)',
    color: '#fff',
    fontFamily: 'sans-serif',
    fontSize: '14px',
    padding: '20px',
    boxSizing: 'border-box',
    overflowY: 'auto',
    transition: 'transform 0.3s ease-in-out',
    transform: 'translateX(0)',
    zIndex: '1000',
  });

  // Collapse button
  const collapseBtn = document.createElement('button');
  collapseBtn.textContent = '◀';
  Object.assign(collapseBtn.style, {
    position: 'absolute',
    left: '-30px',
    top: '20px',
    width: '30px',
    height: '30px',
    background: 'rgba(0,0,0,0.6)',
    border: 'none',
    color: '#fff',
    cursor: 'pointer',
  });
  collapseBtn.onclick = () => {
    const hidden = drawer.style.transform !== 'translateX(0)';
    drawer.style.transform = hidden ? 'translateX(0)' : 'translateX(100%)';
    collapseBtn.textContent = hidden ? '◀' : '▶';
  };
  drawer.appendChild(collapseBtn);

  // Helper to create a labeled slider
  function createSlider(label, id, min, max, step, defaultVal) {
    const wrapper = document.createElement('div');
    wrapper.style.marginBottom = '12px';
    const lbl = document.createElement('label');
    lbl.htmlFor = id;
    lbl.textContent = `${label}: `;
    lbl.style.display = 'block';
    const valueSpan = document.createElement('span');
    valueSpan.id = `${id}-value`;
    valueSpan.textContent = defaultVal;
    lbl.appendChild(valueSpan);
    const input = document.createElement('input');
    input.type = 'range';
    input.id = id;
    input.min = min;
    input.max = max;
    input.step = step;
    input.value = defaultVal;
    input.style.width = '100%';
    input.oninput = () => {
      valueSpan.textContent = input.value;
      sendControlMessage();
    };
    wrapper.appendChild(lbl);
    wrapper.appendChild(input);
    return {wrapper, input};
  }

  // Sliders definitions
  const sliders = {};
  const sliderDefs = [
    {label: 'Heart Rate (bpm)', id: 'hr', min: 40, max: 180, step: 1, def: 70},
    {label: 'Peripheral Resistance', id: 'pr', min: 0.5, max: 2.0, step: 0.01, def: 1.0},
    {label: 'Arterial Compliance', id: 'ac', min: 0.5, max: 2.0, step: 0.01, def: 1.0},
    {label: 'Simulation Speed', id: 'sp', min: 0.25, max: 4.0, step: 0.05, def: 1.0},
  ];
  sliderDefs.forEach(def => {
    const {wrapper, input} = createSlider(def.label, `slider-${def.id}`, def.min, def.max, def.step, def.def);
    drawer.appendChild(wrapper);
    sliders[def.id] = input;
  });

  // Preset buttons
  const presetContainer = document.createElement('div');
  presetContainer.style.margin = '16px 0';
  const presetLabel = document.createElement('div');
  presetLabel.textContent = 'Presets:';
  presetLabel.style.marginBottom = '8px';
  presetContainer.appendChild(presetLabel);
  const presets = [
    {name: 'Resting', values: {hr: 70, pr: 1.0, ac: 1.0, sp: 1.0}},
    {name: 'Exercise', values: {hr: 120, pr: 1.2, ac: 0.8, sp: 2.0}},
    {name: 'Hypertension', values: {hr: 85, pr: 1.5, ac: 1.0, sp: 1.0}},
    {name: 'Haemorrhage', values: {hr: 110, pr: 0.7, ac: 0.5, sp: 1.0}},
    {name: 'Arrhythmia', values: {hr: 70, pr: 1.0, ac: 1.0, sp: 1.0}, arrhythmia: true},
  ];
  presets.forEach(p => {
    const btn = document.createElement('button');
    btn.textContent = p.name;
    btn.style.marginRight = '6px';
    btn.style.marginBottom = '6px';
    btn.onclick = () => {
      // Apply slider values
      Object.entries(p.values).forEach(([k, v]) => {
        if (sliders[k]) sliders[k].value = v;
        const span = document.getElementById(`slider-${k}-value`);
        if (span) span.textContent = v;
      });
      sendControlMessage();
      // Handle arrhythmia special case
      if (p.arrhythmia) startArrhythmia();
      else stopArrhythmia();
    };
    presetContainer.appendChild(btn);
  });
  drawer.appendChild(presetContainer);

  // Toggle switches
  const toggleContainer = document.createElement('div');
  toggleContainer.style.margin = '16px 0';
  const toggleLabel = document.createElement('div');
  toggleLabel.textContent = 'Toggle switches:';
  toggleLabel.style.marginBottom = '8px';
  toggleContainer.appendChild(toggleLabel);
  const toggles = [
    {label: 'Show Particles', id: 'particles'},
    {label: 'Show Vessels', id: 'vessels'},
    {label: 'Bloom Effect', id: 'bloom'},
    {label: 'Surrogate Mode', id: 'surrogate'},
  ];
  toggles.forEach(t => {
    const wrapper = document.createElement('label');
    wrapper.style.display = 'block';
    wrapper.style.marginBottom = '6px';
    const chk = document.createElement('input');
    chk.type = 'checkbox';
    chk.id = `toggle-${t.id}`;
    chk.onchange = () => {
      sendToggleMessage(t.id, chk.checked);
    };
    wrapper.appendChild(chk);
    wrapper.appendChild(document.createTextNode(' ' + t.label));
    toggleContainer.appendChild(wrapper);
  });
  drawer.appendChild(toggleContainer);

  // Append drawer to body
  document.body.appendChild(drawer);

  // Helper to send control messages
  function sendControlMessage() {
    if (!window.ws) return;
    const msg = {
      type: 'control',
      heart_rate: parseFloat(sliders.hr.value),
      peripheral_resistance: parseFloat(sliders.pr.value),
      arterial_compliance: parseFloat(sliders.ac.value),
      simulation_speed: parseFloat(sliders.sp.value),
    };
    window.ws.send(JSON.stringify(msg));
  }

  function sendToggleMessage(name, enabled) {
    if (!window.ws) return;
    const msg = {type: 'toggle', name, enabled};
    window.ws.send(JSON.stringify(msg));
  }

  // Arrhythmia handling
  let arrhythmiaInterval = null;
  function startArrhythmia() {
    stopArrhythmia();
    arrhythmiaInterval = setInterval(() => {
      const randHR = Math.floor(Math.random() * (160 - 40 + 1)) + 40;
      if (window.ws) {
        const msg = {type: 'control', heart_rate: randHR};
        window.ws.send(JSON.stringify(msg));
      }
    }, 2000);
  }
  function stopArrhythmia() {
    if (arrhythmiaInterval) clearInterval(arrhythmiaInterval);
    arrhythmiaInterval = null;
  }

  // Initialize with default values
  sendControlMessage();
}
