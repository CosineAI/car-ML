/* global loadPyodide */
(() => {
  const canvas = document.getElementById('world');
  const ctx = canvas.getContext('2d');

  const playBtn = document.getElementById('playBtn');
  const stopBtn = document.getElementById('stopBtn');
  const resetBtn = document.getElementById('resetBtn');
  const speedRange = document.getElementById('speedRange');
  const speedValue = document.getElementById('speedValue');

  const attemptIdxEl = document.getElementById('attemptIdx');
  const bestDistEl = document.getElementById('bestDist');
  const curDistEl = document.getElementById('curDist');

  // World/view settings
  const UNITS_X = 60; // how many world units fit across the canvas width
  const xScale = canvas.width / UNITS_X;
  const yScale = 25; // pixels per world unit
  const marginLeftPx = 150;
  const marginBottomPx = 60;

  let pyodide = null;
  let sim = null; // PyProxy to Simulator instance
  let terrain = null; // {xs: Float64Array, ys: Float64Array, length: number}
  let playing = false;
  let rafId = null;

  function toPxX(x, camX) {
    return (x - camX) * xScale + marginLeftPx;
  }
  function toPxY(y) {
    return canvas.height - (y * yScale + marginBottomPx);
  }

  function clear() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // subtle gradient sky already set by CSS background
  }

  function drawGrid(camX) {
    ctx.save();
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';

    // vertical grid lines every 5 units
    const startUnit = Math.floor((camX - marginLeftPx / xScale) / 5) * 5;
    const endUnit = Math.ceil((camX + (canvas.width - marginLeftPx) / xScale) / 5) * 5;

    for (let u = startUnit; u <= endUnit; u += 5) {
      const x = toPxX(u, camX);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }

    // horizontal grid lines every 1 unit in world y
    const maxWorldY = (canvas.height - marginBottomPx) / yScale;
    for (let y = 0; y <= maxWorldY; y += 1) {
      const py = toPxY(y);
      ctx.beginPath();
      ctx.moveTo(0, py);
      ctx.lineTo(canvas.width, py);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawTerrain(camX) {
    if (!terrain) return;
    const xs = terrain.xs;
    const ys = terrain.ys;
    const N = xs.length;

    // Determine visible range of indices
    const xMin = camX - marginLeftPx / xScale - 2;
    const xMax = camX + (canvas.width - marginLeftPx) / xScale + 2;

    let iStart = 0;
    while (iStart < N - 1 && xs[iStart] < xMin) iStart++;
    let iEnd = iStart;
    while (iEnd < N && xs[iEnd] < xMax) iEnd++;

    ctx.save();
    ctx.lineWidth = 2;
    const grad = ctx.createLinearGradient(0, 0, 0, canvas.height);
    grad.addColorStop(0, '#1b2638');
    grad.addColorStop(1, '#0b0f16');
    ctx.strokeStyle = '#3f6ca7';
    ctx.fillStyle = grad;

    // Fill below terrain to bottom to give ground color
    ctx.beginPath();
    let first = true;
    for (let i = Math.max(0, iStart - 1); i < Math.min(N, iEnd + 1); i++) {
      const sx = toPxX(xs[i], camX);
      const sy = toPxY(ys[i]);
      if (first) {
        ctx.moveTo(sx, sy);
        first = false;
      } else {
        ctx.lineTo(sx, sy);
      }
    }
    // close down to bottom right and bottom left
    ctx.lineTo(toPxX(xs[Math.min(N - 1, iEnd)], camX), canvas.height);
    ctx.lineTo(toPxX(xs[Math.max(0, iStart - 1)], camX), canvas.height);
    ctx.closePath();
    ctx.fill();

    // Terrain stroke
    ctx.beginPath();
    first = true;
    for (let i = Math.max(0, iStart - 1); i < Math.min(N, iEnd + 1); i++) {
      const sx = toPxX(xs[i], camX);
      const sy = toPxY(ys[i]);
      if (first) {
        ctx.moveTo(sx, sy);
        first = false;
      } else {
        ctx.lineTo(sx, sy);
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  function drawWheel(x, y, r, camX, colorOuter = '#94c5ff', colorInner = '#1f6bb3') {
    const px = toPxX(x, camX);
    const py = toPxY(y);
    const pr = r * yScale;

    ctx.save();
    ctx.beginPath();
    ctx.arc(px, py, pr, 0, Math.PI * 2);
    ctx.fillStyle = colorInner;
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = colorOuter;
    ctx.stroke();

    // spokes for motion hint
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    for (let i = 0; i < 3; i++) {
      const a = (i / 3) * Math.PI * 2;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px + pr * Math.cos(a), py + pr * Math.sin(a));
      ctx.stroke();
    }

    ctx.restore();
  }

  function drawBody(back, front, body, camX) {
    // mid-point
    const mx = 0.5 * (back.x + front.x);
    const my = 0.5 * (back.y + front.y);
    const phi = body.phi;
    const halfBase = 0.5 * body.base_len;
    const height = body.height;

    // Local tri points relative to mid point in axle-aligned coords
    const pts = [
      [-halfBase, 0],
      [halfBase, 0],
      [0, height]
    ];

    // rotate by phi and translate to world, then to screen
    const toScreen = (vx, vy) => {
      const wx = mx + vx * Math.cos(phi) - vy * Math.sin(phi);
      const wy = my + vx * Math.sin(phi) + vy * Math.cos(phi);
      return [toPxX(wx, camX), toPxY(wy)];
    };

    const p0 = toScreen(pts[0][0], pts[0][1]);
    const p1 = toScreen(pts[1][0], pts[1][1]);
    const p2 = toScreen(pts[2][0], pts[2][1]);

    ctx.save();
    const grad = ctx.createLinearGradient(p0[0], p2[1], p2[0], p0[1]);
    grad.addColorStop(0, '#4b97d6');
    grad.addColorStop(1, '#3dc5a1');

    ctx.beginPath();
    ctx.moveTo(p0[0], p0[1]);
    ctx.lineTo(p1[0], p1[1]);
    ctx.lineTo(p2[0], p2[1]);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // axle
    ctx.beginPath();
    ctx.moveTo(toPxX(back.x, camX), toPxY(back.y));
    ctx.lineTo(toPxX(front.x, camX), toPxY(front.y));
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.stroke();

    ctx.restore();
  }

  function updateStats(data) {
    attemptIdxEl.textContent = String(data.attempt);
    bestDistEl.textContent = data.best_distance.toFixed(2);
    curDistEl.textContent = data.current_distance.toFixed(2);
  }

  async function init() {
    playBtn.disabled = true;
    resetBtn.disabled = true;

    const statusSpan = document.createElement('span');
    statusSpan.textContent = 'Loading Python (Pyodide)...';
    statusSpan.style.marginLeft = '12px';
    statusSpan.style.color = '#9ca3af';
    document.querySelector('.controls').appendChild(statusSpan);

    pyodide = await loadPyodide({ indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.28.2/full/' });

    // Load Python simulation code
    const simCode = await (await fetch('static/py/sim.py')).text();
    await pyodide.runPythonAsync(simCode);

    // Create simulator instance in Python
    const seed = Math.floor(Math.random() * 1e9);
    sim = pyodide.runPython(`sim = Simulator(seed=${seed})
sim`);

    // Fetch a pre-sampled terrain profile for rendering
    const profileProxy = pyodide.runPython('sim.export_terrain_profile()');
    const profile = profileProxy.toJs();
    profileProxy.destroy();

    terrain = {
      xs: profile.xs,
      ys: profile.ys,
      length: profile.length
    };

    statusSpan.remove();
    playBtn.disabled = false;
    resetBtn.disabled = false;

    // initial draw
    const initialProxy = sim.next_frame(0); // just to read starting camera
    const initial = initialProxy.toJs();
    initialProxy.destroy();
    clear();
    drawGrid(initial.camera_x);
    drawTerrain(initial.camera_x);
    updateStats(initial);
  }

  function loop() {
    if (!playing) return;
    const stepsPerFrame = parseInt(speedRange.value, 10);
    const frameProxy = sim.next_frame(stepsPerFrame);
    const frame = frameProxy.toJs();
    frameProxy.destroy();

    clear();
    drawGrid(frame.camera_x);
    drawTerrain(frame.camera_x);

    const back = frame.back_wheel;
    const front = frame.front_wheel;
    const body = {
      phi: frame.phi,
      base_len: frame.body_base_len,
      height: frame.body_height
    };

    drawWheel(back.x, back.y, back.r, frame.camera_x, '#93c5fd', '#1e3a8a');
    drawWheel(front.x, front.y, front.r, frame.camera_x, '#86efac', '#064e3b');
    drawBody(back, front, body, frame.camera_x);

    updateStats(frame);

    if (frame.done) {
      // Allow a small pause to let the viewer see end of run
      // but keep loop going to next car
    }

    rafId = requestAnimationFrame(loop);
  }

  playBtn.addEventListener('click', () => {
    if (playing) return;
    playing = true;
    playBtn.disabled = true;
    stopBtn.disabled = false;
    loop();
  });

  stopBtn.addEventListener('click', () => {
    playing = false;
    playBtn.disabled = false;
    stopBtn.disabled = true;
    if (rafId) cancelAnimationFrame(rafId);
  });

  speedRange.addEventListener('input', () => {
    speedValue.textContent = `${speedRange.value}x`;
  });

  resetBtn.addEventListener('click', async () => {
    // Stop current sim
    playing = false;
    playBtn.disabled = false;
    stopBtn.disabled = true;
    if (rafId) cancelAnimationFrame(rafId);

    const seed = Math.floor(Math.random() * 1e9);
    const resProxy = pyodide.runPython(`sim.reset(seed=${seed}); sim.export_terrain_profile()`);
    const profile = resProxy.toJs();
    resProxy.destroy();
    terrain = {
      xs: profile.xs,
      ys: profile.ys,
      length: profile.length
    };
    const initProxy = sim.next_frame(0);
    const init = initProxy.toJs();
    initProxy.destroy();
    clear();
    drawGrid(init.camera_x);
    drawTerrain(init.camera_x);
    updateStats(init);
  });

  window.addEventListener('resize', () => {
    // Canvas is fixed size for now; could add responsive behavior if needed.
  });

  // Kick off
  init().catch(err => {
    console.error(err);
    const msg = document.createElement('div');
    msg.textContent = 'Failed to initialize Pyodide. Check your connection.';
    msg.style.color = '#f87171';
    document.body.appendChild(msg);
  });
})();