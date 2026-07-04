// NimbleCAS renderer — WebGPU plot pipeline with a complete Canvas2D fallback.
//
// ES module. Exports:
//   initWebGPU()                       -> Promise<boolean>   (sets up device/pipeline)
//   GPU                                -> { ok, device, format, pipeline }
//   renderPlot(host, spec, opts)       -> void               (appends canvases to host)
//   palette(), cssVar(name, fallback)  -> theme helpers
//
// The standalone web/index.html embeds an equivalent copy of this code inline;
// this module is the split-out source for a bundler-free ES-module setup
// (e.g. <script type="module" src="app.js">).

// Loaded from the sibling shaders.wgsl at runtime (see initWebGPU). Kept here so
// a caller can override it if desired.
export const GPU = { ok: false, device: null, format: null, pipeline: null };

const SERIES_COLORS = ['#2563eb', '#e0533d', '#12a150', '#a855f7', '#d97706', '#0891b2'];

export function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}
export function palette() {
  return {
    panel: cssVar('--panel', '#f6f7f9'),
    axis: cssVar('--axis', '#3a4149'),
    grid: cssVar('--grid', '#dde2e8'),
    ink: cssVar('--ink', '#1b1f24'),
    inkSoft: cssVar('--ink-soft', '#55606b'),
    border: cssVar('--border', '#d7dce2'),
  };
}

function hexToRGBA(hex, a) {
  if (typeof hex !== 'string') return [0.2, 0.4, 0.9, a == null ? 1 : a];
  let h = hex.replace('#', '');
  if (h.length === 3) h = h.split('').map(c => c + c).join('');
  const n = parseInt(h, 16);
  if (!isFinite(n)) return [0.2, 0.4, 0.9, a == null ? 1 : a];
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255, a == null ? 1 : a];
}

// --- WebGPU init ------------------------------------------------------------
export async function initWebGPU(wgslSource, onLost) {
  if (!('gpu' in navigator) || !navigator.gpu) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return false;
    const device = await adapter.requestDevice();
    if (!device) return false;
    device.lost.then((info) => { GPU.ok = false; if (onLost) onLost(info); });

    // If no source passed, try fetching the sibling shaders.wgsl.
    let code = wgslSource;
    if (!code) {
      const r = await fetch(new URL('./shaders.wgsl', import.meta.url));
      code = await r.text();
    }
    const format = navigator.gpu.getPreferredCanvasFormat();
    const module = device.createShaderModule({ code });
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module, entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 6 * 4,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x2' },
            { shaderLocation: 1, offset: 2 * 4, format: 'float32x4' },
          ],
        }],
      },
      fragment: {
        module, entryPoint: 'fs_main',
        targets: [{
          format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });
    GPU.device = device; GPU.format = format; GPU.pipeline = pipeline; GPU.ok = true;
    return true;
  } catch (e) {
    console.warn('WebGPU init failed:', e);
    return false;
  }
}

// --- layout / ticks ---------------------------------------------------------
function niceNum(range, round) {
  const exp = Math.floor(Math.log10(range || 1));
  const f = (range || 1) / Math.pow(10, exp);
  let nf;
  if (round) nf = f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10;
  else nf = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
  return nf * Math.pow(10, exp);
}
function niceTicks(min, max, count) {
  if (!(isFinite(min) && isFinite(max)) || min === max) { min -= 1; max += 1; }
  const range = niceNum(max - min, false);
  const step = niceNum(range / Math.max(1, count - 1), true);
  const gmin = Math.floor(min / step) * step;
  const gmax = Math.ceil(max / step) * step;
  const ticks = [];
  for (let v = gmin; v <= gmax + step * 0.5; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return ticks;
}
function fmtTick(v) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e4 || a < 1e-3) return v.toExponential(1);
  return String(Math.round(v * 1000) / 1000);
}
function autoRange(series, axis) {
  let lo = Infinity, hi = -Infinity;
  for (const s of series) for (const p of (s.points || [])) {
    const v = axis === 0 ? p[0] : p[1];
    if (isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  }
  if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.05;
  return [lo - pad, hi + pad];
}
function computeLayout(spec, W, H) {
  const M = { l: 58, r: 16, t: 34, b: 46 };
  const rect = { x: M.l, y: M.t, w: Math.max(10, W - M.l - M.r), h: Math.max(10, H - M.t - M.b) };
  const series = spec.series || [];
  const xr = (spec.xRange && spec.xRange.length === 2) ? spec.xRange.slice() : autoRange(series, 0);
  const yr = (spec.yRange && spec.yRange.length === 2) ? spec.yRange.slice() : autoRange(series, 1);
  const [x0, x1] = xr, [y0, y1] = yr;
  const xToPx = (x) => rect.x + (x - x0) / (x1 - x0 || 1) * rect.w;
  const yToPx = (y) => rect.y + rect.h - (y - y0) / (y1 - y0 || 1) * rect.h;
  return {
    W, H, rect, xr, yr, xToPx, yToPx,
    xticks: niceTicks(x0, x1, 8).filter(t => t >= x0 - 1e-9 && t <= x1 + 1e-9),
    yticks: niceTicks(y0, y1, 6).filter(t => t >= y0 - 1e-9 && t <= y1 + 1e-9),
  };
}

// --- text overlay (Canvas2D, both backends) ---------------------------------
function drawText(ctx, layout, spec) {
  const pal = palette();
  const { rect } = layout;
  ctx.clearRect(0, 0, layout.W, layout.H);
  ctx.textBaseline = 'middle';
  ctx.fillStyle = pal.ink;
  ctx.font = '650 15px system-ui, sans-serif';
  ctx.textAlign = 'center';
  if (spec.title) ctx.fillText(spec.title, layout.W / 2, 16);
  ctx.fillStyle = pal.inkSoft;
  ctx.font = '11px system-ui, sans-serif';
  ctx.textAlign = 'center';
  for (const t of layout.xticks) ctx.fillText(fmtTick(t), layout.xToPx(t), rect.y + rect.h + 14);
  ctx.textAlign = 'right';
  for (const t of layout.yticks) ctx.fillText(fmtTick(t), rect.x - 8, layout.yToPx(t));
  ctx.fillStyle = pal.inkSoft;
  ctx.font = '600 12px system-ui, sans-serif';
  ctx.textAlign = 'center';
  if (spec.xLabel) ctx.fillText(spec.xLabel, rect.x + rect.w / 2, layout.H - 8);
  if (spec.yLabel) {
    ctx.save(); ctx.translate(14, rect.y + rect.h / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText(spec.yLabel, 0, 0); ctx.restore();
  }
  const series = spec.series || [];
  if (series.length) {
    ctx.font = '600 11px system-ui, sans-serif';
    ctx.textAlign = 'left';
    let ly = rect.y + 12; const lx = rect.x + rect.w - 128;
    for (let i = 0; i < series.length; i++) {
      const s = series[i];
      ctx.fillStyle = s.color || SERIES_COLORS[i % SERIES_COLORS.length];
      ctx.fillRect(lx, ly - 4, 14, 8);
      ctx.fillStyle = pal.ink;
      ctx.fillText(s.label || ('series ' + (i + 1)), lx + 20, ly);
      ly += 16;
    }
  }
}

// --- Canvas2D geometry ------------------------------------------------------
function drawGeom2D(ctx, layout, spec) {
  const pal = palette();
  const { rect } = layout;
  ctx.clearRect(0, 0, layout.W, layout.H);
  ctx.fillStyle = pal.panel;
  ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
  ctx.strokeStyle = pal.grid; ctx.lineWidth = 1;
  ctx.beginPath();
  for (const t of layout.xticks) { const px = Math.round(layout.xToPx(t)) + 0.5; ctx.moveTo(px, rect.y); ctx.lineTo(px, rect.y + rect.h); }
  for (const t of layout.yticks) { const py = Math.round(layout.yToPx(t)) + 0.5; ctx.moveTo(rect.x, py); ctx.lineTo(rect.x + rect.w, py); }
  ctx.stroke();
  ctx.strokeStyle = pal.axis; ctx.lineWidth = 1.25;
  ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.w, rect.h);
  ctx.save();
  ctx.beginPath(); ctx.rect(rect.x, rect.y, rect.w, rect.h); ctx.clip();
  const series = spec.series || [];
  for (let i = 0; i < series.length; i++) {
    const s = series[i];
    const col = s.color || SERIES_COLORS[i % SERIES_COLORS.length];
    const pts = s.points || [];
    if (s.kind === 'scatter') {
      ctx.fillStyle = col;
      for (const p of pts) { ctx.beginPath(); ctx.arc(layout.xToPx(p[0]), layout.yToPx(p[1]), 3.2, 0, Math.PI * 2); ctx.fill(); }
    } else {
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      ctx.beginPath();
      let started = false;
      for (const p of pts) {
        const px = layout.xToPx(p[0]), py = layout.yToPx(p[1]);
        if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
  }
  ctx.restore();
}

// --- WebGPU geometry --------------------------------------------------------
function buildGeometry(layout, spec, W, H) {
  const pal = palette();
  const verts = [];
  const cx = (px) => (px / W) * 2 - 1;
  const cy = (py) => 1 - (py / H) * 2;
  function pushQuadPx(x0, y0, x1, y1, col) {
    const ax = cx(x0), ay = cy(y0), bx = cx(x1), by = cy(y1);
    const [r, g, b, a] = col;
    verts.push(ax, ay, r, g, b, a, bx, ay, r, g, b, a, bx, by, r, g, b, a);
    verts.push(ax, ay, r, g, b, a, bx, by, r, g, b, a, ax, by, r, g, b, a);
  }
  function pushSegPx(x0, y0, x1, y1, width, col) {
    let dx = x1 - x0, dy = y1 - y0;
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len * width / 2, ny = dx / len * width / 2;
    const p = [[x0 + nx, y0 + ny], [x1 + nx, y1 + ny], [x1 - nx, y1 - ny], [x0 - nx, y0 - ny]];
    const [r, g, b, a] = col;
    const tri = (A, B, C) => {
      verts.push(cx(A[0]), cy(A[1]), r, g, b, a);
      verts.push(cx(B[0]), cy(B[1]), r, g, b, a);
      verts.push(cx(C[0]), cy(C[1]), r, g, b, a);
    };
    tri(p[0], p[1], p[2]); tri(p[0], p[2], p[3]);
  }
  const { rect } = layout;
  pushQuadPx(rect.x, rect.y, rect.x + rect.w, rect.y + rect.h, hexToRGBA(pal.panel, 1));
  const gcol = hexToRGBA(pal.grid, 1);
  for (const t of layout.xticks) { const px = layout.xToPx(t); pushQuadPx(px - 0.5, rect.y, px + 0.5, rect.y + rect.h, gcol); }
  for (const t of layout.yticks) { const py = layout.yToPx(t); pushQuadPx(rect.x, py - 0.5, rect.x + rect.w, py + 0.5, gcol); }
  const acol = hexToRGBA(pal.axis, 1);
  pushQuadPx(rect.x, rect.y, rect.x + rect.w, rect.y + 1.25, acol);
  pushQuadPx(rect.x, rect.y + rect.h - 1.25, rect.x + rect.w, rect.y + rect.h, acol);
  pushQuadPx(rect.x, rect.y, rect.x + 1.25, rect.y + rect.h, acol);
  pushQuadPx(rect.x + rect.w - 1.25, rect.y, rect.x + rect.w, rect.y + rect.h, acol);
  const inX = (x) => Math.max(rect.x, Math.min(rect.x + rect.w, x));
  const inY = (y) => Math.max(rect.y, Math.min(rect.y + rect.h, y));
  const series = spec.series || [];
  for (let i = 0; i < series.length; i++) {
    const s = series[i];
    const col = hexToRGBA(s.color || SERIES_COLORS[i % SERIES_COLORS.length], 1);
    const pts = s.points || [];
    if (s.kind === 'scatter') {
      for (const p of pts) {
        const px = layout.xToPx(p[0]), py = layout.yToPx(p[1]);
        if (px < rect.x - 4 || px > rect.x + rect.w + 4 || py < rect.y - 4 || py > rect.y + rect.h + 4) continue;
        pushQuadPx(px - 3.2, py - 3.2, px + 3.2, py + 3.2, col);
      }
    } else {
      for (let k = 1; k < pts.length; k++) {
        const ax = inX(layout.xToPx(pts[k - 1][0])), ay = inY(layout.yToPx(pts[k - 1][1]));
        const bx = inX(layout.xToPx(pts[k][0])), by = inY(layout.yToPx(pts[k][1]));
        pushSegPx(ax, ay, bx, by, 2.0, col);
      }
    }
  }
  return new Float32Array(verts);
}

function renderGeomGPU(canvas, layout, spec) {
  const device = GPU.device;
  const ctx = canvas.getContext('webgpu');
  ctx.configure({ device, format: GPU.format, alphaMode: 'premultiplied' });
  const data = buildGeometry(layout, spec, layout.W, layout.H);
  const vbuf = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vbuf, 0, data);
  const bg = hexToRGBA(cssVar('--bg', '#ffffff'), 1);
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: ctx.getCurrentTexture().createView(),
      clearValue: { r: bg[0], g: bg[1], b: bg[2], a: 1 },
      loadOp: 'clear', storeOp: 'store',
    }],
  });
  pass.setPipeline(GPU.pipeline);
  pass.setVertexBuffer(0, vbuf);
  pass.draw(data.length / 6);
  pass.end();
  device.queue.submit([encoder.finish()]);
  vbuf.destroy && vbuf.destroy();
}

// --- public entry -----------------------------------------------------------
function makeCanvas(W, H, dpr, cls) {
  const c = document.createElement('canvas');
  c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
  c.style.width = W + 'px'; c.style.height = H + 'px';
  if (cls) c.className = cls;
  return c;
}

export function renderPlot(host, spec, opts = {}) {
  const wrap = document.createElement('div');
  wrap.className = 'plotwrap';
  host.appendChild(wrap);
  const W = Math.max(320, Math.floor(wrap.clientWidth || host.clientWidth || 720));
  const H = Math.round(Math.min(440, Math.max(240, W * 0.58)));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const base = makeCanvas(W, H, dpr, 'base');
  const overlay = makeCanvas(W, H, dpr, 'overlay');
  wrap.appendChild(base); wrap.appendChild(overlay);
  const layout = computeLayout(spec, W, H);
  const octx = overlay.getContext('2d'); octx.scale(dpr, dpr);
  drawText(octx, layout, spec);
  if (GPU.ok) {
    try { renderGeomGPU(base, layout, spec); return; }
    catch (e) {
      console.warn('Per-plot WebGPU render failed; Canvas2D:', e);
      GPU.ok = false;
      if (opts.onBackendChange) opts.onBackendChange('cpu');
      const fresh = makeCanvas(W, H, dpr, 'base');
      wrap.replaceChild(fresh, base);
      const b2 = fresh.getContext('2d'); b2.scale(dpr, dpr);
      drawGeom2D(b2, layout, spec);
      return;
    }
  }
  const bctx = base.getContext('2d'); bctx.scale(dpr, dpr);
  drawGeom2D(bctx, layout, spec);
}
