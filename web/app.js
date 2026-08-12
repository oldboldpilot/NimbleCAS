// NimbleCAS Viewer — application logic (split-out ES module).
//
// Wire-up for a bundler-free module setup:
//   <script type="module" src="app.js"></script>
// It imports the renderer and provides: Markdown-subset prose, a minimal
// LaTeX -> MathML converter, the document renderer, the optional WASM kernel
// hook, the file input, and boot. The standalone web/index.html embeds an
// equivalent copy of all of this inline.

import { GPU, initWebGPU, renderPlot } from './renderer.js';

// --- Markdown subset --------------------------------------------------------
function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function inlineMD(s) {
  let t = escapeHTML(s);
  t = t.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  t = t.replace(/`([^`]+)`/g, '<code>$1</code>');
  return t;
}
function renderProse(text) {
  const lines = String(text).replace(/\r\n/g, '\n').split('\n');
  let html = '', para = [], list = null;
  const flushPara = () => { if (para.length) { html += '<p>' + inlineMD(para.join(' ')) + '</p>'; para = []; } };
  const flushList = () => { if (list) { html += '<ul>' + list.map(li => '<li>' + inlineMD(li) + '</li>').join('') + '</ul>'; list = null; } };
  for (const raw of lines) {
    const line = raw.trimEnd();
    const h = /^(#{1,3})\s+(.*)$/.exec(line);
    const li = /^\s*[-*]\s+(.*)$/.exec(line);
    if (h) { flushPara(); flushList(); const lvl = h[1].length; html += `<h${lvl}>${inlineMD(h[2])}</h${lvl}>`; }
    else if (li) { flushPara(); (list || (list = [])).push(li[1]); }
    else if (line.trim() === '') { flushPara(); flushList(); }
    else { flushList(); para.push(line.trim()); }
  }
  flushPara(); flushList();
  return html;
}

// --- LaTeX -> MathML --------------------------------------------------------
const GREEK = {
  alpha:'α', beta:'β', gamma:'γ', delta:'δ', epsilon:'ε', varepsilon:'ε', zeta:'ζ',
  eta:'η', theta:'θ', vartheta:'ϑ', iota:'ι', kappa:'κ', lambda:'λ', mu:'μ', nu:'ν',
  xi:'ξ', pi:'π', rho:'ρ', sigma:'σ', tau:'τ', upsilon:'υ', phi:'φ', varphi:'φ',
  chi:'χ', psi:'ψ', omega:'ω', Gamma:'Γ', Delta:'Δ', Theta:'Θ', Lambda:'Λ', Xi:'Ξ',
  Pi:'Π', Sigma:'Σ', Phi:'Φ', Psi:'Ψ', Omega:'Ω',
};
const OPS = {
  cdot:'⋅', times:'×', div:'÷', pm:'±', mp:'∓', le:'≤', leq:'≤', ge:'≥', geq:'≥',
  neq:'≠', ne:'≠', approx:'≈', equiv:'≡', int:'∫', sum:'∑', prod:'∏', infty:'∞',
  partial:'∂', nabla:'∇', rightarrow:'→', to:'→', leftarrow:'←', Rightarrow:'⇒',
  leftrightarrow:'↔', in:'∈', notin:'∉', subset:'⊂', subseteq:'⊆', cup:'∪', cap:'∩',
  forall:'∀', exists:'∃', langle:'⟨', rangle:'⟩', ldots:'…', dots:'…', cdots:'⋯',
  mapsto:'↦', propto:'∝', sim:'∼', star:'⋆', ast:'∗', circ:'∘',
};
const FUNCS = ['sin','cos','tan','cot','sec','csc','sinh','cosh','tanh','log','ln',
  'exp','det','dim','ker','deg','gcd','lim','max','min','arg','sup','inf'];

function tokenizeLatex(src) {
  const toks = []; let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (c === '\\') {
      let j = i + 1, name = '';
      if (/[a-zA-Z]/.test(src[j])) { while (j < src.length && /[a-zA-Z]/.test(src[j])) name += src[j++]; }
      else { name = src[j]; j++; }
      toks.push({ t: 'cmd', v: name }); i = j;
    } else if (/\s/.test(c)) i++;
    else if (c === '{' || c === '}' || c === '^' || c === '_') { toks.push({ t: c }); i++; }
    else if (/[0-9.]/.test(c)) { let n = ''; while (i < src.length && /[0-9.]/.test(src[i])) n += src[i++]; toks.push({ t: 'num', v: n }); }
    else if (/[a-zA-Z]/.test(c)) { toks.push({ t: 'ident', v: c }); i++; }
    else { toks.push({ t: 'op', v: c }); i++; }
  }
  return toks;
}
function parseSeq(toks, i, stop) {
  let out = '';
  while (i < toks.length && !(stop && toks[i].t === stop)) {
    let atom = parseAtom(toks, i); i = atom.i;
    let node = atom.node, sub = null, sup = null;
    while (i < toks.length && (toks[i].t === '^' || toks[i].t === '_')) {
      const kind = toks[i].t; i++;
      const s = parseAtom(toks, i); i = s.i;
      if (kind === '^') sup = s.node; else sub = s.node;
    }
    if (sub && sup) node = `<msubsup>${node}${sub}${sup}</msubsup>`;
    else if (sub) node = `<msub>${node}${sub}</msub>`;
    else if (sup) node = `<msup>${node}${sup}</msup>`;
    out += node;
  }
  return { node: out, i };
}
function parseAtom(toks, i) {
  const tk = toks[i];
  if (!tk) return { node: '<mi></mi>', i };
  if (tk.t === '{') {
    const g = parseSeq(toks, i + 1, '}');
    let j = g.i; if (toks[j] && toks[j].t === '}') j++;
    return { node: `<mrow>${g.node}</mrow>`, i: j };
  }
  if (tk.t === 'num') return { node: `<mn>${escapeHTML(tk.v)}</mn>`, i: i + 1 };
  if (tk.t === 'ident') return { node: `<mi>${escapeHTML(tk.v)}</mi>`, i: i + 1 };
  if (tk.t === 'op') return { node: `<mo>${escapeHTML(tk.v)}</mo>`, i: i + 1 };
  if (tk.t === '^' || tk.t === '_') return { node: '<mi></mi>', i };
  if (tk.t === 'cmd') {
    const name = tk.v;
    if (name === 'frac') { const a = parseAtom(toks, i + 1); const b = parseAtom(toks, a.i); return { node: `<mfrac>${a.node}${b.node}</mfrac>`, i: b.i }; }
    if (name === 'sqrt') { const a = parseAtom(toks, i + 1); return { node: `<msqrt>${a.node}</msqrt>`, i: a.i }; }
    if (name === 'left' || name === 'right') {
      const d = toks[i + 1];
      if (d && (d.t === 'op' || d.t === 'cmd')) { const sym = d.t === 'op' ? d.v : (OPS[d.v] || ''); return { node: `<mo>${escapeHTML(sym)}</mo>`, i: i + 2 }; }
      return { node: '', i: i + 1 };
    }
    if (Object.prototype.hasOwnProperty.call(GREEK, name)) return { node: `<mi>${GREEK[name]}</mi>`, i: i + 1 };
    if (Object.prototype.hasOwnProperty.call(OPS, name)) return { node: `<mo>${OPS[name]}</mo>`, i: i + 1 };
    if (FUNCS.indexOf(name) !== -1) return { node: `<mi>${name}</mi>`, i: i + 1 };
    if (name === '{' || name === '}' || name === ',' || name === ';' || name === ' ') return { node: `<mo>${escapeHTML(name.trim())}</mo>`, i: i + 1 };
    throw new Error('unsupported LaTeX command: \\' + name);
  }
  return { node: '<mi></mi>', i: i + 1 };
}
function latexToMathML(latex) {
  const seq = parseSeq(tokenizeLatex(latex), 0, null);
  return `<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow>${seq.node}</mrow></math>`;
}
function renderMath(host, latex) {
  const box = document.createElement('div'); box.className = 'mathblock';
  try { box.innerHTML = latexToMathML(latex); }
  catch (e) {
    const code = document.createElement('code'); code.className = 'math-fallback';
    code.textContent = latex; box.appendChild(code);
    box.title = 'Unsupported LaTeX construct — showing source';
  }
  host.appendChild(box);
}

// --- WASM kernel hook -------------------------------------------------------
const WASM = { ready: false, mod: null, active: false, coeffPtr: 1024 };
const COEFF_PTR_FALLBACK = 1024;
async function initWasm() {
  try {
    const resp = await fetch('kernel.wasm');
    if (!resp || !resp.ok) return false;
    const { instance } = await WebAssembly.instantiate(await resp.arrayBuffer(), {});
    const ex = instance.exports;
    if (typeof ex.poly_eval !== 'function' || !(ex.memory instanceof WebAssembly.Memory)) return false;
    // Query the kernel's exported scratch-buffer offset; fall back to the fixed offset.
    WASM.coeffPtr = (typeof ex.coeff_buffer === 'function') ? ex.coeff_buffer() : COEFF_PTR_FALLBACK;
    WASM.mod = instance; WASM.ready = true; return true;
  } catch (e) { return false; }
}
function polyJS(c, x) { let a = 0; for (let k = c.length - 1; k >= 0; k--) a = a * x + c[k]; return a; }
function polyWasm(c, x) { const ex = WASM.mod.exports; new Float64Array(ex.memory.buffer, WASM.coeffPtr, c.length).set(c); return ex.poly_eval(WASM.coeffPtr, c.length, x); }
function samplePolynomial() {
  const coeffs = [0.5, -0.8, -0.3, 0.12];
  let useWasm = WASM.ready; const N = 240, x0 = -4, x1 = 4, pts = [];
  // Degrade to the JS Horner path on any WASM error (e.g. a bad offset -> RangeError).
  const evalAt = (x) => { if (useWasm) { try { return polyWasm(coeffs, x); } catch (e) { useWasm = false; } } return polyJS(coeffs, x); };
  for (let k = 0; k <= N; k++) { const x = x0 + (x1 - x0) * k / N; pts.push([x, evalAt(x)]); }
  WASM.active = useWasm;
  return { type: 'plot', title: 'Live polynomial p(x) = 0.5 − 0.8x − 0.3x² + 0.12x³', xLabel: 'x', yLabel: 'p(x)', xRange: [x0, x1], yRange: null,
    series: [{ kind: 'line', label: useWasm ? 'WASM kernel' : 'JS kernel', color: '#a855f7', points: pts }] };
}
function buildLiveControls() {
  const box = document.createElement('div'); box.className = 'controls';
  const btn = document.createElement('button'); btn.className = 'btn'; btn.textContent = 'Sample a polynomial live';
  const tag = document.createElement('span'); tag.className = 'live-tag'; tag.textContent = WASM.ready ? 'kernel.wasm loaded' : 'JS kernel (no kernel.wasm)';
  const hint = document.createElement('span'); hint.className = 'hint'; hint.textContent = 'Evaluates a cubic across x∈[−4,4] and plots it through the active renderer.';
  const target = document.createElement('div'); target.style.width = '100%';
  btn.addEventListener('click', () => {
    try {
      target.innerHTML = '';
      const fig = document.createElement('figure'); fig.className = 'plot'; target.appendChild(fig);
      renderPlot(fig, samplePolynomial(), { onBackendChange: () => setBadge('cpu') });
      const cap = document.createElement('figcaption');
      cap.textContent = (WASM.active ? 'Computed via WASM kernel' : 'Computed in JS') + ' · rendered via ' + (GPU.ok ? 'WebGPU' : 'Canvas2D');
      fig.appendChild(cap);
    } catch (e) {
      target.innerHTML = '';
      const n = document.createElement('div'); n.className = 'notice'; n.textContent = 'Live sample failed: ' + e.message; target.appendChild(n);
    }
  });
  box.append(btn, tag, hint, target);
  return box;
}

// --- full symbolic + linear-algebra engine (WASM) ---------------------------
// The REAL exact-over-Q CAS — the symbolic core PLUS the numeric/linear-algebra chain
// (simd, polynomial, ratpoly, matrix, roots, numeric, matdecomp, bandsolve, eigen) — compiled
// to WebAssembly, loaded lazily as an ES module. Unlike the freestanding kernel.wasm above
// (raw poly_eval), this is the whole engine behind two C ABI entry points:
//   nimblecas_eval_latex(expr)     text -> parse -> simplify -> LaTeX
//   nimblecas_matrix_det_latex(m)  "a,b;c,d" (semicolon rows, comma cells) -> exact
//                                  determinant -> LaTeX (exercises the linalg chain)
// Degrades gracefully to "engine unavailable" if nimblecas.js / nimblecas.wasm are not served.
const CAS = { ready: false, evalLatex: null, det: null, sampleFn: null };
async function initCAS() {
  try {
    const { default: NimbleCAS } = await import('./nimblecas.js');
    const mod = await NimbleCAS();
    CAS.evalLatex = (src) => mod.ccall('nimblecas_eval_latex', 'string', ['string'], [src]);
    CAS.det = (src) => mod.ccall('nimblecas_matrix_det_latex', 'string', ['string'], [src]);
    // Sample y = f(x) across [x0, x1] at n points through the exact engine (evalnum).
    // Returns the parsed JSON { points: [[x, y], ...] } or { error: "..." }.
    CAS.sampleFn = (expr, x0, x1, n) => JSON.parse(mod.ccall(
      'nimblecas_sample_function_json', 'string',
      ['string', 'number', 'number', 'number'], [expr, x0, x1, n]));
    CAS.ready = true;
  } catch (e) {
    CAS.ready = false;  // no engine served -> CAS cells/REPL show an inline hint
  }
  return CAS.ready;
}

// Render one executable CAS cell: the input expression, then its evaluated result as math
// (parse -> simplify -> LaTeX -> MathML through the same renderer as static math blocks).
function renderCasCell(host, source) {
  const cell = document.createElement('div'); cell.className = 'cas-cell';
  const src = document.createElement('div'); src.className = 'cas-src';
  src.textContent = '› ' + source;  // '› expr'
  cell.appendChild(src);
  if (CAS.ready) {
    try { renderMath(cell, CAS.evalLatex(source)); }
    catch (e) { const n = document.createElement('div'); n.className = 'cas-note'; n.textContent = 'eval failed: ' + e.message; cell.appendChild(n); }
  } else {
    const n = document.createElement('div'); n.className = 'cas-note';
    n.textContent = 'CAS engine not loaded (serve nimblecas.js + nimblecas.wasm).';
    cell.appendChild(n);
  }
  host.appendChild(cell);
}

// Render one executable matrix-determinant cell: "a,b;c,d" -> det(...) rendered as math,
// exercising the linear-algebra chain (matrix -> ratpoly -> polynomial -> simd) end to end.
function renderDetCell(host, source) {
  const cell = document.createElement('div'); cell.className = 'cas-cell';
  const src = document.createElement('div'); src.className = 'cas-src';
  src.textContent = '› det(' + source + ')';
  cell.appendChild(src);
  if (CAS.ready) {
    try { renderMath(cell, CAS.det(source)); }
    catch (e) { const n = document.createElement('div'); n.className = 'cas-note'; n.textContent = 'eval failed: ' + e.message; cell.appendChild(n); }
  } else {
    const n = document.createElement('div'); n.className = 'cas-note';
    n.textContent = 'CAS engine not loaded (serve nimblecas.js + nimblecas.wasm).';
    cell.appendChild(n);
  }
  host.appendChild(cell);
}

// A live CAS REPL: type an expression, see the real engine parse + simplify + render it.
function buildCasRepl() {
  const box = document.createElement('div'); box.className = 'controls cas-repl';
  const hint = document.createElement('span'); hint.className = 'hint';
  hint.textContent = CAS.ready
    ? 'Live CAS (exact over ℚ) — type an expression:'
    : 'Live CAS unavailable — serve nimblecas.js + nimblecas.wasm next to this page.';
  const input = document.createElement('input'); input.className = 'cas-input'; input.type = 'text';
  input.placeholder = 'e.g.  2/4 + 1/4    or    (x + 1)^2    or    sin(x) + sin(x)';
  input.disabled = !CAS.ready;
  const out = document.createElement('div'); out.className = 'cas-out';
  const run = () => {
    const s = input.value.trim(); out.innerHTML = '';
    if (!s || !CAS.ready) return;
    try { renderMath(out, CAS.evalLatex(s)); }
    catch (e) { const n = document.createElement('div'); n.className = 'cas-note'; n.textContent = 'eval failed: ' + e.message; out.appendChild(n); }
  };
  input.addEventListener('input', run);
  box.append(hint, input, out);
  return box;
}

// Render one executable function-plot cell: sample y = f(x) across a range through the exact
// engine (nimblecas_sample_function_json -> evalnum) and plot it via the active renderer.
// block: { kind:'plot-fn', expr:'sin(x)*x', xRange:[x0,x1]?, n:240?, title?, color? }
function renderFnPlot(host, block) {
  const fig = document.createElement('figure'); fig.className = 'plot'; host.appendChild(fig);
  const expr = String(block.expr || '').trim();
  const range = Array.isArray(block.xRange) && block.xRange.length === 2 ? block.xRange : [-4, 4];
  const x0 = Number(range[0]), x1 = Number(range[1]);
  const n = Number.isFinite(block.n) ? Math.max(2, Math.min(4000, block.n | 0)) : 240;
  const fail = (msg) => { const c = document.createElement('figcaption'); c.textContent = msg; fig.appendChild(c); };
  if (!expr) { fail('Empty function expression.'); return; }
  if (!CAS.ready || !CAS.sampleFn) { fail('CAS engine not loaded — cannot sample f(x). Serve nimblecas.js + nimblecas.wasm.'); return; }
  let res;
  try { res = CAS.sampleFn(expr, x0, x1, n); }
  catch (e) { fail('sample failed: ' + e.message); return; }
  if (!res || res.error || !Array.isArray(res.points) || res.points.length === 0) {
    fail('f(x) = ' + expr + ' — ' + ((res && res.error) || 'no finite samples')); return;
  }
  const title = block.title || ('f(x) = ' + expr);
  const spec = { type: 'plot', title, xLabel: 'x', yLabel: 'f(x)', xRange: [x0, x1], yRange: null,
    series: [{ kind: 'line', label: expr, color: block.color || '#0ea5e9', points: res.points }] };
  try { renderPlot(fig, spec, { onBackendChange: () => setBadge('cpu') }); }
  catch (e) { fail('render failed: ' + e.message); return; }
  const cap = document.createElement('figcaption');
  cap.textContent = title + ' · ' + res.points.length + ' samples via nimblecas.wasm (evalnum)';
  fig.appendChild(cap);
}

// --- document renderer ------------------------------------------------------
let CURRENT_DOC = null;

// Render a single block's CONTENT into `host` (no editing chrome).
function renderBlockContent(host, block) {
  if (block.kind === 'prose') { const d = document.createElement('div'); d.className = 'prose'; d.innerHTML = renderProse(block.text || ''); host.appendChild(d); }
  else if (block.kind === 'math') renderMath(host, block.latex || '');
  else if (block.kind === 'nimblecas') renderCasCell(host, block.source || '');
  else if (block.kind === 'nimblecas-det') renderDetCell(host, block.source || '');
  else if (block.kind === 'plot-fn') renderFnPlot(host, block);
  else if (block.kind === 'plot' && block.plot) {
    const fig = document.createElement('figure'); fig.className = 'plot'; host.appendChild(fig);
    renderPlot(fig, block.plot, { onBackendChange: () => setBadge('cpu') });
    if (block.plot.title) { const cap = document.createElement('figcaption'); cap.textContent = block.plot.title; fig.appendChild(cap); }
  }
}

// --- block CRUD (mutate CURRENT_DOC.blocks in place, then re-render) ---------
function moveBlock(i, delta) {
  const b = CURRENT_DOC && CURRENT_DOC.blocks; if (!Array.isArray(b)) return;
  const j = i + delta; if (j < 0 || j >= b.length) return;
  [b[i], b[j]] = [b[j], b[i]]; renderDocument(CURRENT_DOC);
}
function deleteBlock(i) {
  const b = CURRENT_DOC && CURRENT_DOC.blocks; if (!Array.isArray(b)) return;
  b.splice(i, 1); renderDocument(CURRENT_DOC);
}
function addBlock(block) {
  if (!CURRENT_DOC) return;
  if (!Array.isArray(CURRENT_DOC.blocks)) CURRENT_DOC.blocks = [];
  CURRENT_DOC.blocks.push(block); renderDocument(CURRENT_DOC);
}

// The "add a cell" toolbar: pick a kind, type its content, append it to the document.
function buildAddCell() {
  const box = document.createElement('div'); box.className = 'controls addcell';
  const sel = document.createElement('select'); sel.className = 'cell-kind';
  [['prose', 'Prose'], ['math', 'Math (LaTeX)'], ['nimblecas', 'CAS cell'],
   ['nimblecas-det', 'Determinant'], ['plot-fn', 'Plot f(x)']].forEach(([v, l]) => {
    const o = document.createElement('option'); o.value = v; o.textContent = l; sel.appendChild(o);
  });
  const input = document.createElement('input'); input.className = 'cell-input'; input.type = 'text';
  const placeholders = {
    prose: 'markdown text (headings, **bold**, `code`, - lists)',
    math: 'LaTeX, e.g.  e^x = \\sum x^n/n!',
    nimblecas: 'expression, e.g.  (a + b)*(a - b)',
    'nimblecas-det': 'matrix "a,b;c,d", e.g.  1,2;3,4',
    'plot-fn': 'function of x, e.g.  sin(x)*x   (plotted over [-4, 4])',
  };
  input.placeholder = placeholders[sel.value];
  sel.addEventListener('change', () => { input.placeholder = placeholders[sel.value] || ''; });
  const add = document.createElement('button'); add.className = 'btn'; add.textContent = 'Add cell';
  const doAdd = () => {
    const kind = sel.value; const v = input.value.trim();
    if (!v && kind !== 'prose') return;
    let block;
    if (kind === 'prose') block = { kind: 'prose', text: v };
    else if (kind === 'math') block = { kind: 'math', latex: v };
    else if (kind === 'nimblecas') block = { kind: 'nimblecas', source: v };
    else if (kind === 'nimblecas-det') block = { kind: 'nimblecas-det', source: v };
    else if (kind === 'plot-fn') block = { kind: 'plot-fn', expr: v, xRange: [-4, 4], n: 240 };
    else return;
    input.value = ''; addBlock(block);
  };
  add.addEventListener('click', doAdd);
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); doAdd(); } });
  const hint = document.createElement('span'); hint.className = 'hint';
  hint.textContent = 'Add a cell to the document (edits live in the browser; use “Open document…” to load JSON, or the buttons above each block to reorder/remove).';
  box.append(sel, input, add, hint);
  return box;
}

function renderDocument(doc) {
  const app = document.getElementById('app');
  app.innerHTML = '';
  if (!doc || doc.type !== 'document') {
    const n = document.createElement('div'); n.className = 'notice';
    n.textContent = 'Not a NimbleCAS document (expected {"type":"document", ...}).';
    app.appendChild(n); return;
  }
  if (doc.title) { const h = document.createElement('div'); h.className = 'doc-title'; h.textContent = doc.title; app.appendChild(h); }
  const blocks = Array.isArray(doc.blocks) ? doc.blocks : [];
  blocks.forEach((block, idx) => {
    if (!block || !block.kind) return;
    const wrap = document.createElement('div'); wrap.className = 'block';
    const tools = document.createElement('div'); tools.className = 'block-tools';
    const mk = (label, title, fn) => {
      const b = document.createElement('button'); b.className = 'block-btn'; b.type = 'button';
      b.textContent = label; b.title = title; b.setAttribute('aria-label', title);
      b.addEventListener('click', fn); return b;
    };
    tools.append(
      mk('↑', 'Move cell up', () => moveBlock(idx, -1)),
      mk('↓', 'Move cell down', () => moveBlock(idx, 1)),
      mk('✕', 'Delete cell', () => deleteBlock(idx)));
    wrap.appendChild(tools);
    const content = document.createElement('div'); content.className = 'block-content';
    try { renderBlockContent(content, block); }
    catch (e) {
      const err = document.createElement('div'); err.className = 'notice';
      err.textContent = 'Skipped a malformed ' + block.kind + ' block: ' + e.message; content.appendChild(err);
    }
    wrap.appendChild(content);
    app.appendChild(wrap);
  });
  app.appendChild(buildAddCell());
  app.appendChild(buildLiveControls());
  app.appendChild(buildCasRepl());
}
function rerender() { if (CURRENT_DOC) renderDocument(CURRENT_DOC); }

// --- badge / file input / resize --------------------------------------------
function setBadge(kind) {
  const b = document.getElementById('backendBadge'); if (!b) return;
  b.classList.remove('gpu', 'cpu');
  if (kind === 'gpu') { b.classList.add('gpu'); b.textContent = 'WebGPU'; }
  else { b.classList.add('cpu'); b.textContent = 'Canvas2D fallback'; }
}
function wireFileInput() {
  const input = document.getElementById('fileInput');
  input.addEventListener('change', (ev) => {
    const file = ev.target.files && ev.target.files[0]; if (!file) return;
    const showNotice = (msg) => { const app = document.getElementById('app'); app.innerHTML = ''; const n = document.createElement('div'); n.className = 'notice'; n.textContent = msg; app.appendChild(n); };
    const reader = new FileReader();
    reader.onerror = () => showNotice('Could not read the file.');
    reader.onload = () => {
      let doc;
      try { doc = JSON.parse(reader.result); }
      catch (e) { showNotice('Could not parse JSON: ' + e.message); return; }
      // Render errors are distinct from parse errors; renderDocument also isolates
      // each block, so a partial document still shows.
      try { CURRENT_DOC = doc; renderDocument(doc); }
      catch (e) { showNotice('Could not render document: ' + e.message); }
    };
    reader.readAsText(file);
    ev.target.value = '';  // re-selecting the same file fires 'change' again
  });
}

// --- embedded sample --------------------------------------------------------
function buildSampleDocument() {
  const TWO_PI = Math.PI * 2, sinPts = [], cosPts = [];
  for (let k = 0; k <= 160; k++) { const x = -TWO_PI + 2 * TWO_PI * k / 160; sinPts.push([x, Math.sin(x)]); cosPts.push([x, Math.cos(x)]); }
  const scatter = [], fit = [], rand = (i) => { const s = Math.sin(i * 12.9898) * 43758.5453; return s - Math.floor(s); };
  for (let i = 0; i <= 44; i++) { const x = i / 44 * 10; scatter.push([x, 0.5 * x + 1 + (rand(i) - 0.5) * 1.6]); }
  fit.push([0, 1], [10, 6]);
  return {
    type: 'document', title: 'NimbleCAS — Exact Computer Algebra',
    blocks: [
      { kind: 'prose', text: '# Welcome to NimbleCAS\n\n**NimbleCAS** is a C++23 exact computer-algebra system. This viewer renders **prose**, **math**, and **plots** with zero external dependencies.\n\n## Block kinds\n\n- Prose uses a minimal Markdown subset (headings, `code`, **bold**, lists).\n- Math converts a small LaTeX subset to native **MathML**.\n- Plots render through a **WebGPU** pipeline with a **Canvas2D** fallback.' },
      { kind: 'prose', text: '## A little mathematics\n\nThe exponential function has the Taylor series:' },
      { kind: 'math', latex: 'e^x = \\sum_{n=0}^{\\infty} \\frac{x^n}{n!}' },
      { kind: 'prose', text: 'The 2×2 determinant is a classic exact result:' },
      { kind: 'math', latex: '\\det(A) = a_{11} a_{22} - a_{12} a_{21}' },
      { kind: 'prose', text: 'And a Gaussian integral:' },
      { kind: 'math', latex: '\\int_{-\\infty}^{\\infty} e^{-x^2} \\, dx = \\sqrt{\\pi}' },
      { kind: 'prose', text: '## Plots' },
      { kind: 'plot', plot: { type: 'plot', title: 'Trigonometric functions', xLabel: 'x (radians)', yLabel: 'value', xRange: [-TWO_PI, TWO_PI], yRange: [-1.2, 1.2],
        series: [ { kind: 'line', label: 'sin(x)', color: '#2563eb', points: sinPts }, { kind: 'line', label: 'cos(x)', color: '#e0533d', points: cosPts } ] } },
      { kind: 'plot', plot: { type: 'plot', title: 'Noisy samples with linear trend', xLabel: 'x', yLabel: 'y', xRange: null, yRange: null,
        series: [ { kind: 'scatter', label: 'samples', color: '#12a150', points: scatter }, { kind: 'line', label: 'y = 0.5x + 1', color: '#a855f7', points: fit } ] } },
      { kind: 'prose', text: '## Executable CAS cells\n\nThese `nimblecas` cells are evaluated **in the browser** by the real exact engine (compiled to WebAssembly) — parsed, simplified, and rendered. Everything stays exact over ℚ:' },
      { kind: 'nimblecas', source: '2/4 + 1/4' },
      { kind: 'nimblecas', source: '(a + b)*(a - b)' },
      { kind: 'nimblecas', source: 'x^2 + 3*x^2' },
      { kind: 'nimblecas', source: 'sin(x) + sin(x)' },
      { kind: 'prose', text: '## Linear algebra, in the browser\n\nThe symbolic engine\'s numeric/linear-algebra chain (`matrix` → `ratpoly` → `polynomial` → `simd`) is compiled into the same WASM module. Determinants are computed exactly over ℚ:' },
      { kind: 'nimblecas-det', source: '1,2;3,4' },
      { kind: 'nimblecas-det', source: '2,0,0;0,3,0;0,0,4' },
      { kind: 'nimblecas-det', source: '1/2,1;1,1' },
      { kind: 'prose', text: '## Plot a function through the exact engine\n\nThis `plot-fn` cell samples `f(x)` across a range by calling the WASM engine\'s `nimblecas_sample_function_json` endpoint — every sample evaluated by `evalnum` (exact ℚ, collapsed to a double only at the leaves) — then drawn through the same WebGPU/Canvas2D renderer. Add your own with the **Plot f(x)** cell type below.' },
      { kind: 'plot-fn', expr: 'x^3 - 3*x', xRange: [-3, 3], n: 240, title: 'f(x) = x³ − 3x' },
      { kind: 'prose', text: '## Live compute\n\nType your own expression in the CAS box below, or sample a polynomial through the plot renderer. The CAS runs `nimblecas.wasm` (the whole symbolic engine); the polynomial control runs the freestanding `kernel.wasm` if present, else JavaScript.' },
    ],
  };
}

// --- boot -------------------------------------------------------------------
(async function boot() {
  wireFileInput();
  const gpuOk = await initWebGPU(null, () => { setBadge('cpu'); rerender(); });
  setBadge(gpuOk ? 'gpu' : 'cpu');
  await initWasm();
  await initCAS();
  CURRENT_DOC = buildSampleDocument();
  renderDocument(CURRENT_DOC);
  let t = null;
  window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(rerender, 150); });
  if (window.matchMedia) {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    (mq.addEventListener ? mq.addEventListener.bind(mq, 'change') : mq.addListener.bind(mq))(rerender);
  }
})();
