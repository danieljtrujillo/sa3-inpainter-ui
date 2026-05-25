<script>
import { onMount, untrack } from "svelte";
import { session } from "./session.svelte.js";

let overviewCanvas = $state(null);  // full-track backdrop (rendered once per audio load)
let canvas = $state(null);          // detail layer, retargeted to visible region

let pcm = null;             // Float32Array, mono
let sampleRate = 44100;
let worker = null;

// ranges (normalized [0,1]) currently drawn into each canvas
let overviewRange = $state({ start: 0, end: 1 });
let renderedRange = $state({ start: 0, end: 1 });

// scheduling / lifecycle
let renderToken = 0;
let inFlight = false;
let queuedRange = null;
let renderTimer = 0;
let loadedVersion = -1;
let scratch = null;          // offscreen scratch for atomic blit

// scan + first-render coordination
let dbMax = null;
let pendingFirstRender = false;

const N_FFT = 8192;
const PAD_FACTOR = 0;
const MIN_VISIBLE_FRAC = 0.7;
const MAX_TILES = 40;            // LRU cap; ~40 * (canvasW * canvasH * 4 bytes) ≈ <100MB

// LRU cache of previously-rendered detail tiles. Each tile is a stand-alone
// HTMLCanvasElement holding the painted pixels for a specific (start, end) range.
// On zoom change we paint the best-matching cached tile FIRST so the user
// always sees a sharp image (or near-sharp), then schedule a fresh worker
// render to replace it. The overview canvas is the last-resort fallback.
const tileCache = new Map();     // key → { range:{start,end}, canvas, w, h }
const tileOrder = [];            // LRU: oldest first

function tileKey(start, end) {
  return `${start.toFixed(5)}_${end.toFixed(5)}`;
}

function cacheStore(range, sourceCanvas) {
  const key = tileKey(range.start, range.end);
  const dst = document.createElement("canvas");
  dst.width = sourceCanvas.width;
  dst.height = sourceCanvas.height;
  dst.getContext("2d").drawImage(sourceCanvas, 0, 0);
  if (tileCache.has(key)) {
    const i = tileOrder.indexOf(key); if (i >= 0) tileOrder.splice(i, 1);
  }
  tileCache.set(key, { range: { ...range }, canvas: dst, w: dst.width, h: dst.height });
  tileOrder.push(key);
  while (tileOrder.length > MAX_TILES) {
    const evict = tileOrder.shift();
    tileCache.delete(evict);
  }
}

function cacheClear() {
  tileCache.clear();
  tileOrder.length = 0;
}

// Find the cached tile that best covers the current visible range. Prefer the
// SMALLEST range that fully contains visible (most detailed coverage). If no
// containing tile exists, prefer the LARGEST overlap (better than overview).
function cacheBest() {
  const vs = session.zoomStart, ve = session.zoomEnd;
  let containing = null;          // smallest span that covers visible
  let containingSpan = Infinity;
  let overlap = null;             // best partial overlap fallback
  let overlapAmount = -1;
  for (const t of tileCache.values()) {
    const rs = t.range.start, re = t.range.end;
    if (rs <= vs && re >= ve) {
      const span = re - rs;
      if (span < containingSpan) { containing = t; containingSpan = span; }
    } else {
      const o = Math.min(re, ve) - Math.max(rs, vs);
      if (o > overlapAmount) { overlapAmount = o; overlap = t; }
    }
  }
  return containing || overlap || null;
}

function paintFromCache() {
  const t = cacheBest();
  if (!t || !canvas) return false;
  // Promote in LRU
  const k = tileKey(t.range.start, t.range.end);
  const i = tileOrder.indexOf(k); if (i >= 0) { tileOrder.splice(i, 1); tileOrder.push(k); }
  if (canvas.width !== t.w || canvas.height !== t.h) {
    canvas.width = t.w; canvas.height = t.h;
  }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(t.canvas, 0, 0);
  renderedRange = { ...t.range };
  return true;
}

// ─── CSS transforms (live, GPU-cheap) ───────────────────────────────────────
function rangeToVisibleStyle(rs, re) {
  const rspan = Math.max(1e-9, re - rs);
  const v0 = (session.zoomStart - rs) / rspan;
  const v1 = (session.zoomEnd   - rs) / rspan;
  const sx = 1 / Math.max(1e-9, v1 - v0);
  const tx = -v0 * 100;
  return `transform: scaleX(${sx}) translateX(${tx}%); transform-origin: 0 0;`;
}
let detailTransform = $derived.by(() => rangeToVisibleStyle(renderedRange.start, renderedRange.end));
let overviewTransform = $derived.by(() => rangeToVisibleStyle(overviewRange.start, overviewRange.end));

// ─── audio load + global scan ──────────────────────────────────────────────
async function loadAudio(v) {
  if (v === loadedVersion) return;
  try {
    const r = await fetch(`/api/audio?v=${v}`);
    if (!r.ok) return;
    const buf = await r.arrayBuffer();
    const ac = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(1, 44100, 44100);
    const decoded = await ac.decodeAudioData(buf);
    sampleRate = decoded.sampleRate;
    if (decoded.numberOfChannels > 1) {
      const ch0 = decoded.getChannelData(0);
      const ch1 = decoded.getChannelData(1);
      const mono = new Float32Array(ch0.length);
      for (let i = 0; i < ch0.length; i++) mono[i] = (ch0[i] + ch1[i]) * 0.5;
      pcm = mono;
    } else {
      pcm = decoded.getChannelData(0).slice();
    }
    loadedVersion = v;
    renderedRange = { start: 0, end: 1 };
    overviewRange = { start: 0, end: 1 };
    dbMax = null;
    pendingFirstRender = true;
    cacheClear();
    scanGlobal();
  } catch (e) {
    console.warn("[spec] audio load/decode failed:", e);
  }
}

function scanGlobal() {
  if (!worker || !pcm) return;
  worker.postMessage({
    op: "scan",
    audio: pcm,
    sampleRate,
    nFft: N_FFT,
    samples: 200,
  });
}

// ─── render scheduling ─────────────────────────────────────────────────────
function scheduleRender(immediate = false) {
  if (renderTimer) clearTimeout(renderTimer);
  if (immediate) { renderDetail(); return; }
  renderTimer = setTimeout(renderDetail, 140);
}

function computeDetailRange() {
  const vSpan = Math.max(1e-9, session.zoomEnd - session.zoomStart);
  let s = session.zoomStart - vSpan * PAD_FACTOR;
  let e = session.zoomEnd   + vSpan * PAD_FACTOR;
  if (s < 0) { e -= s; s = 0; }
  if (e > 1) { s -= (e - 1); e = 1; }
  return { start: Math.max(0, s), end: Math.min(1, e) };
}

function paneDimsFor(c) {
  if (!c) return { w: 0, h: 0 };
  const dpr = window.devicePixelRatio || 1;
  return {
    w: Math.max(1, Math.floor(c.offsetWidth  * dpr)),
    h: Math.max(1, Math.floor(c.offsetHeight * dpr)),
  };
}

function renderOverview() {
  if (!overviewCanvas || !pcm || !worker || dbMax === null) return;
  const { w, h } = paneDimsFor(overviewCanvas);
  if (w <= 0 || h <= 0) return;
  // overview can be coarse — half resolution is fine since it's a backdrop
  const ow = Math.max(256, Math.floor(w / 2));
  // .slice() returns an independent copy, so transferring its buffer doesn't
  // touch our `pcm` reference. The detail renderer can fire in parallel.
  const slice = pcm.slice();
  worker.postMessage({
    op: "render",
    target: "overview",
    token: -1,
    audio: slice,
    sampleRate,
    width: ow,
    height: h,
    nFft: N_FFT,
    dbMax,
  }, [slice.buffer]);
}

// pick an FFT window proportional to the hop so deep zooms get finer time
// localization (and shallow zooms keep good frequency resolution).
function chooseNFft(visibleSamples, widthPx) {
  const hop = Math.max(1, Math.floor(visibleSamples / widthPx));
  let n = hop * 4;
  let p = 256;
  while (p < n) p *= 2;
  return Math.max(256, Math.min(8192, p));
}

function renderDetail() {
  if (!canvas || !pcm || !worker) return;
  if (dbMax === null) { pendingFirstRender = true; return; }
  const { w, h } = paneDimsFor(canvas);
  if (w <= 0 || h <= 0) return;

  const target = computeDetailRange();
  if (inFlight) { queuedRange = target; return; }

  // Paint the best cached tile NOW so the user sees something sharp while the
  // worker chews on the precise render. No-op if there's no useful tile yet.
  paintFromCache();

  const startSample = Math.max(0, Math.floor(target.start * pcm.length));
  const endSample   = Math.min(pcm.length, Math.ceil(target.end * pcm.length));
  if (endSample - startSample < 4) return;

  inFlight = true;
  const token = ++renderToken;
  const slice = pcm.slice(startSample, endSample);
  // fixed N_FFT: adaptive windows break brightness consistency (per-bin power
  // depends on N) and smear the bass. Real fix needs multi-resolution / CQT —
  // see TODO below.
  worker._pendingRange = target;
  worker.postMessage({
    op: "render",
    target: "detail",
    token,
    audio: slice,
    sampleRate,
    width: w,
    height: h,
    nFft: N_FFT,
    dbMax,
  }, [slice.buffer]);
}

function blit(targetCanvas, m) {
  if (!scratch) scratch = document.createElement("canvas");
  if (scratch.width !== m.width || scratch.height !== m.height) {
    scratch.width = m.width;
    scratch.height = m.height;
  }
  const sctx = scratch.getContext("2d");
  const img = new ImageData(new Uint8ClampedArray(m.buf), m.width, m.height);
  sctx.putImageData(img, 0, 0);
  if (targetCanvas.width !== m.width || targetCanvas.height !== m.height) {
    targetCanvas.width = m.width;
    targetCanvas.height = m.height;
  }
  const ctx = targetCanvas.getContext("2d");
  ctx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
  ctx.drawImage(scratch, 0, 0);
}

function onWorkerMessage(e) {
  const m = e.data;
  if (m.op === "scan") {
    dbMax = m.dbRef;
    // first render after scan: overview (full track backdrop) then detail
    renderOverview();
    if (pendingFirstRender) { pendingFirstRender = false; scheduleRender(true); }
    return;
  }
  if (m.op === "render") {
    if (m.target === "overview") {
      if (m.error) { console.warn("[spec] overview error:", m.error); return; }
      if (!overviewCanvas) return;
      blit(overviewCanvas, m);
      overviewRange = { start: 0, end: 1 };
      // also seed the cache with the overview tile (low-res but better than black)
      cacheStore({ start: 0, end: 1 }, overviewCanvas);
      return;
    }
    // detail
    inFlight = false;
    if (m.token !== renderToken) {
      if (queuedRange) { queuedRange = null; scheduleRender(true); }
      return;
    }
    if (m.error) { console.warn("[spec] detail error:", m.error); return; }
    if (!canvas) return;
    blit(canvas, m);
    renderedRange = worker._pendingRange;
    cacheStore(renderedRange, canvas);
    if (queuedRange) { queuedRange = null; scheduleRender(true); }
  }
}

// ─── reactivity ────────────────────────────────────────────────────────────
$effect(() => {
  if (session.hasAudio && session.version) {
    loadAudio(session.version);
  } else {
    pcm = null;
    loadedVersion = -1;
    dbMax = null;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    if (overviewCanvas) {
      const ctx = overviewCanvas.getContext("2d");
      ctx.clearRect(0, 0, overviewCanvas.width, overviewCanvas.height);
    }
  }
});

$effect(() => {
  // Live zoom/pan changes only trigger a debounced re-render — they DO NOT
  // touch the canvas. CSS transform on renderedRange→visible keeps the picture
  // smooth between frames. When the debounce fires, paintFromCache() instantly
  // swaps in the best cached tile (no black flash), then the worker computes a
  // sharp render in the background.
  // untrack renderedRange so updating it (from paint/render) doesn't re-fire
  // this effect (which would loop).
  session.zoomStart; session.zoomEnd;
  if (!pcm || dbMax === null) return;
  const { start: rs, end: re } = untrack(() => renderedRange);
  const visSpan = Math.max(1e-9, session.zoomEnd - session.zoomStart);
  const renSpan = re - rs;
  const out = session.zoomStart < rs - 1e-6 || session.zoomEnd > re + 1e-6;
  const tooBlurry = (visSpan / renSpan) < MIN_VISIBLE_FRAC;
  if (out || tooBlurry) scheduleRender();
});

onMount(() => {
  worker = new Worker(new URL("./spec_worker.js", import.meta.url), { type: "module" });
  worker.onmessage = onWorkerMessage;
  const ro = new ResizeObserver(() => scheduleRender());
  if (canvas) ro.observe(canvas);
  return () => {
    if (renderTimer) clearTimeout(renderTimer);
    ro.disconnect();
    worker?.terminate();
    worker = null;
  };
});
</script>

<div class="layers">
  <canvas bind:this={overviewCanvas} class="overview" style={overviewTransform}></canvas>
  <canvas bind:this={canvas} class="detail" style={detailTransform}></canvas>
</div>

<style>
.layers { position: relative; width: 100%; height: 100%; }
canvas {
  position: absolute;
  inset: 0;
  display: block;
  width: 100%;
  height: 100%;
  will-change: transform;
}
canvas.overview { background: #000; }
canvas.detail { background: transparent; }
</style>
