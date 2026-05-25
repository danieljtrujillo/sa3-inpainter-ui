<script>
import { onMount } from "svelte";
import { session } from "./session.svelte.js";

// Continuous PCM waveform layered over the colored per-latent WaveformCanvas.
// At zoom-out: min/max envelope per pixel. At deep zoom: individual sample
// points connected by lines, plus dots when samples_per_pixel < 1.

let canvas = $state(null);
let pcm = null;             // Float32Array, mono
let envelope = null;        // { sr, downsample, count, data:[[peak,r,g,b], ...] } — per-latent colors
let loadedVersion = -1;
let renderTimer = 0;

// Opacity fades in as you zoom in. The waveform isn't useful when 1 px
// covers thousands of samples; smoothly reveal it once each pixel is fine
// enough to actually trace the shape.
const FADE_HIDE_SPAN = 0.35;   // visible span >= this → invisible
const FADE_SHOW_SPAN = 0.05;   // visible span <= this → fully opaque
let zoomOpacity = $derived.by(() => {
  const span = Math.max(1e-9, session.zoomEnd - session.zoomStart);
  if (span >= FADE_HIDE_SPAN) return 0;
  if (span <= FADE_SHOW_SPAN) return 1;
  // smoothstep interpolation between the thresholds
  const t = (FADE_HIDE_SPAN - span) / (FADE_HIDE_SPAN - FADE_SHOW_SPAN);
  return t * t * (3 - 2 * t);
});

async function loadEnvelope(v) {
  try {
    const r = await fetch(`/api/envelope.json?v=${v}`);
    if (!r.ok) return;
    const j = await r.json();
    envelope = j.count > 0 ? j : null;
    scheduleRender();
  } catch {}
}

async function loadAudio(v) {
  if (v === loadedVersion) return;
  try {
    const r = await fetch(`/api/audio?v=${v}`);
    if (!r.ok) return;
    const buf = await r.arrayBuffer();
    const ac = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(1, 44100, 44100);
    const decoded = await ac.decodeAudioData(buf);
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
    loadEnvelope(v);
    scheduleRender();
  } catch (e) {
    console.warn("[waveform] audio load failed:", e);
  }
}

// Darker variant of the underlying latent color — same hue, lower brightness,
// so the waveform overlay stays visually distinct from the bars beneath.
const COLOR_DARKEN = 0.5;
function latentColorAtSample(s) {
  if (!envelope || !envelope.data || envelope.data.length === 0) return "rgb(0,0,0)";
  const idx = Math.max(0, Math.min(envelope.count - 1, Math.floor(s / envelope.downsample)));
  const d = envelope.data[idx];
  const r = Math.round(d[1] * 255 * COLOR_DARKEN);
  const g = Math.round(d[2] * 255 * COLOR_DARKEN);
  const b = Math.round(d[3] * 255 * COLOR_DARKEN);
  return `rgb(${r}, ${g}, ${b})`;
}

function scheduleRender() {
  if (renderTimer) cancelAnimationFrame(renderTimer);
  renderTimer = requestAnimationFrame(render);
}

function render() {
  renderTimer = 0;
  if (!canvas || !pcm) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.offsetWidth, cssH = canvas.offsetHeight;
  if (cssW <= 0 || cssH <= 0) return;
  const w = Math.floor(cssW * dpr);
  const h = Math.floor(cssH * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
  }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);

  const startSample = Math.max(0, Math.floor(session.zoomStart * pcm.length));
  const endSample   = Math.min(pcm.length, Math.ceil(session.zoomEnd * pcm.length));
  const visible = endSample - startSample;
  if (visible < 2) return;

  const samplesPerPx = visible / w;
  const midY = h * 0.5;
  const amp = h * 0.5 * 0.95;     // leave a tiny margin

  ctx.lineWidth = Math.max(1, dpr * 0.9);
  ctx.lineCap = "round";

  if (samplesPerPx >= 1.5) {
    // ---- Envelope mode: per-pixel min/max, colored by the latent at that pixel ----
    for (let x = 0; x < w; x++) {
      const s0 = startSample + Math.floor(x * samplesPerPx);
      const s1 = startSample + Math.floor((x + 1) * samplesPerPx);
      let mn = Infinity, mx = -Infinity;
      for (let i = s0; i < s1; i++) {
        const v = pcm[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      if (mn === Infinity) continue;
      const yTop = midY - mx * amp;
      const yBot = midY - mn * amp;
      ctx.strokeStyle = latentColorAtSample(s0);
      ctx.beginPath();
      ctx.moveTo(x + 0.5, yTop);
      ctx.lineTo(x + 0.5, Math.max(yTop + 1, yBot));
      ctx.stroke();
    }
  } else {
    // ---- Per-sample mode: segments colored by their latent ----
    const firstSample = startSample;
    const lastSample  = endSample;
    let prevColor = latentColorAtSample(firstSample);
    let prevX = (firstSample - startSample) / samplesPerPx;
    let prevY = midY - (pcm[firstSample] || 0) * amp;
    ctx.strokeStyle = prevColor;
    ctx.beginPath();
    ctx.moveTo(prevX, prevY);
    for (let i = firstSample + 1; i <= lastSample; i++) {
      const x = (i - startSample) / samplesPerPx;
      const y = midY - (pcm[i] || 0) * amp;
      const c = latentColorAtSample(i);
      if (c !== prevColor) {
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.strokeStyle = c;
        ctx.beginPath();
        ctx.moveTo(x, y);
        prevColor = c;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Dot the individual samples once they're at least ~6px apart
    if (samplesPerPx < 1 / 6) {
      const rDot = Math.min(4, Math.max(1.5, 1 / samplesPerPx * 0.2));
      for (let i = firstSample; i <= lastSample; i++) {
        const x = (i - startSample) / samplesPerPx;
        const y = midY - (pcm[i] || 0) * amp;
        ctx.fillStyle = latentColorAtSample(i);
        ctx.beginPath();
        ctx.arc(x, y, rDot, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // Centerline (subtle white)
  ctx.strokeStyle = "rgba(255, 255, 255, 0.18)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, midY);
  ctx.lineTo(w, midY);
  ctx.stroke();
}

$effect(() => {
  if (session.hasAudio && session.version) loadAudio(session.version);
  else {
    pcm = null;
    loadedVersion = -1;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }
});

$effect(() => {
  session.zoomStart; session.zoomEnd;
  if (pcm) scheduleRender();
});

onMount(() => {
  const ro = new ResizeObserver(() => scheduleRender());
  if (canvas) ro.observe(canvas);
  return () => {
    if (renderTimer) cancelAnimationFrame(renderTimer);
    ro.disconnect();
  };
});
</script>

<canvas bind:this={canvas} style="opacity: {zoomOpacity}"></canvas>

<style>
canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  display: block;
  transition: opacity 120ms ease-out;
}
</style>
