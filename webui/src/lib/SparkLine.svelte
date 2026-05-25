<script>
import { onMount } from "svelte";

let {
  values = [],
  color = "#0078ca",
  height = null,
  fill = true,
  xLabels = null,
  xLabel = "",
  yLabel = "",
  yPrecision = 3,
  // Synced hover cursor (set by parent so all charts move together)
  hoverFrac = null,       // 0..1 across plot area, null = no hover
  onHover = null,         // (frac) => void
  onLeave = null,         // () => void
} = $props();

let canvas = $state(null);
let wrap = $state(null);

const PAD_L = 38, PAD_B = 16, PAD_T = 4, PAD_R = 6;
const TICKS = 4;

function formatY(v) {
  if (!isFinite(v)) return "";
  const abs = Math.abs(v);
  if (abs >= 1000)   return v.toFixed(0);
  if (abs >= 10)     return v.toFixed(1);
  if (abs >= 1)      return v.toFixed(2);
  return v.toFixed(yPrecision);
}
function formatX(v) {
  if (typeof v !== "number") return String(v);
  if (v >= 10000) return (v / 1000).toFixed(1) + "k";
  if (v >= 1000)  return (v / 1000).toFixed(2) + "k";
  return String(v);
}

function layout(w, h) {
  const showAxes = w >= 90 && h >= 50;
  const padL = showAxes ? PAD_L : 1.5;
  const padB = showAxes ? PAD_B : 1.5;
  return { showAxes, padL, padB, padT: PAD_T, padR: PAD_R,
           plotW: Math.max(1, w - padL - PAD_R),
           plotH: Math.max(1, h - PAD_T - padB) };
}

function draw() {
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, canvas.offsetWidth);
  const h = Math.max(1, canvas.offsetHeight);
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width  = w * dpr;
    canvas.height = h * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const { showAxes, padL, padB, padT, padR, plotW, plotH } = layout(w, h);

  // axes frame
  if (showAxes) {
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + plotH);
    ctx.moveTo(padL, padT + plotH); ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();
  }

  if (!values || values.length < 1) {
    if (showAxes && yLabel) {
      ctx.fillStyle = "rgba(255,255,255,0.35)";
      ctx.font = "9px ui-monospace, Menlo, monospace";
      ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText(yLabel, 4, 2);
    }
    return;
  }

  let mn = Infinity, mx = -Infinity;
  for (const v of values) { if (v < mn) mn = v; if (v > mx) mx = v; }
  if (mn === mx) { mn -= 0.5; mx += 0.5; }
  const range = mx - mn;
  const pts = values.map((v, i) => ({
    x: padL + (values.length > 1 ? (i / (values.length - 1)) * plotW : plotW / 2),
    y: padT + plotH - ((v - mn) / range) * plotH,
  }));

  if (fill && pts.length >= 2) {
    ctx.beginPath();
    ctx.moveTo(pts[0].x, padT + plotH);
    for (const p of pts) ctx.lineTo(p.x, p.y);
    ctx.lineTo(pts[pts.length - 1].x, padT + plotH);
    ctx.closePath();
    ctx.fillStyle = color + "30";
    ctx.fill();
  }
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (const p of pts) ctx.lineTo(p.x, p.y);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.lineJoin = "round";
  ctx.stroke();

  if (showAxes) {
    // Y ticks
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.strokeStyle = "rgba(255,255,255,0.07)";
    ctx.font = "9px ui-monospace, Menlo, monospace";
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let i = 0; i <= TICKS; i++) {
      const v = mx - (range * i / TICKS);
      const y = padT + (plotH * i / TICKS);
      ctx.fillText(formatY(v), padL - 4, y);
      if (i > 0 && i < TICKS) {
        ctx.beginPath();
        ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y);
        ctx.stroke();
      }
    }
    // X ticks
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    const xTicks = Math.min(4, values.length);
    for (let i = 0; i < xTicks; i++) {
      const idx = Math.round((values.length - 1) * (i / Math.max(1, xTicks - 1)));
      const x = padL + (values.length > 1 ? (idx / (values.length - 1)) * plotW : plotW / 2);
      const label = xLabels && idx < xLabels.length ? xLabels[idx] : idx + 1;
      ctx.fillText(formatX(label), x, padT + plotH + 2);
    }
    // titles
    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.textAlign = "left"; ctx.textBaseline = "top";
    if (yLabel) ctx.fillText(yLabel, 4, 2);
    if (xLabel) {
      ctx.textAlign = "right";
      ctx.fillText(xLabel, w - 2, padT + plotH + 2);
    }
  }

  // Synced hover cursor + value tooltip
  if (hoverFrac != null && values.length > 0) {
    const f = Math.max(0, Math.min(1, hoverFrac));
    const idx = Math.round(f * (values.length - 1));
    const px = padL + (values.length > 1 ? (idx / (values.length - 1)) * plotW : plotW / 2);
    const v = values[idx];
    const py = padT + plotH - ((v - mn) / range) * plotH;

    // dashed vertical line
    ctx.save();
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = "rgba(255,255,255,0.45)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(px, padT); ctx.lineTo(px, padT + plotH);
    ctx.stroke();
    ctx.restore();

    // marker dot
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(px, py, 2.5, 0, Math.PI * 2);
    ctx.fill();

    // value tag at top
    const yLab = formatY(v);
    const xLab = xLabels && idx < xLabels.length ? formatX(xLabels[idx]) : String(idx + 1);
    const tag = `${yLab}  @${xLab}`;
    ctx.font = "10px ui-monospace, Menlo, monospace";
    const tw = ctx.measureText(tag).width + 8;
    const th = 14;
    let tx = px - tw / 2;
    if (tx < padL) tx = padL;
    if (tx + tw > padL + plotW) tx = padL + plotW - tw;
    const ty = padT;
    ctx.fillStyle = "rgba(0,0,0,0.78)";
    ctx.fillRect(tx, ty, tw, th);
    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.lineWidth = 1;
    ctx.strokeRect(tx + 0.5, ty + 0.5, tw - 1, th - 1);
    ctx.fillStyle = "rgba(255,255,255,0.92)";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(tag, tx + tw / 2, ty + th / 2);
  }
}

$effect(() => { values; xLabels; xLabel; yLabel; hoverFrac; draw(); });

onMount(() => {
  const ro = new ResizeObserver(draw);
  if (canvas) ro.observe(canvas);
  return () => ro.disconnect();
});

function pointerToFrac(e) {
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect();
  const w = rect.width, h = rect.height;
  const { padL, plotW } = layout(w, h);
  const x = e.clientX - rect.left;
  if (x < padL || x > padL + plotW) return null;
  return (x - padL) / plotW;
}
function handleMove(e) {
  const f = pointerToFrac(e);
  if (f == null) { onLeave?.(); return; }
  onHover?.(f);
}
function handleLeave() { onLeave?.(); }
</script>

<div bind:this={wrap}
     style={height != null ? `height: ${height}px` : "height: 100%"}
     onmousemove={handleMove}
     onmouseleave={handleLeave}
     role="presentation">
  <canvas bind:this={canvas}></canvas>
</div>

<style>
div { width: 100%; }
canvas { display: block; width: 100%; height: 100%; }
</style>
