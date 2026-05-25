import { toasts } from "./toast.svelte.js";

// Shared reactive session state.

function maskToRanges(mask) {
  const out = [];
  let start = -1;
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] && start < 0) start = i;
    else if (!mask[i] && start >= 0) {
      out.push([start, i]);
      start = -1;
    }
  }
  if (start >= 0) out.push([start, mask.length]);
  return out;
}

class Session {
  // track
  trackSeconds = $state(0);
  sampleRate = $state(44100);
  downsampleRatio = $state(4096);

  // backend session
  version = $state(0); // bumped by backend on every change; bust caches
  hasAudio = $state(false);
  bpm = $state(null);
  bpmSource = $state(""); // "tag" or "detect"

  // mask is the single source of truth for what's painted
  mask = $state(new Uint8Array(0));
  ghostMask = $state(new Uint8Array(0)); // last-inpainted regions, for visual recall

  // zoom window over full track, normalized 0..1
  zoomStart = $state(0.0);
  zoomEnd = $state(1.0);

  // playhead, normalized 0..1 of full track
  playhead = $state(0.0);
  playing = $state(false);
  looping = $state(false);
  volume = $state(0.7); // 0..1
  visMode = $state("both");
  advancedMode = $state(false);
  activeTab = $state("inpainter"); // "inpainter" | "trainer" | "settings"

  // prompt + settings
  prompt = $state("");
  negativePrompt = $state("");
  model = $state("medium");
  steps = $state(8);
  cfg = $state(1.0);
  noise = $state(0.65);
  seed = $state(-1);
  lastSeed = $state(null);
  duration = $state(190); // text-to-audio length (sec)
  samplerType = $state("");
  apgScale = $state(1.0);
  backend = $state("");
  switchingModel = $state(false);

  // advanced
  scalePhi = $state(0.0);
  cfgIntervalStart = $state(0.0);
  cfgIntervalEnd = $state(1.0);
  cfgNormThreshold = $state(0.0);
  exitLayerIx = $state(0);
  durationPaddingSec = $state(6.0);
  distShiftType = $state("default");
  distShiftAnchorLogsnr = $state(-6.2);
  distShiftRate = $state(0.0);
  distShiftLogsnrEnd = $state(2.0);
  distShiftAlphaMin = $state(1.0);
  distShiftAlphaMax = $state(1.0);
  distShiftBaseShift = $state(0.5);
  distShiftMaxShift = $state(1.15);

  // speedups
  kvCache = $state(false);
  tomeRatio = $state(0.0);

  // decode quality
  decodeFp32 = $state(true);
  decodeOverlap = $state(32);

  loras = $state([]);

  // memory tokens
  memtokAvailable = $state(false);
  memtokStrength = $state(1.0);
  memtokPresets = $state([]);


  // variant history
  variants = $state([]); // array of { version, prompt, seed }
  variantIndex = $state(-1); // -1 = no variants yet

  get variantLabel() {
    if (this.variants.length === 0) return "";
    return `${this.variantIndex + 1} / ${this.variants.length}`;
  }

  pushVariant() {
    const snap = {
      version: this.version,
      prompt: this.prompt,
      seed: this.seed,
    };
    // if we're not at the end, truncate forward history
    this.variants = [...this.variants.slice(0, this.variantIndex + 1), snap];
    this.variantIndex = this.variants.length - 1;
  }

  canUndo = $state(false);
  canRedo = $state(false);

  precision = $state("fp32");
  generating = $state(false);
  scrubbingNoise = $state(false);
  modelLoaded = $state(false);
  stats = $state({ cpu: 0, vram: 0, ram: 0 });

  get latentCount() {
    return this.mask.length;
  }
  get paintedRanges() {
    return maskToRanges(this.mask);
  }
  get ghostRanges() {
    return maskToRanges(this.ghostMask);
  }
  get hasMask() {
    for (let i = 0; i < this.mask.length; i++) if (this.mask[i]) return true;
    return false;
  }

  setTrackInfo({ count, duration }) {
    this.trackSeconds = duration;
    // resize mask preserving existing values where possible
    const next = new Uint8Array(count);
    const old = this.mask;
    const lim = Math.min(old.length, count);
    for (let i = 0; i < lim; i++) next[i] = old[i];
    this.mask = next;
  }

  // unified undo stack: entries are { type: "mask" | "audio" }
  // mask entries store the mask snapshot; audio entries are handled by the backend
  _undoStack = [];
  _redoStack = [];

  _pushMaskUndo() {
    this._undoStack.push({ type: "mask", mask: new Uint8Array(this.mask) });
    if (this._undoStack.length > 50) this._undoStack.shift();
    this._redoStack = [];
  }
  pushAudioMarker() {
    this._undoStack.push({ type: "audio" });
    if (this._undoStack.length > 50) this._undoStack.shift();
    this._redoStack = [];
  }

  // returns the type of the entry undone ("mask" | "audio" | null)
  undo() {
    if (this._undoStack.length === 0) return null;
    const entry = this._undoStack.pop();
    if (entry.type === "mask") {
      this._redoStack.push({ type: "mask", mask: new Uint8Array(this.mask) });
      this.mask = entry.mask;
      return "mask";
    }
    this._redoStack.push({ type: "audio" });
    return "audio";
  }
  redo() {
    if (this._redoStack.length === 0) return null;
    const entry = this._redoStack.pop();
    if (entry.type === "mask") {
      this._undoStack.push({ type: "mask", mask: new Uint8Array(this.mask) });
      this.mask = entry.mask;
      return "mask";
    }
    this._undoStack.push({ type: "audio" });
    return "audio";
  }

  get canUndoAnything() {
    return this._undoStack.length > 0 || this.canUndo;
  }
  get canRedoAnything() {
    return this._redoStack.length > 0 || this.canRedo;
  }

  paint(startLatent, endLatent, mode) {
    this._pushMaskUndo();
    if (endLatent < startLatent)
      [startLatent, endLatent] = [endLatent, startLatent];
    startLatent = Math.max(0, Math.floor(startLatent));
    endLatent = Math.min(this.mask.length, Math.ceil(endLatent));
    if (endLatent <= startLatent) return;
    const m = new Uint8Array(this.mask);
    const v = mode === "regen" ? 1 : 0;
    for (let i = startLatent; i < endLatent; i++) m[i] = v;
    this.mask = m;
  }

  clearMask() {
    this.mask = new Uint8Array(this.mask.length);
    this.ghostMask = new Uint8Array(this.ghostMask.length);
  }

  // Snap a normalized [0,1] playhead to the START of a latent (its left edge).
  // Track is divided into N slots [i/N, (i+1)/N); the playhead lands on i/N.
  // Used by user-input paths (clicks, drags, arrow keys). The audio-tick
  // effect in App.svelte bypasses this and updates playhead continuously
  // during playback.
  snapPlayhead(norm) {
    const N = this.latentCount;
    if (!N || N <= 1) { this.playhead = Math.max(0, Math.min(1, norm)); return; }
    const idx = Math.max(0, Math.min(N - 1, Math.floor(norm * N)));
    this.playhead = idx / N;
  }

  movePlayheadLatents(delta) {
    const N = this.latentCount;
    if (!N || N <= 1) return;
    const cur = Math.max(0, Math.min(N - 1, Math.floor(this.playhead * N)));
    const next = Math.max(0, Math.min(N - 1, cur + delta));
    this.playhead = next / N;
  }
}

export const session = new Session();

// ---------- backend api ----------

export async function apiState() {
  const r = await fetch("/api/state");
  const j = await r.json();
  session.hasAudio = j.has_audio;
  session.version = j.version;
  if (j.backend) session.backend = j.backend;
  if (j.model) session.model = j.model;
  if (j.bpm != null) session.bpm = j.bpm;
  return j;
}

export async function apiSwitchModel(name) {
  session.switchingModel = true;
  session.modelLoaded = false;
  try {
    const r = await fetch("/api/model", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: name }),
    });
    if (!r.ok) throw new Error("model switch failed: " + r.status);
    const j = await r.json();
    session.model = name;
    session.modelLoaded = true;
    toasts.success(`Model ${name} loaded`);
    return j;
  } catch (e) {
    toasts.error("Model switch failed: " + e.message);
  } finally {
    session.switchingModel = false;
  }
}

export async function apiUpload(file) {
  try {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch("/api/upload", { method: "POST", body: fd });
    if (!r.ok) throw new Error("upload failed: " + r.status);
    const j = await r.json();
    session.hasAudio = true;
    session.version = j.version;
    session.setTrackInfo(j);
    session.duration = Math.round(j.duration);
    session.zoomStart = 0;
    session.zoomEnd = 1;
    session.playhead = 0;
    if (j.bpm != null) session.bpm = j.bpm;
    if (j.bpm_source) session.bpmSource = j.bpm_source;
    session.pushAudioMarker();
    session.canUndo = true;
    session.canRedo = false;
    return j;
  } catch (e) {
    toasts.error("Upload failed: " + e.message);
  }
}

export async function apiRedetectBpm() {
  try {
    const r = await fetch("/api/detect_bpm", { method: "POST" });
    if (!r.ok) throw new Error("BPM detect failed: " + r.status);
    const j = await r.json();
    session.bpm = j.bpm;
    session.bpmSource = j.bpm_source;
    return j;
  } catch (e) {
    toasts.error("BPM detection failed: " + e.message);
  }
}

export async function apiClear() {
  const r = await fetch("/api/clear", { method: "POST" });
  const j = await r.json();
  session.hasAudio = false;
  session.version = j.version;
  session.mask = new Uint8Array(0);
  session.ghostMask = new Uint8Array(0);
  session.trackSeconds = 0;
  session.zoomStart = 0;
  session.zoomEnd = 1;
  session.playhead = 0;
  return j;
}

export async function apiTempo(factor, targetBpm = null) {
  try {
    const payload = { factor };
    if (targetBpm) payload.target_bpm = targetBpm;
    const r = await fetch("/api/tempo", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error("tempo change failed: " + r.status);
    const j = await r.json();
    session.version = j.version;
    session.setTrackInfo(j);
    session.mask = new Uint8Array(session.mask.length > 0 ? j.count : 0);
    if (j.bpm != null) session.bpm = j.bpm;
    session.pushAudioMarker();
    session.canUndo = true;
    session.canRedo = false;
    toasts.success(`Tempo ×${factor.toFixed(2)}`);
    return j;
  } catch (e) {
    toasts.error("Tempo failed: " + e.message);
  }
}

let _genAbort = null;

export function cancelGenerate() {
  if (_genAbort) _genAbort.abort();
  _genAbort = null;
  session.generating = false;
}

export async function apiGenerate() {
  cancelGenerate();
  session.generating = true;
  _genAbort = new AbortController();
  try {
    const body = {
      prompt: session.prompt,
      ...(session.negativePrompt
        ? { negative_prompt: session.negativePrompt }
        : {}),
      mask: Array.from(session.mask),
      loras: session.loras
        .filter((l) => l.enabled)
        .map((l) => ({ name: l.name, strength: l.strength })),
      settings: {
        steps: session.steps,
        cfg: session.cfg,
        seed: session.seed,
        noise: session.noise,
        duration: session.trackSeconds || session.duration,
        ...(session.samplerType ? { sampler_type: session.samplerType } : {}),
        apg_scale: session.apgScale,
        scale_phi: session.scalePhi,
        cfg_interval: [session.cfgIntervalStart, session.cfgIntervalEnd],
        cfg_norm_threshold: session.cfgNormThreshold,
        exit_layer_ix: session.exitLayerIx || null,
        duration_padding_sec: session.durationPaddingSec,
        dist_shift_type: session.distShiftType,
        ...(session.distShiftType === "logsnr"
          ? {
              dist_shift_anchor_logsnr: session.distShiftAnchorLogsnr,
              dist_shift_rate: session.distShiftRate,
              dist_shift_logsnr_end: session.distShiftLogsnrEnd,
            }
          : {}),
        ...(session.distShiftType === "flux"
          ? {
              dist_shift_alpha_min: session.distShiftAlphaMin,
              dist_shift_alpha_max: session.distShiftAlphaMax,
            }
          : {}),
        ...(session.distShiftType === "full"
          ? {
              dist_shift_base_shift: session.distShiftBaseShift,
              dist_shift_max_shift: session.distShiftMaxShift,
            }
          : {}),
        kvCache: session.kvCache,
        tomeRatio: session.tomeRatio,
      },
    };
    const r = await fetch("/api/generate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: _genAbort.signal,
    });
    if (!r.ok) throw new Error("generate failed: " + r.status);
    const j = await r.json();
    session.hasAudio = true;
    session.version = j.version;
    session.setTrackInfo(j);
    if (j.bpm != null) session.bpm = j.bpm;
    if (j.seed != null) session.lastSeed = j.seed;
    session.pushVariant();
    // remember the inpainted regions as ghost (visual recall), then clear the live mask
    if (body.mask.some((v) => v)) {
      session.ghostMask = new Uint8Array(body.mask);
    }
    session.mask = new Uint8Array(session.mask.length);
    session.pushAudioMarker();
    session.canUndo = true;
    session.canRedo = false;
    toasts.success("Generation complete");
    return j;
  } catch (e) {
    if (e.name === "AbortError") return null;
    toasts.error("Generate failed: " + e.message);
  } finally {
    _genAbort = null;
    session.generating = false;
  }
}

export async function apiUndo() {
  try {
    const r = await fetch("/api/undo", { method: "POST" });
    if (!r.ok) throw new Error("undo failed: " + r.status);
    const j = await r.json();
    session.hasAudio = true;
    session.version = j.version;
    session.setTrackInfo(j);
    if (j.bpm != null) session.bpm = j.bpm;
    session.canUndo = j.can_undo;
    session.canRedo = j.can_redo;
    session.mask = new Uint8Array(session.mask.length);
    session.ghostMask = new Uint8Array(session.ghostMask.length);
    return j;
  } catch (e) {
    toasts.error("Undo failed: " + e.message);
  }
}

export async function apiRedo() {
  try {
    const r = await fetch("/api/redo", { method: "POST" });
    if (!r.ok) throw new Error("redo failed: " + r.status);
    const j = await r.json();
    session.hasAudio = true;
    session.version = j.version;
    session.setTrackInfo(j);
    if (j.bpm != null) session.bpm = j.bpm;
    session.canUndo = j.can_undo;
    session.canRedo = j.can_redo;
    session.mask = new Uint8Array(session.mask.length);
    session.ghostMask = new Uint8Array(session.ghostMask.length);
    return j;
  } catch (e) {
    toasts.error("Redo failed: " + e.message);
  }
}

// -------- memory tokens --------

export async function apiMemtokInfo() {
  try {
    const r = await fetch("/api/memory_tokens");
    const j = await r.json();
    session.memtokAvailable = j.available;
    session.memtokStrength = j.strength ?? 1.0;
    session.memtokPresets = j.presets ?? [];
    return j;
  } catch (e) {
    console.warn("memtok info failed:", e);
  }
}

export async function apiMemtokSet(strength) {
  try {
    const r = await fetch("/api/memory_tokens", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ strength, action: "set" }),
    });
    const j = await r.json();
    session.memtokStrength = j.strength;
  } catch (e) {
    toasts.error("Memory token update failed: " + e.message);
  }
}

export async function apiMemtokAction(action, name = null) {
  try {
    const r = await fetch("/api/memory_tokens", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ action, name, preset: name }),
    });
    const j = await r.json();
    if (j.strength != null) session.memtokStrength = j.strength;
    if (action === "save") toasts.success(`Saved memory tokens: ${name}`);
    if (action === "load") toasts.success(`Loaded memory tokens: ${name}`);
    if (action === "reset") toasts.success("Memory tokens reset");
    await apiMemtokInfo();
    return j;
  } catch (e) {
    toasts.error(`Memory token ${action} failed: ` + e.message);
  }
}

// -------- decode settings --------

export async function apiDecodeSettings() {
  try {
    const r = await fetch("/api/decode_settings");
    const j = await r.json();
    session.decodeFp32 = j.decode_fp32;
    session.decodeOverlap = j.decode_overlap;
    return j;
  } catch (e) {
    console.warn("decode settings fetch failed:", e);
  }
}

export async function apiSetDecodeSettings(opts) {
  try {
    const r = await fetch("/api/decode_settings", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(opts),
    });
    const j = await r.json();
    session.decodeFp32 = j.decode_fp32;
    session.decodeOverlap = j.decode_overlap;
  } catch (e) {
    console.warn("decode settings update failed:", e);
  }
}

// -------- LoRA data folder management --------

export async function apiLoraDataList(name, folder = null) {
  try {
    const r = await fetch("/api/lora_data/list", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name, folder }),
    });
    if (!r.ok) return null;
    return await r.json();
  } catch (e) {
    console.warn("lora data list failed:", e);
    return null;
  }
}

export async function apiLoraDataUpload(name, folder, fileList) {
  const fd = new FormData();
  fd.append("name", name || "");
  fd.append("folder", folder || "");
  for (const f of fileList) fd.append("files", f);
  try {
    const r = await fetch("/api/lora_data/upload", { method: "POST", body: fd });
    if (!r.ok) {
      const txt = await r.text();
      toasts.error("Upload failed: " + txt.slice(0, 200));
      return null;
    }
    return await r.json();
  } catch (e) {
    toasts.error("Upload failed: " + e.message);
    return null;
  }
}

export async function apiLoraDataGetCaption(name, folder, file) {
  try {
    const q = new URLSearchParams({ name: name || "", folder: folder || "", file });
    const r = await fetch(`/api/lora_data/caption?${q}`);
    if (!r.ok) return null;
    return await r.json();
  } catch (e) {
    console.warn("caption get failed:", e);
    return null;
  }
}

export async function apiLoraDataSetCaption(name, folder, file, text) {
  try {
    const r = await fetch("/api/lora_data/caption", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name, folder, file, text }),
    });
    if (!r.ok) {
      const txt = await r.text();
      toasts.error("Caption save failed: " + txt.slice(0, 200));
      return null;
    }
    return await r.json();
  } catch (e) {
    toasts.error("Caption save failed: " + e.message);
    return null;
  }
}

export async function apiLoraDataClear(name, folder = null) {
  try {
    const r = await fetch("/api/lora_data/clear", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name, folder }),
    });
    if (!r.ok) {
      const txt = await r.text();
      toasts.error("Clear failed: " + txt.slice(0, 200));
      return null;
    }
    return await r.json();
  } catch (e) {
    toasts.error("Clear failed: " + e.message);
    return null;
  }
}

export async function apiLoraDataClearCaptions(name, folder = null) {
  return _postJson("/api/lora_data/clear_captions", { name, folder });
}

export async function apiClearEncoded(name) {
  if (!name) return null;
  return _postJson("/api/pre_encode/clear", { name });
}

export async function apiCancelPreEncode() {
  return _postJson("/api/pre_encode/cancel", {});
}

export async function apiCancelLoraTrain() {
  return _postJson("/api/train_lora/cancel", {});
}

export async function apiGetTrainSettings(name) {
  if (!name) return {};
  return (await _postJson("/api/lora_data/train_settings/get", { name })) || {};
}
export async function apiSetTrainSettings(name, settings) {
  if (!name) return null;
  return _postJson("/api/lora_data/train_settings/set", { name, settings });
}

// ── auto-captioner ──────────────────────────────────────────────────────────
async function _postJson(url, body) {
  try {
    const r = await fetch(url, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) { const t = await r.text(); toasts.error(t.slice(0, 200)); return null; }
    return await r.json();
  } catch (e) { toasts.error(String(e)); return null; }
}

export async function apiAutocaptionEstimate(name, folder, opts = {}) {
  return _postJson("/api/lora_data/autocaption/estimate", {
    name, folder, only_uncaptioned: opts.onlyUncaptioned ?? true,
    model: opts.model || "pro",
  });
}
export async function apiAutocaptionStart(name, folder, opts) {
  return _postJson("/api/lora_data/autocaption/start", {
    name, folder, only_uncaptioned: opts.onlyUncaptioned ?? true,
    examples: opts.examples, parallel: opts.parallel, model: opts.model || "pro",
  });
}
export async function apiAutocaptionStatus() {
  try {
    const r = await fetch("/api/lora_data/autocaption/status");
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}
export async function apiAutocaptionCancel() {
  return _postJson("/api/lora_data/autocaption/cancel", {});
}
export async function apiCaptionerExamplesGet(name, folder) {
  return _postJson("/api/lora_data/captioner_examples/get", { name, folder });
}
export async function apiCaptionerExamplesSet(name, folder, examples, model, parallel) {
  return _postJson("/api/lora_data/captioner_examples/set", { name, folder, examples, model, parallel });
}


export async function apiLoraDataRename(oldName, newName) {
  try {
    const r = await fetch("/api/lora_data/rename", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ old_name: oldName, new_name: newName }),
    });
    if (!r.ok) {
      const txt = await r.text();
      toasts.error("Rename failed: " + txt.slice(0, 200));
      return null;
    }
    return await r.json();
  } catch (e) {
    toasts.error("Rename failed: " + e.message);
    return null;
  }
}

export async function apiLoraDataDelete(name, folder, file) {
  try {
    const r = await fetch("/api/lora_data/delete", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name, folder, file }),
    });
    if (!r.ok) return null;
    return await r.json();
  } catch (e) {
    console.warn("lora data delete failed:", e);
    return null;
  }
}


export async function apiTrainLora(opts) {
  try {
    const r = await fetch("/api/train_lora", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(opts),
    });
    if (!r.ok) {
      const err = await r.text();
      throw new Error(err);
    }
    const j = await r.json();
    toasts.success(`LoRA training started: ${opts.name}`);
    return j;
  } catch (e) {
    toasts.error("LoRA train failed: " + e.message);
  }
}

export async function apiPreEncode(folder, name, caption) {
  try {
    const r = await fetch("/api/pre_encode", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ folder, name, caption }),
    });
    if (!r.ok) {
      const err = await r.text();
      throw new Error(err);
    }
    const j = await r.json();
    toasts.success(`Pre-encoding started: ${name}`);
    return j;
  } catch (e) {
    toasts.error("Pre-encode failed: " + e.message);
  }
}

export async function apiPreEncodeStatus() {
  try {
    const r = await fetch("/api/pre_encode/status");
    return await r.json();
  } catch (e) {
    return { status: "error", error: e.message };
  }
}

export async function apiCheckEncoded(name) {
  try {
    const r = await fetch(
      `/api/lora_training/${encodeURIComponent(name)}/has_encoded`,
    );
    return await r.json();
  } catch (e) {
    return { has_encoded: false, latents: 0 };
  }
}

export async function apiLoraTrainStatus() {
  try {
    const r = await fetch("/api/train_lora/status");
    return await r.json();
  } catch (e) {
    return { status: "error", error: e.message };
  }
}

export async function apiGetSettings() {
  try {
    const r = await fetch("/api/settings");
    return await r.json();
  } catch (e) {
    return null;
  }
}

export async function apiSaveSettings(settings) {
  try {
    const r = await fetch("/api/settings", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(settings),
    });
    if (!r.ok) throw new Error(await r.text());
    const j = await r.json();
    toasts.success("Settings saved");
    return j;
  } catch (e) {
    toasts.error("Settings save failed: " + e.message);
    return null;
  }
}
