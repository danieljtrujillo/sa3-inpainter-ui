<script>
import { onMount, untrack } from "svelte";
import SparkLine from "./SparkLine.svelte";
import { session, apiTrainLora, apiLoraTrainStatus, apiPreEncode, apiPreEncodeStatus, apiCheckEncoded,
         apiLoraDataList, apiLoraDataUpload, apiLoraDataDelete, apiLoraDataRename, apiLoraDataClear,
         apiLoraDataClearCaptions, apiLoraDataGetCaption, apiLoraDataSetCaption,
         apiGetTrainSettings, apiSetTrainSettings, apiClearEncoded, apiCancelPreEncode, apiCancelLoraTrain,
         apiAutocaptionEstimate, apiAutocaptionStart, apiAutocaptionStatus, apiAutocaptionCancel,
         apiGetSettings } from "./session.svelte.js";
import { toasts } from "./toast.svelte.js";

// Single shared scratch folder for uncommitted uploads. Renaming away from
// this name moves the files to a real LoRA name; closing/reopening keeps the
// scratch around so you can pick up where you left off.
const SCRATCH_NAME = "_scratch";
const LORA_NAME_STORAGE_KEY = "sa3.trainer.loraName";
function defaultLoraName() {
  // Restore the last-edited lora across refreshes so the user doesn't have to
  // retype the name every time. Empty/missing → scratch placeholder.
  try {
    const saved = localStorage.getItem(LORA_NAME_STORAGE_KEY);
    if (saved && saved.trim()) return saved.trim();
  } catch {}
  return SCRATCH_NAME;
}

// form state
let loraFolder = $state("");   // override; blank = default lora_train_dir/<name>/data
let loraName = $state(defaultLoraName());
let lastCommittedName = $state(loraName);
$effect(() => {
  try {
    if (loraName && loraName !== SCRATCH_NAME) localStorage.setItem(LORA_NAME_STORAGE_KEY, loraName);
    else localStorage.removeItem(LORA_NAME_STORAGE_KEY);
  } catch {}
});
// Display value for the Name input — empty when loraName is the scratch
// placeholder, so the input looks blank with a placeholder instead of "_scratch".
let nameInputValue = $state("");
$effect(() => { nameInputValue = (loraName === SCRATCH_NAME ? "" : loraName); });
let loraCaption = $state("");
let loraTrigger = $state("");
let loraRank = $state(32);
let loraSteps = $state(30000);
let loraLr   = $state(1e-4);
let loraAdapter = $state("dora-rows");
let loraCompile = $state(true);
let loraBatch = $state(0);

// Per-lora training settings persistence (LORA_TRAIN_DIR/<name>/train_settings.json)
let loadingTrainSettings = $state(false);
let trainSettingsSaveTimer = 0;
let trainSettingsLoadedFor = "";
async function loadTrainSettings(name) {
  if (!name || name === SCRATCH_NAME) return;
  if (trainSettingsLoadedFor === name) return;
  loadingTrainSettings = true;
  const s = await apiGetTrainSettings(name);
  if (s && typeof s === "object") {
    if (Number.isFinite(s.rank))    loraRank    = s.rank;
    if (Number.isFinite(s.steps))   loraSteps   = s.steps;
    if (Number.isFinite(s.batch))   loraBatch   = s.batch;
    if (Number.isFinite(s.lr))      loraLr      = s.lr;
    if (typeof s.compile  === "boolean") loraCompile = s.compile;
    if (typeof s.trigger  === "string")  loraTrigger = s.trigger;
    // Restore cached profile only if it was measured under matching shape
    // settings — different rank/batch/adapter changes the step time.
    if (s.profile && Number.isFinite(s.profile.msPerStep)
        && s.profile.rank === s.rank && s.profile.batch === s.batch) {
      profileResult = s.profile;
    } else {
      profileResult = null;
    }
  }
  trainSettingsLoadedFor = name;
  // allow the apply-assignments above to flush before re-enabling autosave
  await Promise.resolve();
  loadingTrainSettings = false;
}
function scheduleTrainSettingsSave() {
  if (loadingTrainSettings) return;
  if (!loraName || loraName === SCRATCH_NAME) return;
  if (trainSettingsSaveTimer) clearTimeout(trainSettingsSaveTimer);
  const name = loraName;
  trainSettingsSaveTimer = setTimeout(() => {
    apiSetTrainSettings(name, {
      rank: loraRank, steps: loraSteps, batch: loraBatch, lr: loraLr,
      compile: loraCompile, trigger: loraTrigger,
      profile: profileResult,
    });
  }, 350);
}
$effect(() => {
  // Read every field so Svelte re-runs the effect on any change.
  loraRank; loraSteps; loraBatch; loraLr; loraCompile; loraTrigger;
  scheduleTrainSettingsSave();
});
$effect(() => {
  loraName;
  trainSettingsLoadedFor = "";   // force reload when lora changes
  loadTrainSettings(loraName);
});
let loraStatus = $state("idle");
let loraStep = $state(0);
let loraLoss = $state(0);
let loraGradNorm = $state(null);    // populated when training script emits grad_norm
let loraStage = $state("starting up…");   // human-readable phase, derived from progress.status
// Maps the training script's machine status codes to user-visible labels.
const STAGE_LABELS = {
  init:             "starting up…",
  dataset_loaded:   "loading data…",
  latent_info:      "loading data…",
  loading_dit:      "loading model…",
  dit_loaded:       "loading model…",
  loading_t5gemma:  "loading text encoder…",
  t5gemma_loaded:   "loading text encoder…",
  text_encoded:     "encoding prompt…",
  conditioner_lora: "injecting adapters…",
  lora_injected:    "injecting adapters…",
  params:           "preparing optimizer…",
  grad_checkpoint_enabled: "preparing optimizer…",
  training:         "training…",
  step:             "training…",
  checkpoint:       "saving checkpoint…",
  done:             "finishing up…",
  saved:            "finishing up…",
  warning:          "training…",
  error:            "error",
};
// Per-metric histories (sparkline buffers). Capped to keep them light.
const METRIC_CAP = 600;
let lossHistory = $state([]);
let gradHistory = $state([]);
let cpuHistory  = $state([]);
let ramHistory  = $state([]);
function pushHist(arr, v) {
  if (v == null || !isFinite(v)) return arr;
  arr = [...arr, v];
  if (arr.length > METRIC_CAP) arr = arr.slice(arr.length - METRIC_CAP);
  return arr;
}
function resetHistories() {
  lossHistory = []; gradHistory = []; cpuHistory = []; ramHistory = [];
}
// live elapsed timer for training (tqdm-style)
let loraStartTs = $state(0);
let loraNowTs   = $state(0);
let loraTickHandle = 0;
function startLoraTicker() {
  if (loraTickHandle) return;
  loraTickHandle = setInterval(() => { loraNowTs = Date.now(); }, 500);
}
function stopLoraTicker() {
  if (loraTickHandle) { clearInterval(loraTickHandle); loraTickHandle = 0; }
}
let loraElapsedSec = $derived.by(() => {
  if (!loraStartTs) return 0;
  const end = (loraStatus === "running") ? loraNowTs : (loraNowTs || Date.now());
  return Math.max(0, (end - loraStartTs) / 1000);
});
let loraEtaSec = $derived.by(() => {
  if (loraStatus !== "running" || loraStep === 0 || loraSteps === 0) return null;
  const stepsDone = loraStep * loraTrainBatch;
  const sPerStep  = loraElapsedSec / Math.max(1, stepsDone);
  const remaining = loraSteps - stepsDone;
  return remaining * sPerStep;
});
let loraTrainBatch = $state(1);
let loraPollTimer = null;

// dataset folder state
let dropActive = $state(false);
let dataFiles = $state([]);        // [{name, stem, ext, captioned}]
let dataFolderResolved = $state(""); // absolute path the backend is reading from
let dataTotal = $state(0);
let dataCaptioned = $state(0);
let uploading = $state(false);
let fileInput = $state(null);
let selectedFile = $state(null);   // name of the currently-selected card
let editingFile = $state(null);
let captionText = $state("");
let captionLoading = $state(false);
let captionAnchorRect = $state(null);   // bounding rect of the card we anchored to
let captionTextarea = $state(null);
let captionOriginal = "";               // for change-detection on save
let gridEl = $state(null);
// ── auto-captioner state ─────────────────────────────────────────────────
// Captioner config lives in global Settings now. We snapshot it here so the
// summary line can render model name + cost estimate; refreshed on tab focus.
let captionerExamplesText = $state("");
let captionerModel    = $state("pro");
let captionerParallel = $state(32);
const CAPTIONER_MODEL_LABEL = { pro: "Gemini 3.1 Pro", flash: "Gemini 3.5 Flash" };
let captionerEstimate = $state(null);         // { n_files, per_call, total, n_samples }
let captionerStatus   = $state(null);         // backend status snapshot
let captionerPollTimer = 0;
async function loadCaptionerCfg() {
  const s = await apiGetSettings();
  if (s) {
    captionerModel    = s.captioner_model    || "pro";
    captionerParallel = s.captioner_parallel || 32;
    captionerExamplesText = s.captioner_examples || "";
    if (s.lora_adapter) loraAdapter = s.lora_adapter;
  }
  await refreshCaptionerEstimate();
}
// Refresh whenever the dataset context changes (file count → estimate).
$effect(() => {
  loraName; loraFolder; dataTotal;
  if (loraName || loraFolder) loadCaptionerCfg();
});
async function refreshCaptionerEstimate() {
  const r = await apiAutocaptionEstimate(loraName, loraFolder || null,
    { onlyUncaptioned: true, model: captionerModel });
  captionerEstimate = r;
}
async function startCaptioner() {
  const examples = (captionerExamplesText || "").split("\n").map(s => s.trim()).filter(Boolean);
  if (examples.length === 0) { toasts.error("Set example captions in Settings first."); return; }
  // Optimistic UI: flip to "running 0/N" the instant the button is clicked,
  // so the user sees a reaction before the start request returns.
  const expected = captionerEstimate?.n_files ?? 0;
  captionerStatus = { running: true, done: 0, total: expected, cost_so_far: 0 };
  const r = await apiAutocaptionStart(loraName, loraFolder || null, {
    onlyUncaptioned: true, examples, parallel: captionerParallel, model: captionerModel,
  });
  if (r?.started) pollCaptionerStatus();
  else captionerStatus = null;   // backend refused — clear the optimistic state
}
async function cancelCaptioner() {
  // Instant UI reset — backend confirms async.
  captionerStatus = null;
  if (captionerPollTimer) { clearTimeout(captionerPollTimer); captionerPollTimer = 0; }
  await apiAutocaptionCancel();
}
async function clearAllCaptions() {
  // Clear the stale "36/36" status before the network round-trip so the UI
  // resets immediately.
  captionerStatus = null;
  const r = await apiLoraDataClearCaptions(loraName, loraFolder || null);
  if (r) {
    dataFiles = r.files; dataTotal = r.total; dataCaptioned = r.captioned;
    dataFolderResolved = r.folder;
    refreshCaptionerEstimate();
  }
}
let wasCaptionerRunning = false;
function pollCaptionerStatus() {
  if (captionerPollTimer) clearTimeout(captionerPollTimer);
  (async () => {
    const s = await apiAutocaptionStatus();
    const matches = s && s.lora_name && s.lora_name === loraName;
    captionerStatus = matches ? s : null;
    if (matches && (s.running || s.done > 0)) {
      const r = await apiLoraDataList(loraName, loraFolder || null);
      if (r) { dataFiles = r.files; dataTotal = r.total; dataCaptioned = r.captioned; dataFolderResolved = r.folder; }
    }
    if (matches && s.running) {
      wasCaptionerRunning = true;
      captionerPollTimer = setTimeout(pollCaptionerStatus, 800);
    } else if (wasCaptionerRunning) {
      // Just transitioned from running → done: refresh the estimate so the
      // button flips from "autocaption" to "clear captions" once everything
      // is captioned.
      wasCaptionerRunning = false;
      refreshCaptionerEstimate();
    }
  })();
}

let folderInputEl = $state(null);            // bound to the source-path input
let triggerInputEl = $state(null);           // bound to the trigger input
let folderInputWidth = $state(0);            // measured each layout

async function openCaptionEditor(name) {
  editingFile = name;
  captionLoading = true;
  captionText = "";
  captionOriginal = "";
  // capture the card's bounding rect now (before any layout shift)
  const el = gridEl?.querySelector(`[data-file="${cssEscape(name)}"]`);
  captionAnchorRect = el ? el.getBoundingClientRect() : null;
  const r = await apiLoraDataGetCaption(loraName, loraFolder || null, name);
  if (r) { captionText = r.text || ""; captionOriginal = captionText; }
  captionLoading = false;
  // focus the textarea once it mounts
  queueMicrotask(() => captionTextarea?.focus());
}

async function closeCaptionEditor(save = true) {
  if (!editingFile) return;
  const name = editingFile;
  editingFile = null;
  captionAnchorRect = null;
  if (save && captionText !== captionOriginal) {
    const r = await apiLoraDataSetCaption(loraName, loraFolder || null, name, captionText);
    if (r) {
      dataFiles = r.files;
      dataTotal = r.total;
      dataCaptioned = r.captioned;
    }
  }
  captionText = "";
  captionOriginal = "";
}

function cssEscape(s) {
  return (window.CSS && window.CSS.escape) ? window.CSS.escape(s) : s.replace(/["\\]/g, "\\$&");
}

function onCaptionKeyDown(e) {
  if (e.key === "Escape")        { e.preventDefault(); closeCaptionEditor(true); }
  else if (e.key === "Enter")    { e.preventDefault(); closeCaptionEditor(true); }
}
function onCaptionInput(e) {
  // strip newlines (paste-from-multiline → flatten to single line)
  const v = e.currentTarget.value;
  if (v.includes("\n") || v.includes("\r")) {
    captionText = v.replace(/[\r\n]+/g, " ").replace(/\s+/g, " ");
  }
}

// position the popover relative to the anchored card: prefer below the card,
// flip above if there's no room.
let captionPopStyle = $derived.by(() => {
  if (!captionAnchorRect) return "display: none";
  const r = captionAnchorRect;
  const POP_W = 320, POP_H = 200, PAD = 8;
  // Anchor the popover's TOP-LEFT near the card's bottom-right corner (i.e.,
  // where the caption-badge cursor sits). Then nudge up 25px.
  let left = r.right - 25;
  const spaceBelow = window.innerHeight - r.bottom - PAD;
  let top;
  if (spaceBelow >= POP_H) { top = r.bottom + 4 - 50; }
  else                     { top = r.top - POP_H - 4 - 50; }
  // clamp to viewport
  left = Math.max(PAD, Math.min(window.innerWidth - POP_W - PAD, left));
  top  = Math.max(PAD, Math.min(window.innerHeight - POP_H - PAD, top));
  return `left: ${left}px; top: ${top}px; width: ${POP_W}px;`;
});

function onWindowMousedown(e) {
  if (!editingFile) return;
  if (e.target.closest(".caption-pop")) return;       // click inside popover, ignore
  if (e.target.closest(".file-card")) return;          // click on a card, handled separately
  closeCaptionEditor(true);
}

// ─── pre-training checklist ────────────────────────────────────────────────
// each check: { label, state: "pass" | "warn" | "fail", detail }
let trainChecks = $derived.by(() => {
  const r = (s) => s.ramTotalGb ? (s.ramTotalGb - s.ramUsedGb) : 999;
  const freeRamGb = r(session.stats);
  const out = [];

  // has audio
  out.push(dataTotal === 0
    ? { label: "no audio",        state: "fail", detail: "Upload at least one audio file." }
    : { label: "audio",           state: "pass", detail: `${dataTotal} files uploaded.` });

  // captions
  if (dataTotal === 0) {
    out.push({ label: "no captions", state: "fail", detail: "Add audio files first." });
  } else if (dataCaptioned === 0) {
    out.push({ label: "no captions", state: "fail", detail: "Pair each audio file with a .txt of the same name." });
  } else if (dataCaptioned < dataTotal) {
    out.push({ label: `${dataTotal - dataCaptioned} no captions`, state: "warn",
               detail: `${dataTotal - dataCaptioned} files have no caption.` });
  } else {
    out.push({ label: "captions",  state: "pass", detail: "All files have paired captions." });
  }

  // lora name (scratch is OK but flag it)
  if (!loraName) {
    out.push({ label: "unnamed",  state: "fail", detail: "Set a name for the LoRA." });
  } else if (loraName === SCRATCH_NAME) {
    out.push({ label: "placeholder name", state: "warn",
               detail: "Using the placeholder/scratch folder — rename to commit this LoRA." });
  } else {
    out.push({ label: "named",    state: "pass", detail: `Will save as ${loraName}.safetensors` });
  }

  // pre-encode (optional but recommended)
  out.push(hasEncoded
    ? { label: "pre-encoded",     state: "pass", detail: `${encodedLatents} latents cached.` }
    : { label: "not pre-encoded", state: "warn",
        detail: "Optional. Pre-encoding speeds up training by skipping the AE on every step." });

  // resources — RAM proxy (disk check is TODO)
  if (freeRamGb < 4) {
    out.push({ label: "low memory", state: "fail",
               detail: `Only ${freeRamGb.toFixed(1)} GB free — training will likely OOM.` });
  } else if (freeRamGb < 8) {
    out.push({ label: "tight memory", state: "warn",
               detail: `${freeRamGb.toFixed(1)} GB free — may OOM on larger ranks/batches.` });
  } else {
    out.push({ label: "system resources ok", state: "pass",
               detail: `${freeRamGb.toFixed(1)} GB RAM available.` });
  }

  return out;
});
// ─── training quality advisory (separate from the checklist) ────────────────
// Soft warnings about dataset size, epoch count, overfit/underfit risk.
let trainAdvisory = $derived.by(() => {
  const n = dataTotal;
  const steps = loraSteps;
  const effBatch = Math.max(1, Number(loraBatch) || 1);
  const out = [];

  // Epochs (info pill — no icon, white text)
  if (n > 0) {
    const epochs = Math.round((steps * effBatch) / n);
    out.push({
      label: `${epochs} epoch${epochs === 1 ? "" : "s"}`,
      state: "info",
      detail: `${steps} steps × batch ${effBatch} / ${n} files`,
    });
  }

  // Dataset size
  if (n === 0) {
    out.push({ label: "no data", state: "fail", detail: "Upload audio files." });
  } else if (n < 100) {
    out.push({ label: "not enough tracks", state: "fail",
               detail: `Only ${n} files — fewer than 100 is risky. Collect more before training.` });
  } else if (n < 500) {
    out.push({ label: "few tracks", state: "warn",
               detail: `${n} files — usable, but more would improve generalization.` });
  }
  // (≥500 files: no badge needed; the epochs pill already gives context.)

  // Overfit — under 1k steps, only flag if the dataset is tiny (≤15 tracks),
  // since short training runs on small-but-not-tiny sets aren't really at risk.
  let overfit = "pass";
  if (n > 0 && n <= 15)                            overfit = "fail";
  else if (n > 0 && n < 100  && steps >= 1000)     overfit = "fail";
  else if (n > 0 && n < 500  && steps > 10000)     overfit = "fail";
  else if (n > 0 && n < 500  && steps > 5000)      overfit = "warn";
  if (overfit !== "pass") out.push({
    label: overfit === "fail" ? "overfit likely" : "overfit risk",
    state: overfit,
    detail: "Many steps + small dataset → likely to memorize specific tracks instead of learning the style.",
  });

  // Underfit
  let underfit = "pass";
  if (steps < 1000)      underfit = "fail";
  else if (steps < 5000) underfit = "warn";
  if (underfit !== "pass") out.push({
    label: underfit === "fail" ? "undertrain" : "low steps",
    state: underfit,
    detail: "Too few steps — model won't have time to learn the dataset.",
  });

  // Learning rate sanity
  if (loraLr <= 0) out.push({ label: "lr=0", state: "fail", detail: "Learning rate must be > 0." });
  else if (loraLr > 1e-2) out.push({ label: "lr too high", state: "fail", detail: "lr > 0.01 will blow up training." });
  else if (loraLr > 5e-4) out.push({ label: "lr high", state: "warn", detail: "Aggressive lr — risk of unstable loss." });
  else if (loraLr < 1e-5) out.push({ label: "lr very low", state: "warn", detail: "Very low lr — may not converge in given steps." });

  // Estimated training time + memory (from a Test System profile run)
  if (profileResult && profileResult.msPerStep > 0) {
    const sec = profileResult.msPerStep * loraSteps / 1000;
    const ms  = profileResult.msPerStep.toFixed(0);
    const gb  = profileResult.estPeakRamGb;
    const parts = [
      `est. ${ms} ms/step`,
      `~${fmtDuration(sec)} total`,
    ];
    if (gb != null) parts.push(`~+${gb.toFixed(1)}GB peak ram`);
    out.push({
      label: parts.join(" · "),
      state: "info", icon: "bi-clock",
      detail: `${ms} ms/step × ${loraSteps} steps`,
    });
  } else {
    out.push({ label: "no timing data available", state: "info", icon: "bi-x-circle",
               detail: "Click Test System to measure step time on your hardware." });
  }

  return out;
});

// Hard blockers, by explicit name: placeholder name, no captions, no audio,
// not enough resources. Everything else — advisory pills, pre-encode state,
// even other checklist fails — is informational and does NOT grey out the
// train button.
let canTrain = $derived(
  loraStatus !== "running"
  && dataTotal > 0
  && dataCaptioned > 0
  && loraName && loraName !== SCRATCH_NAME
  && (() => {
    const r = session.stats?.ramTotalGb ? (session.stats.ramTotalGb - session.stats.ramUsedGb) : 999;
    return r >= 4;
  })()
);

// Sample CPU% and RAM into sparkline buffers on a fixed 1s cadence while
// training is running, independent of the loss/grad poll. Stats themselves
// update on App.svelte's pollStats interval; this just samples whatever's
// current every second so the chart updates at a steady rate.
let resourceTicker = 0;
$effect(() => {
  if (loraStatus !== "running") {
    if (resourceTicker) { clearInterval(resourceTicker); resourceTicker = 0; }
    return;
  }
  if (resourceTicker) return;
  resourceTicker = setInterval(() => {
    const cpu = session.stats?.cpu;
    const ram = session.backend === "cuda"
      ? session.stats?.gpuAllocGb
      : session.stats?.ramUsedGb;
    if (cpu != null) cpuHistory = pushHist(cpuHistory, cpu);
    if (ram != null) ramHistory = pushHist(ramHistory, ram);
  }, 1000);
});

// pre-encode progress numbers — derived once for reuse in markup
let preEncodeTotalShown = $derived(preEncodeTotal || dataTotal || 0);
let preEncodeDoneShown  = $derived(hasEncoded ? preEncodeTotalShown : preEncodeBatch);
let preEncodePct        = $derived(preEncodeTotalShown > 0 ? (preEncodeDoneShown / preEncodeTotalShown) * 100 : 0);

// captioner progress
// Synced hover cursor — when any chart is hovered, all metric tiles draw a
// dashed vertical line at the same fractional x-position so values across
// charts can be compared at a glance (wandb-style).
let hoverFrac = $state(null);   // 0..1, or null when no chart hovered
function onMetricHover(frac) { hoverFrac = frac; }
function onMetricLeave()     { hoverFrac = null; }

// Step labels aligned with the loss/grad sparkline buffers. Each entry in
// lossHistory was pushed at step (loraStep - len + 1 + i), so we generate
// the matching axis labels for the SparkLine x-ticks.
let lossSteps = $derived.by(() => {
  const n = lossHistory.length;
  if (n === 0) return [];
  const start = Math.max(1, loraStep - n + 1);
  return Array.from({ length: n }, (_, i) => start + i);
});

// Captioner progress display, in priority order:
//   1. Live run in progress → status.done / status.total
//   2. Pending uncaptioned files → 0 / estimate.n_files
//   3. All files already captioned → dataCaptioned / dataTotal (full bar)
//   4. Idle, no files → 0 / 0 (hidden by caller if desired)
let capDoneShown = $derived(
  captionerStatus?.done
    ?? (captionerEstimate && captionerEstimate.n_files === 0 && dataTotal > 0 ? dataCaptioned : 0)
);
let capTotalShown = $derived(
  captionerStatus?.total
    ?? (captionerEstimate && captionerEstimate.n_files > 0 ? captionerEstimate.n_files
        : (dataTotal > 0 ? dataTotal : 0))
);
let capPctShown = $derived(capTotalShown > 0 ? (capDoneShown / capTotalShown * 100) : 0);

// ─── system profile (test-system button) ───────────────────────────────────
// Runs a few dummy training steps to measure ms/step + peak memory delta,
// then extrapolates total time/memory for the configured run.
let profileRunning = $state(false);
let profileResult = $state(null);   // { msPerStep, peakRamGb, estTotalSec, estPeakRamGb }
async function runProfile() {
  if (profileRunning) return;
  profileRunning = true;
  try {
    const r = await fetch("/api/train_lora/profile", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        name: loraName,
        adapter_type: loraAdapter,
        rank: loraRank,
        batch_size: loraBatch,
        n_steps: 5,
      }),
    });
    if (!r.ok) {
      const msg = (await r.text()).slice(0, 240) || `HTTP ${r.status}`;
      toasts.error("Profile failed: " + msg);
      return;
    }
    const j = await r.json();
    const estTotal = (j.ms_per_step ?? 0) * loraSteps / 1000;
    profileResult = {
      msPerStep: j.ms_per_step ?? 0,
      peakRamGb: j.peak_ram_gb ?? 0,
      estTotalSec: estTotal,
      estPeakRamGb: j.peak_ram_gb ?? 0,
      ts: Date.now(),
      rank: loraRank,
      batch: loraBatch,
      adapter: loraAdapter,
    };
    scheduleTrainSettingsSave();   // persist alongside the per-lora settings
  } catch (e) {
    toasts.error("Profile failed: " + e.message);
  } finally {
    profileRunning = false;
  }
}
function fmtDuration(s) {
  if (!s || !isFinite(s)) return "—";
  if (s < 60) return `${s.toFixed(0)}s`;
  const m = Math.floor(s / 60);
  const rem = Math.round(s - m * 60);
  if (m < 60) return `${m}m ${rem}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m - h*60}m`;
}

async function refreshDataList() {
  if (!loraName && !loraFolder) {
    dataFiles = []; dataTotal = 0; dataCaptioned = 0; dataFolderResolved = "";
    return;
  }
  const r = await apiLoraDataList(loraName, loraFolder || null);
  if (r) {
    dataFiles = r.files;
    dataTotal = r.total;
    dataCaptioned = r.captioned;
    dataFolderResolved = r.folder;
  }
}

$effect(() => {
  // refire when name or override path changes
  loraName; loraFolder;
  refreshDataList();
});

async function uploadFiles(fileList) {
  if (!fileList || fileList.length === 0) return;
  if (!loraName && !loraFolder) {
    toasts.error("Set a LoRA name first (right panel)");
    return;
  }
  uploading = true;
  const r = await apiLoraDataUpload(loraName, loraFolder || null, fileList);
  if (r) {
    dataFiles = r.files;
    dataTotal = r.total;
    dataCaptioned = r.captioned;
    dataFolderResolved = r.folder;
  }
  uploading = false;
}

async function clearAllUploads() {
  if (!loraName && !loraFolder) return;
  if (dataTotal === 0) return;
  if (!confirm(`Delete all ${dataTotal} files from this LoRA's folder?`)) return;
  const r = await apiLoraDataClear(loraName, loraFolder || null);
  if (r) {
    dataFiles = r.files;
    dataTotal = r.total;
    dataCaptioned = r.captioned;
    selectedFile = null;
  }
}

async function removeFile(name) {
  if (!loraName && !loraFolder) return;
  // compute selection advancement BEFORE the list updates
  let nextSelection = selectedFile;
  if (selectedFile === name) {
    const idx = dataFiles.findIndex(f => f.name === name);
    const after = dataFiles[idx + 1] ?? dataFiles[idx - 1] ?? null;
    nextSelection = after?.name ?? null;
  }
  const r = await apiLoraDataDelete(loraName, loraFolder || null, name);
  if (r) {
    dataFiles = r.files;
    dataTotal = r.total;
    dataCaptioned = r.captioned;
    selectedFile = nextSelection;
  }
}

function onDropOver(e) { e.preventDefault(); dropActive = true; }
function onDropLeave() { dropActive = false; }
function onFilesDrop(e) {
  e.preventDefault();
  dropActive = false;
  const files = e.dataTransfer?.files;
  if (files && files.length > 0) uploadFiles(Array.from(files));
}
function openFilePicker() {
  // clicking the empty drop-zone background also deselects any active card
  selectedFile = null;
  fileInput?.click();
}

function selectFile(name) {
  selectedFile = selectedFile === name ? null : name;
}

function gridColumnCount() {
  if (!gridEl) return 1;
  const cols = getComputedStyle(gridEl).gridTemplateColumns;
  return Math.max(1, cols.split(" ").filter(s => s.trim()).length);
}

function moveSelection(dx, dy) {
  if (dataFiles.length === 0) return;
  if (!selectedFile) {
    selectedFile = dataFiles[0].name;
    return;
  }
  const idx = dataFiles.findIndex(f => f.name === selectedFile);
  if (idx < 0) { selectedFile = dataFiles[0].name; return; }
  const cols = gridColumnCount();
  let target = idx + dx + dy * cols;
  target = Math.max(0, Math.min(dataFiles.length - 1, target));
  if (target !== idx) selectedFile = dataFiles[target].name;
}

function onTrainerKeyDown(e) {
  if (session.activeTab !== "trainer") return;
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
  switch (e.key) {
    case "Escape":     if (selectedFile) { e.preventDefault(); selectedFile = null; } break;
    case "ArrowLeft":  e.preventDefault(); moveSelection(-1, 0); break;
    case "ArrowRight": e.preventDefault(); moveSelection(+1, 0); break;
    case "ArrowUp":    e.preventDefault(); moveSelection(0, -1); break;
    case "ArrowDown":  e.preventDefault(); moveSelection(0, +1); break;
    case "Enter":
      if (selectedFile) { e.preventDefault(); editingFile = selectedFile; }
      break;
    case "Backspace":
    case "Delete":
      if (selectedFile) { e.preventDefault(); removeFile(selectedFile); }
      break;
  }
}

function onWindowClick(e) {
  if (!selectedFile) return;
  // file-card / file-grid / file-x all stopPropagation, so we only see clicks
  // that fell on something outside the grid — those should deselect.
  selectedFile = null;
}

onMount(() => {
  window.addEventListener("keydown", onTrainerKeyDown);
  window.addEventListener("click", onWindowClick);
  window.addEventListener("mousedown", onWindowMousedown);
  // pick up any captioning + training job already running on the backend so
  // tab switches / refreshes don't appear to lose state.
  pollCaptionerStatus();
  pollLoraStatus();
  pollPreEncode();   // resume pre-encode progress if a job is still running
  return () => {
    window.removeEventListener("keydown", onTrainerKeyDown);
    window.removeEventListener("click", onWindowClick);
    window.removeEventListener("mousedown", onWindowMousedown);
    if (captionerPollTimer) clearTimeout(captionerPollTimer);
    if (loraPollTimer) { clearTimeout(loraPollTimer); loraPollTimer = 0; }
  };
});

// Match the trigger input width to the folder input width, exactly.
// Use offsetWidth (border-box) for both — contentRect would miss the padding+border.
$effect(() => {
  if (!folderInputEl) return;
  const measure = () => { folderInputWidth = folderInputEl.offsetWidth; };
  const ro = new ResizeObserver(measure);
  ro.observe(folderInputEl);
  measure();
  return () => ro.disconnect();
});

async function onNameBlur() {
  // empty input means "back to scratch placeholder" — no rename, no committed name change
  const newName = (nameInputValue || "").trim() || SCRATCH_NAME;
  if (newName === lastCommittedName) return;
  loraName = newName;
  if (loraFolder) {
    lastCommittedName = newName;
    return;
  }
  const r = await apiLoraDataRename(lastCommittedName, newName);
  if (r) {
    lastCommittedName = newName;
    await refreshDataList();
  } else {
    loraName = lastCommittedName;
    nameInputValue = (loraName === SCRATCH_NAME ? "" : loraName);
  }
}
function onFilesPicked(e) {
  const files = e.target.files;
  if (files && files.length > 0) uploadFiles(Array.from(files));
  if (fileInput) fileInput.value = "";
}

// pre-encode
let preEncodeStatus = $state("idle");
let preEncodeBatch = $state(0);
let preEncodeTotal = $state(0);
let hasEncoded = $state(false);
let encodedLatents = $state(0);

async function browseLoraFolder() {
  try {
    const start = loraFolder || "~";
    const r = await fetch(`/api/browse_folder?start=${encodeURIComponent(start)}`);
    if (!r.ok) return;
    const j = await r.json();
    if (j.path) loraFolder = j.path;
  } catch {}
}

async function checkEncodedStatus() {
  if (!loraName) { hasEncoded = false; encodedLatents = 0; return; }
  const j = await apiCheckEncoded(loraName);
  hasEncoded = j.has_encoded;
  encodedLatents = j.latents;
}

async function startPreEncode() {
  const folder = loraFolder || dataFolderResolved;
  if (!folder || !loraName) return;
  const caption = loraTrigger || loraCaption || loraName;
  const result = await apiPreEncode(folder, loraName, caption);
  if (result) {
    preEncodeStatus = "running";
    preEncodeBatch = 0;
    pollPreEncode();
  }
}

async function clearEncodedCache() {
  if (!loraName) return;
  // Optimistic UI — flip the icon/label immediately, backend confirms.
  hasEncoded = false;
  encodedLatents = 0;
  await apiClearEncoded(loraName);
  await checkEncodedStatus();
}

async function cancelPreEncode() {
  // Optimistic: flip state before the backend confirms so the button reverts.
  preEncodeStatus = "idle";
  await apiCancelPreEncode();
  await checkEncodedStatus();
}

async function pollPreEncode() {
  const s = await apiPreEncodeStatus();
  preEncodeStatus = s.status;
  if (s.progress?.batch !== undefined) preEncodeBatch = s.progress.batch;
  if (s.progress?.total !== undefined) preEncodeTotal = s.progress.total;
  if (s.status === "running") {
    setTimeout(pollPreEncode, 3000);
  } else if (s.status === "done") {
    await checkEncodedStatus();
  }
}

async function startLoraTraining() {
  const folder = loraFolder || dataFolderResolved;
  if (!folder || !loraName) return;
  const caption = loraTrigger || loraCaption || loraName;
  // Optimistic UI — flip status, reset counters, start the elapsed-timer the
  // instant the user clicks so the button swaps to "training…" before the
  // backend ack arrives. Backend confirms on next poll.
  session.modelLoaded = false;
  loraStatus = "running";
  loraStep = 0;
  loraLoss = 0;
  loraGradNorm = null;
  loraTrainBatch = (loraBatch && loraBatch > 0) ? loraBatch : 1;
  resetHistories();
  loraStartTs = Date.now();
  loraNowTs = loraStartTs;
  startLoraTicker();
  const result = await apiTrainLora({
    folder, name: loraName, caption,
    rank: loraRank, adapter_type: loraAdapter, steps: loraSteps,
    batch_size: loraBatch, use_compile: loraCompile, lr: loraLr,
  });
  if (result) {
    loraTrainBatch = result.batch_size || loraTrainBatch;
    loraStage = "starting up…";
    pollLoraStatus();
  } else {
    // Backend rejected — revert optimistic state.
    loraStatus = "idle";
    loraStage = "starting up…";
    stopLoraTicker();
  }
}

async function cancelLoraTraining() {
  // Optimistic — flip state immediately, backend kills the subprocess async.
  loraStatus = "idle";
  loraStage = "starting up…";
  stopLoraTicker();
  await apiCancelLoraTrain();
}

async function pollLoraStatus() {
  const s = await apiLoraTrainStatus();
  loraStatus = s.status;
  // Surface the current phase as a human-readable label.
  const stageKey = s.progress?.status ?? s.result?.status;
  if (stageKey && STAGE_LABELS[stageKey]) loraStage = STAGE_LABELS[stageKey];
  // Rebuild histories from the full per-step list each poll. This is
  // idempotent (so a tab-switch remount can recover the full chart history
  // from the backend in one shot) and trivially cheap up to a few thousand
  // steps. Cap to METRIC_CAP for chart density; backend already caps at 8k.
  if (Array.isArray(s.steps_all) && s.steps_all.length) {
    const events = s.steps_all.length > METRIC_CAP
      ? s.steps_all.slice(s.steps_all.length - METRIC_CAP)
      : s.steps_all;
    const nl = []; const ng = [];
    let lastStep = null, lastLoss = null, lastGrad = null;
    for (const ev of events) {
      if (typeof ev.step === "number") lastStep = ev.step;
      if (typeof ev.loss === "number") { nl.push(ev.loss); lastLoss = ev.loss; }
      if (typeof ev.grad_norm === "number") { ng.push(ev.grad_norm); lastGrad = ev.grad_norm; }
    }
    lossHistory = nl;
    gradHistory = ng;
    if (lastStep != null) loraStep = lastStep;
    if (lastLoss != null) loraLoss = lastLoss;
    if (lastGrad != null) loraGradNorm = lastGrad;
  } else if (s.progress?.step !== undefined) {
    // No history yet — fall back to the latest progress snapshot.
    loraStep = s.progress.step;
    if (s.progress?.loss !== undefined) loraLoss = s.progress.loss;
    if (s.progress?.grad_norm !== undefined) loraGradNorm = s.progress.grad_norm;
  }
  // Restore elapsed timer from backend started_ts on first sync after remount.
  if (s.status === "running" && s.started_ts && !loraStartTs) {
    loraStartTs = s.started_ts * 1000;
    loraNowTs = Date.now();
    if (!loraTickHandle) startLoraTicker();
  }
  if (s.result?.step !== undefined) loraStep = s.result.step;
  if (s.status === "running") {
    loraPollTimer = setTimeout(pollLoraStatus, 500);
  } else if (s.status === "done" || s.status === "error") {
    if (s.status === "done") {
      loraStep = Math.ceil(loraSteps / loraTrainBatch);
    } else if (s.status === "error") {
      // Pull the meaningful last line out of the stdout tail the backend sends.
      const tail = (s.error || "").split("\n").filter(l => l.trim());
      const lastErr = tail.reverse().find(l =>
        /Error|Exception|FileNotFound|Assert|Value/.test(l)
      ) || tail[tail.length - 1] || "training failed";
      toasts.error("Training failed: " + lastErr.slice(0, 300));
      loraStage = "error";
    }
    if (s.model_reloaded) session.modelLoaded = true;
    stopLoraTicker();
  }
}

$effect(() => {
  if (loraName) checkEncodedStatus();
});
</script>

<div class="view">
  <aside class="pane dataset-pane">
    <div class="pane-body dataset-body" class:dim={editingFile}>
      <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
      <div class="drop-zone" class:active={dropActive} class:has-files={dataFiles.length > 0}
              onclick={openFilePicker}
              ondragover={onDropOver}
              ondragleave={onDropLeave}
              ondrop={onFilesDrop}
              role="button" tabindex="0">
       <div class="drop-scroll">
        {#if dataFiles.length === 0}
          <div class="drop-empty">
            <i class="bi bi-music-note-beamed"></i>
            <div class="drop-title">Drop audio files or click to browse</div>
            <div class="muted small">Paired <code>.txt</code> files become captions.</div>
          </div>
        {:else}
          <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
          <div class="file-grid" bind:this={gridEl}
               onclick={(e) => { e.stopPropagation(); selectedFile = null; }}>
            {#each dataFiles as f}
              <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
              <div class="file-card" class:selected={selectedFile === f.name}
                   data-file={f.name}
                   title={f.name}
                   onclick={(e) => { e.stopPropagation(); selectFile(f.name); }}
                   ondblclick={(e) => { e.stopPropagation(); openCaptionEditor(f.name); }}>
                <div class="file-thumb">
                  <img class="thumb-img" alt="" loading="lazy"
                       onload={(e) => e.currentTarget.classList.add("loaded")}
                       src={`/api/lora_data/thumb?name=${encodeURIComponent(loraName)}&folder=${encodeURIComponent(loraFolder || "")}&file=${encodeURIComponent(f.name)}`} />
                  <!-- <i class="bi bi-music-note-beamed thumb-fallback"></i> -->

                  <button class="file-x" onclick={(e) => { e.stopPropagation(); removeFile(f.name); }}
                          title="Remove">
                    <i class="bi bi-x"></i>
                  </button>
                  <button class="file-cap-badge" class:captioned={f.captioned}
                          onclick={(e) => { e.stopPropagation(); openCaptionEditor(f.name); }}
                          title={f.captioned ? (f.caption || "(captioned — click to edit)") : "No caption — click to add"}>
                    <i class="bi {f.captioned ? 'bi-check-lg' : 'bi-file-earmark-text'}"></i>
                  </button>
                </div>
                <div class="file-name">{f.name}</div>
              </div>
            {/each}
          </div>
        {/if}
       </div>
      </div>

      <div class="data-stats">
        <span class="data-stats-text">
          {#if dataTotal === 0}
            0 tracks
          {:else}
            {dataTotal} track{dataTotal === 1 ? "" : "s"} · {dataCaptioned} captioned · {encodedLatents} encoded
          {/if}
          {#if uploading}
            <span class="uploading"><i class="bi bi-arrow-clockwise spin"></i> uploading…</span>
          {/if}
          {#if captionerStatus?.running}
            <span class="uploading">
              <i class="bi bi-arrow-clockwise spin"></i>
              captioning {captionerStatus.done}/{captionerStatus.total}
              {#if captionerStatus.cost_so_far > 0} · ${captionerStatus.cost_so_far.toFixed(3)}{/if}
            </span>
          {/if}
        </span>
      </div>

      <div class="folder-pinned">
        <div class="folder-row">
          <input type="text" class="input-full" bind:this={folderInputEl} bind:value={loraFolder}
                 placeholder={loraName ? `default: lora_train_dir/${loraName}/data` : "or point to a folder…"}
                 disabled={loraStatus === "running"}>
          <button class="btn btn-sm folder-upload" onclick={browseLoraFolder}
                  disabled={loraStatus === "running"} title="Point to a folder instead of the default">
            <i class="bi bi-folder2-open"></i> set folder
          </button>
          <button class="btn btn-sm folder-upload right-action" onclick={clearAllUploads}
                  disabled={loraStatus === "running" || dataTotal === 0}
                  title="Delete every file in this LoRA's folder">
            <i class="bi bi-trash"></i> clear uploads
          </button>
        </div>
        <div class="trigger-pinned">
          <input type="text" class="input-full" bind:this={triggerInputEl} bind:value={loraTrigger}
                 placeholder="trigger word (optional)"
                 disabled={loraStatus === "running"}
                 style={folderInputWidth ? `flex: 0 0 ${folderInputWidth}px` : ""}>
          <div class="trigger-status">
            {#if loraTrigger.trim()}
              <i class="bi bi-check-circle-fill encoded-icon"></i>
              <span>→ “{loraTrigger.trim()}, [caption]”</span>
            {:else}
              <i class="bi bi-circle"></i>
              <span>no trigger word set</span>
            {/if}
          </div>
        </div>

        <div class="captioner-pinned">
          <div class="captioner-status">
            {#if captionerStatus?.running}
              <i class="bi bi-arrow-clockwise spin"></i>
              <span><strong>{captionerStatus.done}</strong>/{captionerStatus.total} captioning</span>
            {:else if dataTotal === 0}
              <i class="bi bi-circle"></i>
              <span>no files to caption</span>
            {:else if !captionerEstimate || captionerEstimate.n_files === 0}
              <i class="bi bi-check-circle-fill encoded-icon"></i>
              <span>
                all files captioned
                {#if captionerStatus?.cost_so_far > 0} · ${captionerStatus.cost_so_far.toFixed(3)}{/if}
              </span>
            {:else}
              <i class="bi bi-stars"></i>
              <span>
                <strong>{captionerEstimate.n_files}</strong> file{captionerEstimate.n_files === 1 ? "" : "s"} ·
                {CAPTIONER_MODEL_LABEL[captionerModel]} ·
                {#if captionerEstimate.per_call == null}
                  <span class="hint">first run, no estimate</span>
                {:else}
                  est. ${captionerEstimate.total.toFixed(3)}
                {/if}
              </span>
            {/if}
            <div class="progress-bar">
              <div class="progress-fill"
                   style="width: {capPctShown}%"></div>
            </div>
            <span class="progress-text">{capDoneShown}/{capTotalShown}</span>
          </div>
          <!--
          <button class="btn btn-sm folder-upload icon-only" title="Captioner settings"
                  onclick={() => { session.activeTab = 'settings'; }}>
            <i class="bi bi-gear"></i>
          </button>
          -->
          {#if captionerStatus?.running}
            <button class="btn btn-sm folder-upload" onclick={cancelCaptioner}
                    title="Cancel captioning">
              <i class="bi bi-stop-circle"></i> Cancel
            </button>
          {:else if dataTotal > 0 && (!captionerEstimate || captionerEstimate.n_files === 0)}
            <button class="btn btn-sm folder-upload" onclick={clearAllCaptions}
                    disabled={loraStatus === "running"}
                    title="Delete every caption sidecar in this folder">
              <i class="bi bi-eraser"></i> clear captions
            </button>
          {:else}
            <button class="btn btn-sm folder-upload" onclick={startCaptioner}
                    disabled={loraStatus === "running" || !captionerEstimate || captionerEstimate.n_files === 0}
                    title="Auto-caption uncaptioned files">
              <i class="bi bi-stars"></i> autocaption
            </button>
          {/if}
        </div>

        <div class="pre-encode-pinned">
          <div class="pre-encode-status">
            {#if hasEncoded}
              <i class="bi bi-check-circle-fill encoded-icon"></i>
              <span>pre-encoded</span>
            {:else if preEncodeStatus === "running"}
              <i class="bi bi-arrow-clockwise spin"></i>
              <span>encoding…</span>
            {:else}
              <i class="bi bi-exclamation-circle-fill warn-icon"></i>
              <span>not pre-encoded</span>
            {/if}
            <div class="progress-bar">
              <div class="progress-fill" class:done={hasEncoded} style="width: {preEncodePct}%"></div>
            </div>
            <span class="progress-text">{preEncodeDoneShown}/{preEncodeTotalShown}</span>
          </div>
          {#if preEncodeStatus === "running"}
            <button class="btn btn-sm folder-upload" onclick={cancelPreEncode}
                    title="Cancel pre-encoding">
              <i class="bi bi-stop-circle"></i> cancel
            </button>
          {:else if hasEncoded}
            <button class="btn btn-sm folder-upload" onclick={clearEncodedCache}
                    disabled={loraStatus === "running"}
                    title="Delete the cached latents for this LoRA">
              <i class="bi bi-eraser"></i> clear cache
            </button>
          {:else}
            <button class="btn btn-sm folder-upload" onclick={startPreEncode}
                    disabled={!loraName || dataTotal === 0 || loraStatus === "running"}
                    title="Encode audio files to latents for faster training (cached on disk)">
              <i class="bi bi-lightning-charge"></i> pre-encode
            </button>
          {/if}
        </div>
      </div>

      <input type="file" multiple accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.aiff,.aif"
             bind:this={fileInput} onchange={onFilesPicked} style="display: none">
    </div>
  </aside>
  <section class="pane config-pane">
    <div class="pane-body">

      <section class="form">
      <div class="form-row">
        <label>Name</label>
        <input type="text" class="input-full" bind:value={nameInputValue}
               onblur={onNameBlur} placeholder="lora name" disabled={loraStatus === "running"}>
      </div>
      <div class="form-grid-2">
        <div class="form-row">
          <label>Rank</label>
          <select class="select" bind:value={loraRank} disabled={loraStatus === "running"}>
            <option value={4}>4</option>
            <option value={8}>8</option>
            <option value={16}>16</option>
            <option value={32}>32</option>
          </select>
        </div>
        <div class="form-row">
          <label>Steps</label>
          <input type="number" class="input-full" bind:value={loraSteps}
                 min="100" max="50000" step="100"
                 disabled={loraStatus === "running"}>
        </div>
        <div class="form-row">
          <label>Batch</label>
          <select class="select" bind:value={loraBatch} disabled={loraStatus === "running"}>
            <option value={0}>Auto</option>
            <option value={1}>1</option>
            <option value={2}>2</option>
            <option value={4}>4</option>
            <option value={8}>8</option>
          </select>
        </div>
        <div class="form-row">
          <label>LR</label>
          <input type="number" class="input-full" bind:value={loraLr}
                 min="0" max="0.01" step="0.00001"
                 disabled={loraStatus === "running"}>
        </div>
        <div class="form-row compile-row">
          <label>Compile</label>
          <button type="button" class="switch" class:on={loraCompile}
                  disabled={loraStatus === "running"}
                  onclick={() => loraCompile = !loraCompile}
                  title={loraCompile ? "Turn off torch.compile" : "Turn on torch.compile"}>
            <span class="switch-knob"></span>
          </button>
        </div>
      </div>

      <div class="cta">
        <div class="checklist advisory-row">
          {#each trainAdvisory as a}
            {#if a.state === "info"}
              <span class="check check-info" title={a.detail}>
                {#if a.icon}<i class="bi {a.icon}"></i>{/if}
                {a.label}
              </span>
            {:else}
              <span class="check check-{a.state}" title={a.detail}>
                <i class="bi {a.state === 'pass' ? 'bi-check-circle-fill'
                            : a.state === 'warn' ? 'bi-exclamation-triangle-fill'
                            :                       'bi-x-circle-fill'}"></i>
                {a.label}
              </span>
            {/if}
          {/each}
        </div>
        <div class="checklist">
          {#each trainChecks as c}
            <span class="check check-{c.state}" title={c.detail}>
              <i class="bi {c.state === 'pass' ? 'bi-check-circle-fill'
                          : c.state === 'warn' ? 'bi-exclamation-triangle-fill'
                          :                       'bi-x-circle-fill'}"></i>
              {c.label}
            </span>
          {/each}
        </div>
        <div class="train-row">
          <button class="btn train-btn"
                  onclick={startLoraTraining}
                  disabled={!canTrain}>
            {#if loraStatus === "running"}
              <i class="bi bi-hourglass-split"></i> {loraStage}
            {:else}
              <i class="bi bi-lightning"></i> start training
            {/if}
          </button>
          {#if loraStatus === "running"}
            <button class="profile-btn" onclick={cancelLoraTraining}
                    title="Stop the training subprocess">
              <i class="bi bi-stop-circle"></i> cancel
            </button>
          {:else}
            <button class="profile-btn" onclick={runProfile}
                    disabled={profileRunning}
                    title="Run a few dummy steps to measure step time and memory, then extrapolate the full run">
              <i class="bi {profileRunning ? 'bi-hourglass-split' : 'bi-speedometer2'}"></i>
              {profileRunning ? "Profiling…" : "Test system"}
            </button>
          {/if}
        </div>
        <div class="train-progress">
          <div class="train-progress-bar">
            <div class="train-progress-fill"
                 class:done={loraStatus === "done"}
                 style="width: {loraSteps > 0 ? Math.min(100, loraStep * loraTrainBatch / loraSteps * 100) : 0}%"></div>
          </div>
          <span class="train-progress-text">
            {loraStep * loraTrainBatch}/{loraSteps}
            {#if loraStatus === "running" || loraStatus === "done"}
              · {fmtDuration(loraElapsedSec)}
              {#if loraEtaSec != null} / {fmtDuration(loraElapsedSec + loraEtaSec)}{/if}
            {/if}
          </span>
        </div>

        {#snippet metric(label, value, history, color, xLabel, xLabels)}
          <div class="metric-tile">
            <div class="metric-tile-head">
              <span class="metric-label">{label}</span>
              <span class="metric-value">{value ?? "—"}</span>
            </div>
            <div class="metric-tile-chart">
              <SparkLine values={history} color={color}
                         xLabel={xLabel} xLabels={xLabels}
                         hoverFrac={hoverFrac}
                         onHover={onMetricHover}
                         onLeave={onMetricLeave} />
            </div>
          </div>
        {/snippet}
        <div class="metric-grid">
          {@render metric("loss",
                          loraStatus === "running" || loraStatus === "done" ? loraLoss.toFixed(4) : null,
                          lossHistory, "#0078ca", "step", lossSteps)}
          {@render metric("grad norm",
                          loraGradNorm != null ? loraGradNorm.toFixed(3) : null,
                          gradHistory, "#d6b34a", "step", lossSteps)}
          {@render metric(
            session.backend === "cuda" ? "cuda %" : "cpu %",
            (session.stats?.cpu != null) ? `${session.stats.cpu}%` : null,
            cpuHistory, "#4ec9b0", "sec", null
          )}
          {@render metric(
            session.backend === "cuda" ? "vram" : "ram",
            session.backend === "cuda"
              ? (session.stats?.gpuAllocGb != null ? `${session.stats.gpuAllocGb.toFixed(1)} GB` : null)
              : (session.stats?.ramUsedGb  != null ? `${session.stats.ramUsedGb.toFixed(1)} / ${session.stats.ramTotalGb?.toFixed(0)} GB` : null),
            ramHistory, "#a07acc", "sec", null
          )}
        </div>
      </div>
      </section>
    </div>
  </section>

  {#if editingFile}
    <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
    <div class="caption-pop" style={captionPopStyle}
         onmousedown={(e) => e.stopPropagation()}>
      <div class="caption-pop-header">
        <img class="caption-pop-thumb" alt=""
             src={`/api/lora_data/thumb?name=${encodeURIComponent(loraName)}&folder=${encodeURIComponent(loraFolder || "")}&file=${encodeURIComponent(editingFile)}`} />
        <div class="caption-pop-name" title={editingFile}>{editingFile}</div>
      </div>
      {#if captionLoading}
        <div class="caption-pop-loading">loading…</div>
      {:else}
        <textarea bind:this={captionTextarea}
                  bind:value={captionText}
                  onkeydown={onCaptionKeyDown}
                  oninput={onCaptionInput}
                  onblur={() => closeCaptionEditor(true)}
                  placeholder="describe this clip…"
                  rows="3"></textarea>
        <div class="caption-pop-hint">
          {captionText.length} chars · ↵ to save · Esc to close
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
.view {
  display: grid;
  grid-template-columns: 1fr 1fr;
  background: var(--bg-dark);
  min-height: 0;
  overflow: hidden;
  flex: 1;
  height: 100%;
}
.pane {
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}
.dataset-body > .drop-zone { transition: filter 140ms ease; }
.dataset-body.dim > .drop-zone { filter: brightness(0.5); }
.pane-body {
  flex: 1;
  overflow-y: auto;
  padding: 48px;
  min-height: 0;
}
.config-pane > .pane-body { display: flex; flex-direction: column; }
.config-pane > .pane-body > .form { flex: 1; min-height: 0; display: flex; flex-direction: column; }
.dataset-body {
  display: flex;
  flex-direction: column;
  height: 100%;
  box-sizing: border-box;
  gap: var(--gap-3);
}
.drop-zone {
  flex: 1;
  border: 2px dashed var(--border-color);
  border-radius: 8px;
  background: var(--bg-dark);
  color: var(--text-muted);
  font-size: 12px;
  font-family: inherit;
  padding: 5px;
  cursor: pointer;
  transition: border-color 120ms ease, color 120ms ease, background-color 120ms ease;
  overflow: hidden;
  min-height: 0;
}
.drop-scroll {
  height: 100%;
  overflow-y: auto;
  padding: 28px 19px 19px;
  box-sizing: border-box;
}
.drop-zone:hover { border-color: var(--text-muted); color: var(--text-secondary); }
.drop-zone.active {
  border-color: var(--accent-blue);
  background: rgba(0, 120, 202, 0.06);
  color: var(--text-secondary);
}
.drop-zone:focus-visible { outline: none; border-color: var(--accent-blue); }
.drop-empty {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  gap: var(--gap-2);
  padding: var(--gap-4);
}
.drop-empty i { font-size: 32px; opacity: 0.5; }
.drop-title { font-size: 13px; color: var(--text-secondary); }
.drop-empty code {
  font-family: ui-monospace, "JetBrains Mono", "SF Mono", Menlo, monospace;
  font-size: 11px;
  color: var(--text-secondary);
}

.file-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(96px, 1fr));
  gap: 12px 8px;
  align-content: start;
}
.file-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  cursor: pointer;
}
.file-card.selected .file-thumb { border-color: var(--accent-blue); }
.file-card.selected .file-name { color: var(--text-primary); }
.file-thumb {
  position: relative;
  width: 72px;
  aspect-ratio: 1 / 1;
  background: var(--bg-lighter);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-secondary);
  transition: border-color 120ms ease;
}
.file-thumb:hover { border-color: var(--text-muted); }
/*
.file-thumb > .thumb-fallback {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(calc(-50% - 2px), calc(-50% + 2px));
  font-size: 26px;
  color: rgba(255,255,255,0.9);
  pointer-events: none;
  z-index: 1;
}
*/
.thumb-img {
  position: absolute;
  inset: 0;
  width: 100%; height: 100%;
  object-fit: cover;
  border-radius: 5px;
  opacity: 0;
  transition: opacity 250ms ease;
  pointer-events: none;
}
.thumb-img.loaded { opacity: 1; }
.file-x {
  position: absolute;
  top: 4px;
  right: 4px;
  width: 18px;
  height: 18px;
  border: none;
  background: rgba(0,0,0,0.55);
  color: var(--text-secondary);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  cursor: pointer;
  opacity: 0;
  transition: opacity 120ms ease, background-color 120ms ease, color 120ms ease;
}
.file-card:hover .file-x { opacity: 1; }
.file-x:hover { background: rgba(220, 70, 70, 0.8); color: #fff; }
.file-cap-badge {
  position: absolute;
  bottom: 4px;
  right: 4px;
  width: 18px;
  height: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  color: rgba(255,255,255,0.85);
  background: rgba(0,0,0,0.55);          /* grey — no caption */
  border: 0;
  border-radius: 50%;
  cursor: pointer;
  z-index: 2;
  transition: background-color 120ms ease, color 120ms ease;
}
.file-cap-badge:hover { background: rgba(0,0,0,0.85); }
.file-cap-badge.captioned {
  color: rgba(255,255,255,0.95);
}
.file-cap-badge.captioned:hover {
  background: rgba(0,0,0,0.85);
  color: #fff;
}

/* ── caption editor popover ── */
.caption-pop {
  position: fixed;
  z-index: 200;
  background: var(--bg-lighter);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  padding: 10px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.55);
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.caption-pop-header { display: flex; align-items: center; gap: 8px; min-width: 0; }
.caption-pop-thumb {
  width: 36px; height: 36px; border-radius: 4px; object-fit: cover; flex-shrink: 0;
  background: #0a0a0a;
}
.caption-pop-name {
  font-size: 11px; color: var(--text-secondary);
  font-family: ui-monospace, "JetBrains Mono", "SF Mono", Menlo, monospace;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.caption-pop textarea {
  width: 100%;
  resize: vertical;
  min-height: 80px;
  background: var(--bg-dark);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  padding: 6px 8px;
  color: var(--text-primary);
  font-size: 12px;
  font-family: inherit;
  box-sizing: border-box;
}
.caption-pop textarea:focus { outline: none; border-color: var(--accent-blue); }
.caption-pop-hint { font-size: 10px; color: var(--text-muted); text-align: right; }
.caption-pop-loading { font-size: 11px; color: var(--text-muted); text-align: center; padding: var(--gap-3); }
.file-name {
  width: 96px;
  font-size: 11px;
  color: var(--text-primary);
  text-align: center;
  line-height: 1.35;
  word-break: break-word;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.data-stats {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--gap-2);
  font-size: 11px;
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
}
.data-stats-text { display: inline-flex; align-items: center; gap: var(--gap-2); padding-left: 5px; }
.data-stats .uploading {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: var(--accent-blue);
}
.spin { animation: spin 1s linear infinite; display: inline-block; }
@keyframes spin { to { transform: rotate(360deg); } }

/* ── captioner summary row (left dataset pane, above pre-encode) ── */
.captioner-pinned {
  display: flex; align-items: center; gap: var(--gap-2);
  flex-shrink: 0;
}
.captioner-status {
  flex: 1; min-width: 0;
  display: flex; align-items: center; gap: var(--gap-2);
  font-size: 11px; color: var(--text-muted);
  padding-left: 5px;
}
.captioner-status .hint { color: var(--text-muted); }
.icon-only { padding: 4px 8px; }
/* Captioner row uses the shared .progress-bar / .progress-fill / .progress-text
   styles from the pre-encode block — same dimensions and behavior. */
.captioner-status .progress-bar { flex: 1; }
.cap-summary-row { display: flex; align-items: center; gap: var(--gap-3); flex-wrap: wrap; }
.cap-summary-text { flex: 1; font-size: 11px; color: var(--text-secondary); min-width: 0; }
.cap-summary-text .hint { color: var(--text-muted); }
.advanced-toggle {
  font-size: 11px;
  color: var(--text-muted);
  cursor: pointer;
  text-decoration: underline;
  text-decoration-color: rgba(255,255,255,0.2);
  text-underline-offset: 3px;
  white-space: nowrap;
}
.advanced-toggle:hover { color: var(--text-primary); text-decoration-color: rgba(255,255,255,0.6); }
.cap-advanced {
  display: flex;
  flex-direction: column;
  gap: var(--gap-3);
  padding-top: var(--gap-2);
  border-top: 1px solid var(--border-color);
}
.modal-header { display: flex; justify-content: space-between; align-items: center; font-size: 13px; }
.modal-close { background: none; border: 0; color: var(--text-muted); font-size: 20px; cursor: pointer; line-height: 1; }
.modal-close:hover { color: var(--text-primary); }
.modal-row { display: flex; align-items: center; gap: var(--gap-3); }
.modal-row label { font-size: 11px; color: var(--text-secondary); width: 80px; text-transform: uppercase; letter-spacing: 0.04em; }
.modal-row-stack { flex-direction: column; align-items: stretch; gap: 6px; }
.modal-row-stack label { width: auto; }
.modal-row .hint { color: var(--text-muted); text-transform: none; letter-spacing: normal; font-size: 10px; }
.model-toggle { display: flex; gap: 0; border: 1px solid #2c2c2c; border-radius: 4px; overflow: hidden; }
.model-toggle button {
  flex: 1; background: #1a1a1a; color: var(--text-secondary);
  border: 0; padding: 6px 12px; font-size: 11px; cursor: pointer;
  display: inline-flex; align-items: center; gap: 5px; justify-content: center;
}
.model-toggle button.active { background: var(--accent-blue); color: #fff; }
.model-toggle button .hint { color: rgba(255,255,255,0.6); font-size: 10px; }
.model-toggle button.active .hint { color: rgba(255,255,255,0.85); }
.examples-textarea {
  background: var(--bg-dark);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  padding: 8px 10px;
  color: var(--text-primary);
  font-family: inherit; font-size: 12px;
  resize: vertical;
}
.examples-textarea:focus { outline: none; border-color: var(--accent-blue); }
.modal-summary { font-size: 11px; color: var(--text-secondary); padding: 4px 0; }
.modal-summary .hint { color: var(--text-muted); }
.modal-actions { display: flex; justify-content: flex-end; gap: var(--gap-2); padding-top: var(--gap-2); }

.cap-progress { display: flex; flex-direction: column; gap: var(--gap-2); }
.cap-progress-text { font-size: 11px; color: var(--text-secondary); }
.cap-recent { display: flex; flex-direction: column; gap: 3px; max-height: 140px; overflow-y: auto; font-size: 11px; padding: var(--gap-2) 0; }
.cap-recent-row { display: flex; gap: var(--gap-2); }
.cap-recent-file { color: var(--text-secondary); flex: 0 0 160px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: ui-monospace, "JetBrains Mono", "SF Mono", Menlo, monospace; }
.cap-recent-text { color: var(--text-primary); }

.folder-pinned { flex-shrink: 0; display: flex; flex-direction: column; gap: var(--gap-2); }
.folder-upload { flex-shrink: 0; display: inline-flex; align-items: center; gap: 6px; }
/* Clear uploads and Pre-encode buttons sit in the same x-position; explicit
   matched width keeps them visually paired. */
.right-action { width: 132px; justify-content: center; }
.trigger-pinned {
  display: flex;
  align-items: center;
  gap: var(--gap-2);                  /* match .folder-row gap so right-edge aligns */
}
.trigger-pinned .input-full { flex: 1; min-width: 0; }
.trigger-status {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: var(--gap-2);
  font-size: 11px;
  color: var(--text-muted);
  /* trigger input width is matched to the folder input via ResizeObserver,
     so the status fills the remaining space naturally. */
  flex: 1; min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  padding-right: 5px;
}
.pre-encode-pinned {
  display: flex;
  align-items: center;
  gap: var(--gap-3);
}
.pre-encode-btn { flex: 0 0 132px; justify-content: center; }
.pre-encode-status {
  display: flex;
  align-items: center;
  gap: var(--gap-2);
  font-size: 11px;
  color: var(--text-muted);
  flex: 1;
  min-width: 0;
  padding-left: 5px;
}
.progress-bar {
  flex: 1;
  height: 4px;
  background: var(--border-color);
  border-radius: 2px;
  overflow: hidden;
  min-width: 0;
}
.progress-fill {
  height: 100%;
  background: var(--accent-blue);
  transition: width 200ms ease, background-color 200ms ease;
}
.progress-fill.done { background: var(--success-green, #4ec9b0); }
.progress-text {
  font-size: 11px;
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.encoded-icon { color: var(--success-green, #4ec9b0); }
.warn-icon { color: #d6b34a; }
.muted { color: var(--text-muted); }
.small { font-size: 11px; max-width: 320px; line-height: 1.5; }
.form { display: flex; flex-direction: column; gap: var(--gap-3); }
.form-grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  column-gap: 32px;
  row-gap: var(--gap-3);
}
/* iOS-style switch for boolean fields */
.switch {
  width: 32px; height: 18px;
  background: #2c2c2c;
  border: 1px solid #3a3a3a;
  border-radius: 9px;
  position: relative;
  cursor: pointer;
  padding: 0;
  flex-shrink: 0;
  transition: background-color 120ms ease;
}
.switch.on { background: var(--accent-blue); border-color: var(--accent-blue); }
.switch:hover:not([disabled]) { filter: brightness(1.1); }
.switch[disabled] { opacity: 0.4; cursor: default; }
.switch-knob {
  position: absolute;
  top: 1px; left: 1px;
  width: 14px; height: 14px;
  background: #fff;
  border-radius: 50%;
  transition: transform 140ms cubic-bezier(.4,.2,.2,1);
}
.switch.on .switch-knob { transform: translateX(14px); }
.form-row {
  display: grid;
  grid-template-columns: 70px 1fr;
  align-items: center;
  gap: 6px;
}
.form-row label {
  font-size: 11px;
  color: var(--text-secondary);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.input-full, .select {
  background: var(--bg-dark);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  padding: 6px 8px;
  color: var(--text-primary);
  font-size: 12px;
  font-family: inherit;
  width: 100%;
}
.input-full:focus, .select:focus { outline: none; border-color: var(--accent-blue); }
/* Pull the dropdown caret in close to the value text instead of pinning it
   to the far right edge. */
.select {
  appearance: none;
  -webkit-appearance: none;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 6' fill='none' stroke='%23888' stroke-width='1.5'><path d='M1 1.5l4 3 4-3'/></svg>");
  background-repeat: no-repeat;
  background-position: right 10px center;
  background-size: 9px 6px;
  padding-right: 24px;
}
.folder-row { display: flex; gap: var(--gap-2); }
.folder-row .input-full { flex: 1; }
.btn-sm {
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 11px;
  cursor: pointer;
  background: #1a1a1a;
  color: var(--text-primary);
  border: 1px solid #2c2c2c;
}
.btn-sm:hover { background: #262626; border-color: #3a3a3a; }
.btn-sm[disabled] { opacity: 0.4; cursor: default; }
.slider-row { display: flex; align-items: center; gap: var(--gap-2); }
.slider-row .slider { flex: 1; }
.value { font-variant-numeric: tabular-nums; color: var(--text-primary); font-size: 11px; min-width: 50px; text-align: right; }
.toggle-sm { display: flex; align-items: center; gap: var(--gap-2); cursor: pointer; }
.toggle-sm input { margin: 0; }
.toggle-sm span { font-size: 11px; color: var(--text-secondary); }
.pre-encode-row { display: flex; align-items: center; gap: var(--gap-2); }
.encoded-badge {
  font-size: 10px;
  color: var(--success-green, #4ec9b0);
  background: rgba(78, 201, 176, 0.1);
  padding: 2px 6px;
  border-radius: 3px;
}
.small { font-size: 11px; }
.text-muted { color: var(--text-muted); }

.cta {
  display: flex;
  flex-direction: column;
  gap: var(--gap-2);
  margin-top: var(--gap-3);
  flex: 1 1 auto;
  min-height: 0;
}
.checklist {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px 14px;
  font-size: 11px;
  color: var(--text-muted);
  padding: 4px 2px;
}
.check { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
.check i { font-size: 12px; }
.check-pass { color: var(--success-green, #4ec9b0); }
.check-warn { color: #d6b34a; }
.check-fail { color: #d96a6a; }
.check-info { color: var(--text-primary); }
.train-row { display: flex; gap: var(--gap-2); }
.train-row .train-btn { flex: 1; }
.profile-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 0 14px;
  border-radius: 4px;
  border: 1px solid #2c2c2c;
  background: #1a1a1a;
  color: var(--text-primary);
  font-size: 11px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  cursor: pointer;
}
.profile-btn:hover { background: #262626; border-color: #3a3a3a; }
.profile-btn[disabled] { opacity: 0.4; cursor: default; }
.profile-result {
  font-size: 11px;
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
  padding: 2px 4px;
}
.train-btn {
  background: var(--accent-blue);
  color: #fff;
  border: 1px solid var(--accent-blue);
  border-radius: 4px;
  padding: 9px 16px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--gap-2);
}
.train-btn:hover { filter: brightness(1.1); }
.train-btn[disabled] { opacity: 0.4; cursor: default; }
.train-btn[disabled]:hover { filter: none; }
.train-progress {
  display: flex; align-items: center; gap: var(--gap-2);
  padding: var(--gap-4) 0 calc(var(--gap-3) + 5px);
}
.train-progress-bar { flex: 1; height: 14px; background: var(--border-color); border-radius: 0; overflow: hidden; }
.compile-row { padding-top: 5px; }
.train-progress-fill {
  height: 100%; background: var(--accent-blue);
  transition: width 200ms ease, background-color 200ms ease;
}
.train-progress-fill.done { background: var(--success-green, #4ec9b0); }
.train-progress-text { font-size: 10px; color: var(--text-muted); font-variant-numeric: tabular-nums; }

/* 2x2 metrics grid below the training pbar — fills remaining vertical space.
   Don't use margin-top: auto here: it absorbs the leftover space that
   flex-grow needs to expand the grid on window resize. flex:1 alone is enough
   to push it to the bottom *and* grow it. */
.metric-grid {
  flex: 1 1 0;
  min-height: 240px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  gap: var(--gap-2);
  padding-top: var(--gap-2);
}
.metric-tile {
  display: flex; flex-direction: column; gap: 4px;
  padding: 8px 10px;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  background: rgba(255,255,255,0.02);
  min-height: 0;
  overflow: hidden;
}
.metric-tile-head {
  display: flex; align-items: baseline; justify-content: space-between;
  gap: var(--gap-2);
}
.metric-tile-chart { flex: 1; min-height: 0; }
.metric-label {
  font-size: 10px; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.06em;
}
.metric-value {
  font-size: 13px; color: var(--text-primary);
  font-variant-numeric: tabular-nums;
}
</style>
