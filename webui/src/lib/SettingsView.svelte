<script>
import { onMount } from "svelte";
import { apiGetSettings, apiSaveSettings } from "./session.svelte.js";

let settings = $state({
  models_dir: "",
  lora_dir: "",
  lora_train_dir: "",
  sa3_root: "",
  hf_token: "",
  openrouter_api_key: "",
  captioner_model: "pro",
  captioner_parallel: 32,
  captioner_examples: "",
  lora_adapter: "dora-rows",
});
let loading = $state(true);
let savedAt = $state(0);
let saveFlashTimer = 0;

const DEFAULT_CAPTIONER_EXAMPLES = [
  "soundscape with piano arps, indie pop with male vocals, offbeat bassline with house drums and ambient electric guitar",
  "bright synth leads, pitched vocal chops, disco house, euphoric and danceable",
  "ethereal electric piano intro, fast 808 melody, experimental drill, plucked guitar outro",
].join("\n");

const groups = [
  {
    title: "General / Inpainting",
    fields: [
      { key: "models_dir", label: "Models directory",  hint: "Where SA3 model weights are stored" },
      { key: "sa3_root",   label: "SA3 source root",   hint: "Path to stable-audio-3 repo clone" },
      { key: "hf_token",   label: "HuggingFace token", hint: "Required for gated model downloads", password: true, placeholder: "hf_..." },
    ],
  },
  {
    title: "LoRA Training",
    fields: [
      { key: "lora_dir",           label: "LoRA output directory",   hint: "Where trained LoRA .safetensors are saved" },
      { key: "lora_train_dir",     label: "LoRA working directory",  hint: "Cached latents and training artifacts" },
      { key: "openrouter_api_key", label: "OpenRouter API key",      hint: "Required for auto-captioning", password: true, placeholder: "sk-or-..." },
    ],
  },
];

async function load() {
  loading = true;
  const s = await apiGetSettings();
  if (s) {
    delete s.first_run;
    Object.assign(settings, s);
  }
  // Backfill defaults for fields that arrived empty from older settings files
  if (!settings.captioner_examples || !settings.captioner_examples.trim()) {
    settings.captioner_examples = DEFAULT_CAPTIONER_EXAMPLES;
  }
  if (!settings.captioner_model)    settings.captioner_model = "pro";
  if (!settings.captioner_parallel) settings.captioner_parallel = 32;
  loading = false;
}

let lastSaved = ""; // JSON snapshot of last saved settings, to skip no-op saves
let saving = false;

async function saveIfChanged() {
  if (loading || saving) return;
  const snap = JSON.stringify(settings);
  if (snap === lastSaved) return;
  saving = true;
  const result = await apiSaveSettings(settings);
  if (result) {
    Object.assign(settings, result);
    lastSaved = JSON.stringify(settings);
    savedAt = Date.now();
    if (saveFlashTimer) clearTimeout(saveFlashTimer);
    saveFlashTimer = setTimeout(() => { savedAt = 0; }, 1500);
  }
  saving = false;
}

onMount(async () => {
  await load();
  lastSaved = JSON.stringify(settings);
});
</script>

<div class="view">
  <div class="view-inner">
    {#if loading}
      <div class="loading">Loading…</div>
    {:else}
      <div class="columns">
        <!-- Left column: general / inpainting -->
        <section class="group">
          <h2 class="group-title">{groups[0].title}</h2>
          <div class="fields">
            {#each groups[0].fields as f}
              <label class="field">
                <span class="field-label">{f.label}</span>
                <input type={f.password ? "password" : "text"} class="field-input"
                       bind:value={settings[f.key]} onblur={saveIfChanged}
                       placeholder={f.placeholder ?? f.hint} />
                <span class="field-hint">{f.hint}</span>
              </label>
            {/each}
          </div>
        </section>

        <!-- Right column: LoRA training + autocaptioner -->
        <section class="group">
          <h2 class="group-title">{groups[1].title}</h2>
          <div class="fields">
            {#each groups[1].fields as f}
              <label class="field">
                <span class="field-label">{f.label}</span>
                <input type={f.password ? "password" : "text"} class="field-input"
                       bind:value={settings[f.key]} onblur={saveIfChanged}
                       placeholder={f.placeholder ?? f.hint} />
                <span class="field-hint">{f.hint}</span>
              </label>
            {/each}

            <div class="field">
              <span class="field-label">Autocaptioner model</span>
              <div class="model-toggle-row">
                <div class="model-toggle-settings">
                  <button class:active={settings.captioner_model === "pro"}
                          onclick={() => { settings.captioner_model = "pro"; saveIfChanged(); }}>Pro</button>
                  <button class:active={settings.captioner_model === "flash"}
                          onclick={() => { settings.captioner_model = "flash"; saveIfChanged(); }}>Flash</button>
                </div>
                <span class="model-blurb">
                  {settings.captioner_model === "pro"
                    ? "more expensive, higher accuracy, doesn't hallucinate as badly"
                    : "cheaper, significantly less accurate captions"}
                </span>
              </div>
            </div>
            <label class="field">
              <span class="field-label">LoRA adapter</span>
              <select class="field-input field-select" bind:value={settings.lora_adapter}
                      onchange={saveIfChanged}>
                <option value="dora-rows">DoRA (recommended)</option>
                <option value="lora">LoRA</option>
                <option value="lora-xs">LoRA-XS (low VRAM)</option>
                <option value="bora">BoRA</option>
              </select>
              <span class="field-hint">Adapter architecture used for training.</span>
            </label>
            <label class="field">
              <span class="field-label">Autocaptioner parallel requests</span>
              <select class="field-input field-select" bind:value={settings.captioner_parallel}
                      onchange={saveIfChanged}>
                {#each [8, 16, 32, 64, 128] as n}<option value={n}>{n}</option>{/each}
              </select>
            </label>
            <label class="field">
              <span class="field-label">Autocaptioner example captions</span>
              <textarea class="field-input field-textarea" rows="5"
                        bind:value={settings.captioner_examples}
                        onblur={saveIfChanged}></textarea>
              <span class="field-hint">One example per line — sets the caption style via few-shot prompt.</span>
            </label>
          </div>
        </section>
      </div>

      <p class="footnote">Stored locally in <code>~/.config/sa3-inpainter/settings.json</code>.</p>

      <div class="save-flash" class:visible={savedAt > 0}>
        <i class="bi bi-check2"></i> saved
      </div>
    {/if}
  </div>
</div>

<style>
.view { overflow-y: auto; background: var(--bg-dark); min-height: 0; }
.view-inner {
  max-width: 1100px;
  margin: 0 auto;
  padding: 96px var(--gap-4) var(--gap-5);
  display: flex;
  flex-direction: column;
  gap: var(--gap-4);
  width: 100%;
}
.loading { color: var(--text-muted); font-size: 12px; }
.footnote {
  font-size: 11px;
  color: var(--text-muted);
  margin: 0;
  padding-top: var(--gap-3);
}
.footnote code {
  font-family: ui-monospace, "JetBrains Mono", "SF Mono", Menlo, monospace;
  color: var(--text-secondary);
  background: rgba(255,255,255,0.04);
  padding: 1px 5px;
  border-radius: 3px;
}

.columns {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--gap-5);
  align-items: start;
}
@media (max-width: 760px) {
  .columns { grid-template-columns: 1fr; }
}
.group {
  display: flex;
  flex-direction: column;
  gap: var(--gap-3);
}
.group-title {
  font-size: 10px;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin: 0 0 var(--gap-2);
}
.fields { display: flex; flex-direction: column; gap: 28px; }
.field { display: flex; flex-direction: column; gap: 5px; }
.field-label {
  font-size: 10px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.field-input {
  background: var(--bg-dark);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  padding: 7px 9px;
  color: var(--text-primary);
  font-size: 12px;
  font-family: ui-monospace, "JetBrains Mono", "SF Mono", Menlo, monospace;
}
.field-input:focus { outline: none; border-color: var(--accent-blue); }
.field-textarea { resize: vertical; font-family: inherit; }
.field-select { font-family: inherit; cursor: pointer; }
.field-hint { font-size: 10px; color: var(--text-muted); }
.model-toggle-row {
  display: flex; align-items: center; gap: var(--gap-3);
}
.model-toggle-settings {
  display: inline-flex; border: 1px solid var(--border-color); border-radius: 4px; overflow: hidden;
  flex-shrink: 0;
}
.model-toggle-settings button {
  background: var(--bg-dark); color: var(--text-secondary);
  border: 0; padding: 7px 18px; cursor: pointer;
  font-size: 12px; font-weight: 600;
}
.model-toggle-settings button + button { border-left: 1px solid var(--border-color); }
.model-toggle-settings button.active {
  background: rgba(0, 120, 202, 0.18);
  color: var(--text-primary);
}
.model-blurb { font-size: 11px; color: var(--text-muted); }
.save-flash {
  position: fixed;
  bottom: calc(var(--bottombar-h) + var(--gap-3));
  right: var(--gap-4);
  color: var(--success-green, #4ec9b0);
  background: rgba(78, 201, 176, 0.1);
  border: 1px solid rgba(78, 201, 176, 0.3);
  padding: 5px 10px;
  border-radius: 4px;
  font-size: 11px;
  opacity: 0;
  transform: translateY(4px);
  transition: opacity 180ms ease, transform 180ms ease;
  pointer-events: none;
}
.save-flash.visible { opacity: 1; transform: translateY(0); }
</style>
