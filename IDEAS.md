# IDEAS

Things we want to build but aren't building yet. Rough sketches, not commitments.

---

## Browser tab (placeholder shipped)

Unified library that treats audio samples and trained LoRAs as first-class
citizens of the same space. Replaces "go into the filesystem to find your
samples + scroll a dropdown for your LoRAs."

**Querying — three orthogonal modalities, stackable:**

- **CLAP** — "similar vibe" by 512-d embedding distance. Seed search ("more
  like this") and free-text search ("tribal percussion") both fall out.
- **MIR features** — scalar facets extracted at index time: brightness,
  harmonicity, noisiness, BPM, key, attack time, spectral centroid, RMS,
  LUFS. Filter as sliders ("brightness > 0.6"). Display as inline mini-bars
  in the table cells, sononym-style.
- **Tags / auto-class** — named buckets (snares, kicks, vocals…). Manual tags
  + auto-classification at index time.

The power query is all three stacked: `snares (tag) + bright > 0.6 (MIR) +
similar to selected (CLAP) + NOT contains "trap" (text)`.

**LoRAs as first-class:**

- Each LoRA gets a sidecar JSON: `{training_set_paths, centroid_clap,
  centroid_mir, trigger_word, created_at, parent_lora_if_finetuned}`.
- Centroid = mean CLAP embedding over training set → the LoRA gets a position
  in the same space as samples.
- Aggregate MIR stats over training set → LoRA shows up under the same
  brightness slider, BPM filter, etc.
- Auto-tag by dominant category in training set ("90% snares → snare LoRA").

**Workflows this unlocks:**

- Hear a sample, "find LoRAs trained on similar material."
- Preview a LoRA, "show samples most representative of its training set."
- "I want to train a LoRA on dub bass" → seed search → multi-select → "make
  training folder" button → opens in Trainer tab pre-populated.
- Find that LoRA you forgot the trigger word for, by filtering `brightness <
  0.4, harmonicity > 0.7, bpm ~140`.

**UMAP panel — navigation, not eye candy:**

- WebGL scatter of CLAP or MIR embeddings (toggle which).
- Color knob: HDBSCAN cluster / tag / scalar feature ("color by brightness")
  / "in LoRA X's training set y/n" / "used in inpaint y/n."
- LoRA centroids overlaid as larger ringed points; surrounded by their
  training set.
- **Lasso → batch ops.** Drag a region, that's a multi-select with semantic
  meaning. "Make training folder from this cluster." "Tag as snares."
- **Filter coupling.** When the bottom filters narrow the table, the UMAP
  *dims* non-matching points rather than removing them — you see WHERE in
  your library the filter cuts.
- **Hover-to-play.** Point → audio in your ear. Click to anchor.

**Aesthetic constraints (per Lyra):**

- No two lists side-by-side. One pane is the source of truth, the other is a
  canvas — UMAP / spectrogram / waveform grid.
- The "find similar" mode is a *filter on the main grid,* not its own window
  with its own file list. Sononym/wavdesk separate them; we don't.
- Negative filters (NOT-contains) as a real first-class operator. People
  think in exclusions ("anything but trap snares") as often as inclusions.

**Indexer:**

- `~/.config/sa3-inpainter/sample_index.sqlite` — path, hash, CLAP embedding,
  MIR features, tags, mtime. Re-scan on demand, watch folders eventually.
- Cost-aware: CLAP is the slow step (~100ms/file on M-series). Run in a
  background worker, index in priority order (visible-first).

---

## Direct latent photoshop

Photoshop on the latent grid itself. New tab. The premise is *paintbrush in
(channel, time) space* — arbitrary 2D region selection in the latent grid,
not whole channels or whole time slices.

**Canvas:**

- 2D heatmap. X = time (latent frames, ~3-13 Hz depending on model), Y =
  latent channel (256 for SAME-L, 8 for SAME-S). Each cell is a scalar
  value, colormapped (diverging palette, zero-centered).
- Zoomable like the inpainter (scroll = zoom anchored at cursor, shift-
  scroll pan, separate zoom on each axis).
- Selection: marquee, lasso, per-channel-or-per-frame stripes, and *channel-
  set selection* (e.g. `[ch 3,7,8,14,32] × [frame 536-620]` — pick non-
  contiguous channels by clicking the channel labels, then time-window with
  a horizontal range).

**Operations on selection:**

- amplify / attenuate (multiply: ×0.5, ×2, ×5…)
- brighten / darken (scalar add)
- invert (negate)
- smooth (gaussian blur in time or channel direction independently)
- noise (add gaussian, scaled by selection magnitude or absolute)
- envelope (draw a multiplier curve over time within the selection)
- copy / paste / move (relocate latent chunks across time)
- denoise toward zero (shrink, useful for cleaning artifacts)
- morph toward another sample's latent at the same coords (interp)
- **inpaint just this region** (run SA3 inpainting on the time range that
  the (C, T) selection covers, with a prompt). Selection stays after the
  inpaint completes, so you can re-roll with different seeds / prompts /
  ops without redoing the selection. Same pattern as the inpainter mask
  persisting after generate, but more valuable here because rich multi-
  channel non-contiguous selections are expensive to re-pay for.

**Live decode preview** — debounced re-decode after each op so you hear what
you're doing.

**Measured M5 decode latency (MLX, SAME-L, no warmup hit):**

```
 1.5s clip  →   28ms
 3s         →   52ms
 6s         →  105ms
10s         →  203ms
18s         →  500ms
30s         →  750ms
60s         → 1500ms
```

~25ms per second of audio, perfectly linear. The Photoshop-feel threshold
(500ms total) holds for any edit window up to ~18s. **For the common case
— paint in a region, decode only that region + overlap padding — you're
under 150ms for windows up to ~6s, well into "feels live" territory.**

So the editor is viable. Strategy: decode only the affected time-range on
each stroke, full-clip re-render becomes a "render" button for global ops
(30s = 750ms is fine for a deliberate render, bad for a live brush
stroke). Undo/redo via the existing history machinery.

**Why this is good — what the channel-axis dispute taught me:**

The earlier instinct ("only useful if channels are disentangled") was wrong,
because that's not what you need. You need channels to be *coherent under
perturbation* — i.e. boosting a multi-channel selection produces a
recognizable artifact, not pure noise. The VAE was trained to reconstruct,
so neighboring channels and neighboring times co-vary smoothly; perturbing
them produces structured damage, which is a sound, which is the point.

The discovery loop *is* the value: select → apply → decode → listen → undo,
repeat. After 20 minutes the user has an intuition for "those channels in
the bottom third do something low-end-y when multiplied" that no amount of
interp work would give you. The user is the probe, the tool is the
instrument.

What this is good for:

- **Sounds prompting can't reach.** Weird textures, glitches with structure,
  partial degradations that aren't bitcrush. Prompts only reach prompt-
  space; this reaches any point the VAE can decode.
- **Local damage as effect.** Take a clean inpaint, hand-corrupt a 2-second
  window's energy channels → "stutter break that didn't come from a sample
  pack."
- **Latent-domain automation.** Draw a multiplier envelope down a channel
  over time — slowly 1.0→3.0→0.5 across 8 bars on dimensions 12-20. That's
  not an effect; it's modulation on a representation no DAW has access to.
- **Combines with the browser.** "Find samples whose latents look like
  THIS," "show this latent edit projected into UMAP-space."

**Semantic-axis layer on top (optional, later):**

Once the raw editor exists, common multi-channel selections naturally get
saved as named macros ("low end punch", "transient kill", "noise floor"
removal). Those macros become *discovered* semantic axes — more honest than
anything a disentanglement probe would surface, because they came from
actual use. PCA / a small probe to suggest starting macros is a follow-up,
not a prerequisite. The raw editor stands on its own.

---

## Misc smaller things

- **Provenance for LoRAs** — auto-write the sidecar JSON on train completion
  even before the browser exists. Just so the data is there when we need it.
- **Hot model switching** — port mateo's pattern (`/api/switch-model` with
  gen_lock + 409-on-busy). Our current picker may already do this; diff and
  decide.
- **Audio library** (mateo's `LibraryExplorer.svelte`) — save generations to
  a local folder with timestamps + recall. Skipped on the first pass because
  the UI doesn't match our aesthetic, but the *backend* (`/api/library`,
  `unique_library_path`, etc.) is solid and worth porting standalone.
- **Stretching** — time-stretch / pitch-shift selections in the inpainter.
  Mentioned as a future feature, no design yet.
- **Per-region prompts** — paint a region, give it its own prompt; multiple
  regions with different prompts in one generation pass.
- **Streaming per-step diffusion previews** — show intermediate decode every
  N steps during generation. Lots of throw-away work but fun.
