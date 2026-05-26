"""SA3 Inpainter backend. FastAPI on :5174.

Loads the SA3 medium model once at startup (~30s), exposes JSON API for the
Svelte frontend.

Dual backend: auto-detects CUDA (torch AE) vs MPS (MLX AE).
Set SA3_BACKEND=cuda or SA3_BACKEND=mlx to force.
"""
import asyncio
import gc
import shutil
import os, sys, json, time, threading, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# AMD/ROCm tuning — must be set before `import torch`. TunableOp auto-tunes
# GEMM kernel selection on first use and caches to disk; the BLAS preference
# picks hipBLASLt for fused fast paths. Both are no-ops on CUDA/MPS, so it's
# safe to leave on everywhere. Ported from mateo19182/sa3-inpainter-ui.
os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "1")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")

import numpy as np
import torch
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from stable_audio_3.factory import create_diffusion_cond_from_config
from stable_audio_3 import StableAudioModel
from stable_audio_3.inference.distribution_shift import (
    IdentityDistributionShift, FluxDistributionShift, DistributionShift, LogSNRShift
)
from safetensors.torch import load_file

# -------- backend detection --------

def _detect_backend():
    forced = os.environ.get("SA3_BACKEND", "").lower()
    if forced == "cuda":
        return "cuda", "cuda"
    if forced == "mlx":
        return "mlx", "mps"
    if torch.cuda.is_available():
        return "cuda", "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mlx", "mps"
    return "cuda", "cpu"

BACKEND, DEVICE = _detect_backend()
HAS_MLX = BACKEND == "mlx"

if HAS_MLX:
    import mlx.core as mx
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mlx_sa3.ae import SA3MediumAE, decode_chunked
    from mlx_sa3.weights import load_ae_weights

# -------- settings persistence --------

_SETTINGS_FILE = Path(os.environ.get(
    "SA3_SETTINGS_FILE",
    str(Path.home() / ".config" / "sa3-inpainter" / "settings.json"),
))

_DEFAULT_CAPTIONER_EXAMPLES = "\n".join([
    "soundscape with piano arps, indie pop with male vocals, offbeat bassline with house drums and ambient electric guitar",
    "bright synth leads, pitched vocal chops, disco house, euphoric and danceable",
    "ethereal electric piano intro, fast 808 melody, experimental drill, plucked guitar outro",
])

def _autodetect_sa3_root() -> str:
    """Look for a stable-audio-3 checkout in a few common locations."""
    candidates = [
        Path.home() / "Projects" / "stable-audio-3",
        Path.home() / "stable-audio-3",
        Path.home() / "code"     / "stable-audio-3",
        Path.home() / "repos"    / "stable-audio-3",
        Path.home() / "src"      / "stable-audio-3",
        Path.home() / "Documents" / "stable-audio-3",
    ]
    for p in candidates:
        if (p / "stable_audio_3").is_dir() or (p / "pyproject.toml").exists() or (p / "setup.py").exists():
            return str(p)
    return ""

_SETTINGS_DEFAULTS = {
    "models_dir": str(Path.home() / "sa3-inpainter" / "models"),
    "lora_dir": str(Path.home() / "sa3-inpainter" / "loras"),
    "lora_train_dir": str(Path.home() / "sa3-inpainter" / "lora_training"),
    "sa3_root": _autodetect_sa3_root(),
    "hf_token": "",
    "openrouter_api_key": "",
    "captioner_model": "pro",
    "captioner_parallel": 32,
    "captioner_examples": _DEFAULT_CAPTIONER_EXAMPLES,
    "lora_adapter": "dora-rows",
}

def _load_settings():
    if _SETTINGS_FILE.exists():
        try:
            saved = json.loads(_SETTINGS_FILE.read_text())
            merged = dict(_SETTINGS_DEFAULTS)
            merged.update({k: v for k, v in saved.items() if k in _SETTINGS_DEFAULTS})
            return merged
        except Exception:
            pass
    return dict(_SETTINGS_DEFAULTS)

def _save_settings(s):
    _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_FILE.write_text(json.dumps(s, indent=2))

_settings = _load_settings()

def _apply_settings():
    global MODELS_DIR, LOCAL_MEDIUM, LORA_DIR, LORA_TRAIN_DIR
    MODELS_DIR = Path(_settings["models_dir"])
    LOCAL_MEDIUM = str(MODELS_DIR / "stable-audio-3-medium")
    LORA_DIR = Path(_settings["lora_dir"])
    LORA_TRAIN_DIR = Path(_settings["lora_train_dir"])

MODELS_DIR = Path(os.environ.get("SA3_MODELS_DIR", _settings["models_dir"]))
LOCAL_MEDIUM = os.environ.get("SA3_MODEL_DIR", str(MODELS_DIR / "stable-audio-3-medium"))
LORA_DIR = Path(os.environ.get("SA3_LORA_DIR", _settings["lora_dir"]))
LORA_TRAIN_DIR = Path(os.environ.get("SA3_LORA_TRAIN_DIR", _settings["lora_train_dir"]))
DATA_DIR = Path("/tmp/sa3-inpainter"); DATA_DIR.mkdir(exist_ok=True)
SR = 44100
DOWNSAMPLE = 4096
BANDS = [(0, 250), (250, 2500), (2500, 22050)]

# map model names → HF repo IDs for download-once caching
_MODEL_REPOS = {
    "medium":       "stabilityai/stable-audio-3-medium",
    "medium-base":  "stabilityai/stable-audio-3-medium-base",
    "small-music":  "stabilityai/stable-audio-3-small-music",
    "small-sfx":    "stabilityai/stable-audio-3-small-sfx",
}


def _resolve_local_path(name):
    """Return a local directory for a model, downloading from HF once if needed."""
    from huggingface_hub import snapshot_download
    local_dir = MODELS_DIR / f"stable-audio-3-{name}"
    if (local_dir / "model.safetensors").exists() and (local_dir / "model_config.json").exists():
        print(f"[backend] {name} found at {local_dir}")
        return str(local_dir)
    repo_id = _MODEL_REPOS.get(name)
    if not repo_id:
        return None
    print(f"[backend] downloading {name} from {repo_id} → {local_dir} (one-time)")
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    print(f"[backend] download complete: {local_dir}")
    return str(local_dir)


# -------- model state --------

sa = None               # StableAudioModel instance
mlx_ae = None           # MLX AE, only for medium on MPS
_use_mlx_ae = False     # whether current model uses MLX AE decode path
_current_model = None   # name of loaded model
_use_fp16 = os.environ.get("SA3_FP16", "0") == "1"
_cancel_event = threading.Event()
_loaded_lora_name = None
_default_memory_tokens = None   # snapshot of original memory tokens

# Conditioning cache — skips T5/Gemma re-encoding (~1-2s) when the same prompt
# is used across multiple inpaint / vary calls. Ported from mateo19182's fork.
# Cleared on model swap (different conditioner) and LoRA changes (different
# conditioner weights when conditioner LoRA is loaded).
_cond_cache: dict = {}
_COND_CACHE_MAX = 8

def _clear_cond_cache():
    global _cond_cache
    _cond_cache = {}

def _get_conditioning(prompt: str, neg_prompt: str | None, duration: float):
    """Return (pos_copy, neg_copy_or_None, was_cached). Shallow-copies before
    returning so generate() can mutate the dicts (it adds inpaint_mask /
    inpaint_masked_input) without corrupting the cached originals."""
    if sa is None:
        return None, None, False
    key = (prompt, neg_prompt or "", round(duration, 2))
    was_cached = key in _cond_cache
    if not was_cached:
        pos_cond, neg_cond = sa._build_conditioning_dicts(prompt, neg_prompt, duration, 1)
        pos_tensors = sa.model.conditioner(pos_cond, DEVICE)
        neg_tensors = sa.model.conditioner(neg_cond, DEVICE) if neg_cond is not None else None
        if len(_cond_cache) >= _COND_CACHE_MAX:
            _cond_cache.pop(next(iter(_cond_cache)))
        _cond_cache[key] = (pos_tensors, neg_tensors)
    pos, neg = _cond_cache[key]
    return dict(pos), (dict(neg) if neg is not None else None), was_cached
_memory_token_strength = 1.0    # user-controllable scale factor
_training_unloaded_model = None # model name to reload after training


def _load_model(name, local_path=None):
    """Load a model by name. Resolves to a local path, downloading once if needed."""
    global sa, mlx_ae, _use_mlx_ae, _current_model, _loaded_lora_name

    # cleanup old model
    sa = None
    mlx_ae = None
    _use_mlx_ae = False
    _loaded_lora_name = None
    _clear_cond_cache()
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    if not local_path:
        local_path = _resolve_local_path(name)

    print(f"[backend] loading {name} on {DEVICE}...")
    want_half = _use_fp16 and DEVICE == "cuda"

    if local_path:
        cfg_path = f"{local_path}/model_config.json"
        ckpt_path = f"{local_path}/model.safetensors"
        cfg = json.load(open(cfg_path))
        for c in cfg["model"]["conditioning"]["configs"]:
            if c["type"] == "t5gemma":
                c["config"]["repo_id"] = local_path
        model = create_diffusion_cond_from_config(cfg)
        model.load_state_dict(load_file(ckpt_path), strict=False)
        model.eval().requires_grad_(False).to(DEVICE)
        if want_half:
            model.half()
        sa = StableAudioModel(model, cfg, device=DEVICE, model_half=want_half)

        if HAS_MLX and name.startswith("medium"):
            print("[backend] loading MLX AE...")
            mlx_ae = SA3MediumAE()
            load_ae_weights(mlx_ae, ckpt_path)
            _use_mlx_ae = True
            print("[backend] MLX AE loaded")
    else:
        sa = StableAudioModel.from_pretrained(name, device=DEVICE, model_half=want_half)

    _current_model = name

    # snapshot default memory tokens for strength scaling
    global _default_memory_tokens, _memory_token_strength
    _memory_token_strength = 1.0
    try:
        mt = sa.model.model.transformer.memory_tokens
        _default_memory_tokens = mt.data.clone()
        print(f"[backend] memory tokens: {mt.shape}")
    except AttributeError:
        _default_memory_tokens = None

    print(f"[backend] {name} ready (mlx_ae={'yes' if _use_mlx_ae else 'no'})")


def _unload_model():
    global sa, mlx_ae, _use_mlx_ae, _loaded_lora_name, _default_memory_tokens
    sa = None
    mlx_ae = None
    _use_mlx_ae = False
    _loaded_lora_name = None
    _default_memory_tokens = None
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print("[backend] model unloaded")


# initial load
_load_model("medium", local_path=LOCAL_MEDIUM)
if _use_fp16 and DEVICE == "cuda":
    print("[backend] fp16 enabled")

# register RES4LYF exponential RK samplers
_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
from res4lyf.sampler import register_samplers, SAMPLER_NAMES as RES4LYF_NAMES
register_samplers()

from kv_cache import enable_kv_cache, disable_kv_cache, clear_kv_cache
from tome import apply_tome, remove_tome


# -------- decode helpers --------

_decode_fp32 = True
_decode_overlap = 32

def _decode_latents(lat_np):
    """Decode latent numpy array → waveform numpy array.
    lat_np: (1, 256, T_lat) float32
    Returns: (channels, T_audio) float32 numpy
    """
    if _use_mlx_ae and mlx_ae is not None:
        lat_mx = mx.array(lat_np)
        if lat_np.shape[-1] > 128:
            wav_m = decode_chunked(mlx_ae, lat_mx, chunk_size=128, overlap=_decode_overlap)
        else:
            wav_m = mlx_ae.decode(lat_mx)
        mx.eval(wav_m)
        return np.array(wav_m)[0]
    else:
        lat_t = torch.from_numpy(lat_np).to(DEVICE)
        use_fp32 = _decode_fp32 and DEVICE == "cuda" and _use_fp16
        with torch.inference_mode():
            if use_fp32:
                sa.same.float()
                lat_t = lat_t.float()
                wav_t = sa.same.decode(lat_t, chunked=True, overlap=_decode_overlap, chunk_size=128)
                sa.same.half()
            else:
                if _use_fp16 and DEVICE == "cuda":
                    lat_t = lat_t.half()
                wav_t = sa.same.decode(lat_t, chunked=True, overlap=_decode_overlap, chunk_size=128)
        return wav_t.float().cpu().numpy()[0]


def render_noise_spec_once():
    out_path = DATA_DIR / "noise_spec.png"
    if out_path.exists(): return
    T_lat = int(30 * SR / DOWNSAMPLE) + 1
    rng = np.random.default_rng(7)
    lat = rng.standard_normal((1, 256, T_lat)).astype(np.float32) * 0.3
    wav_np = _decode_latents(lat)
    render_spec_png(wav_np, out_path)

state = {"audio_path": None, "version": 0, "bpm": None}

# -------- audio history (undo/redo) --------

HISTORY_DIR = DATA_DIR / "history"; HISTORY_DIR.mkdir(exist_ok=True)
_audio_undo = []   # stack of (wav_path, bpm)
_audio_redo = []   # stack of (wav_path, bpm)
MAX_HISTORY = 30


def _snapshot_current():
    """Save the current audio as a history entry. Call BEFORE replacing it."""
    if state["audio_path"] is None:
        return
    idx = len(_audio_undo)
    dst = HISTORY_DIR / f"snap_{idx}.wav"
    shutil.copy2(state["audio_path"], dst)
    _audio_undo.append((str(dst), state.get("bpm")))
    if len(_audio_undo) > MAX_HISTORY:
        old_path, _ = _audio_undo.pop(0)
        Path(old_path).unlink(missing_ok=True)
    _audio_redo.clear()


def _restore_snapshot(entry):
    """Restore audio from a history entry. Returns envelope info."""
    wav_path, bpm = entry
    audio, _ = sf.read(wav_path)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)
    env = persist_audio(audio.T)
    state["bpm"] = bpm
    return env


app = FastAPI()


def compute_envelope(audio_np):
    mono = audio_np.mean(axis=0) if audio_np.ndim == 2 else audio_np
    N = len(mono) // DOWNSAMPLE
    freqs = np.fft.rfftfreq(DOWNSAMPLE, 1.0 / SR)
    masks = [(freqs >= lo) & (freqs < hi) for lo, hi in BANDS]
    data = []
    for i in range(N):
        seg = mono[i*DOWNSAMPLE:(i+1)*DOWNSAMPLE]
        peak = float(np.abs(seg).max())
        spec = np.abs(np.fft.rfft(seg)) ** 2
        e = [float(spec[m].sum()) for m in masks]
        total = sum(e) + 1e-12
        rgb = [(v/total) ** 0.6 for v in e]
        mx_ = max(rgb) + 1e-12
        rgb = [v/mx_ for v in rgb]
        data.append([round(peak, 4)] + [round(c, 3) for c in rgb])
    return {"sr": SR, "downsample": DOWNSAMPLE, "count": N, "data": data}


def render_spec_png(audio_np, out_path):
    mono = audio_np.mean(axis=0) if audio_np.ndim == 2 else audio_np
    n_fft = 8192
    hop = 1024   # 4x finer time resolution than DOWNSAMPLE — better at deep zoom
    f, t, Z = stft(mono, fs=SR, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    P_db = 10.0 * np.log10(P + 1e-12)
    P_db = np.clip(P_db, -55, P_db.max()); P_db -= P_db.min()
    if P_db.max() < 1e-6:
        P_db = np.zeros_like(P_db)
    else:
        P_db /= P_db.max()
        P_db = P_db ** 0.55
    out_h = 600
    log_f = np.geomspace(30, 16000, out_h)
    spec_log = np.zeros((out_h, P_db.shape[1]), dtype=np.float32)
    for j in range(P_db.shape[1]):
        spec_log[:, j] = np.interp(log_f, f, P_db[:, j])
    fig = plt.figure(figsize=(40, 6), dpi=100)   # 4000x600 — room for the finer hop
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0,0,1,1]); ax.set_axis_off()
    ax.imshow(spec_log[::-1], aspect="auto", origin="upper", cmap="magma", interpolation="nearest", extent=(0,1,0,1))
    fig.savefig(out_path, dpi=100, facecolor="black")
    plt.close(fig)


def render_overview_png(audio_np, out_path, W=2000, H=80):
    mono = audio_np.mean(axis=0) if audio_np.ndim == 2 else audio_np
    bin_sz = max(1, len(mono) // W)
    peaks = np.zeros(W)
    for i in range(W):
        s = i * bin_sz
        peaks[i] = np.max(np.abs(mono[s:s+bin_sz])) if s < len(mono) else 0
    # peaks stay in true amplitude [0, 1] — do not normalize, or quiet songs
    # render as sausage and loud songs lose dynamics
    peaks = np.clip(peaks, 0.0, 1.0)
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    fig.patch.set_facecolor("#000000")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, W); ax.set_ylim(-1.05, 1.05)
    ax.vlines(np.arange(W), -peaks, peaks, color="#666666", linewidth=0.7)
    fig.savefig(out_path, dpi=100, facecolor="#000000")
    plt.close(fig)


def persist_audio(audio_np):
    """audio_np: (2, T) float in [-1, 1]."""
    p = DATA_DIR / "current.wav"
    sf.write(p, audio_np.T, SR)
    state["audio_path"] = str(p)
    state["version"] += 1
    env = compute_envelope(audio_np)
    with open(DATA_DIR / "envelope.json", "w") as fh:
        json.dump(env, fh)
    threading.Thread(target=render_spec_png, args=(audio_np, DATA_DIR / "current_spec.png"), daemon=True).start()
    threading.Thread(target=render_overview_png, args=(audio_np, DATA_DIR / "current_overview.png"), daemon=True).start()
    return env


# -------- LoRA helpers --------

def _apply_loras(loras: list[dict]) -> None:
    global _loaded_lora_name
    if not loras:
        return
    for entry in loras:
        name = entry["name"]
        strength = float(entry.get("strength", 1.0))
        lora_path = LORA_DIR / name
        if not lora_path.exists():
            print(f"[lora] not found: {lora_path}, skipping")
            continue
        if _loaded_lora_name != name:
            sa.load_lora([str(lora_path)])
            _loaded_lora_name = name
            # Conditioner-side LoRA may alter T5/Gemma — drop any cached
            # encodings so the next call recomputes them.
            _clear_cond_cache()
            print(f"[lora] loaded {name}")
        sa.set_lora_strength(strength)
        print(f"[lora] strength {name} @ {strength}")
        return


def _unload_loras(loras: list[dict]) -> None:
    if _loaded_lora_name is None:
        return
    sa.set_lora_strength(0.0)
    print(f"[lora] deactivated {_loaded_lora_name}")


# -------- endpoints --------

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    _snapshot_current()
    raw = DATA_DIR / ("upload_raw" + Path(file.filename or ".wav").suffix)
    with open(raw, "wb") as f: f.write(await file.read())
    tag_bpm = read_bpm_tag(str(raw))
    audio, sr = sf.read(raw)
    if audio.ndim == 1: audio = np.stack([audio, audio], axis=-1)
    if sr != SR:
        import torchaudio
        a = torch.from_numpy(audio.T).float()
        a = torchaudio.transforms.Resample(sr, SR)(a)
        audio = a.numpy().T
    env = persist_audio(audio.T)
    bpm_source = "tag" if tag_bpm else "detect"
    bpm = tag_bpm if tag_bpm else detect_bpm(audio.T, SR)
    print(f"[bpm] {bpm_source}: {bpm}")
    state["bpm"] = bpm
    return {"version": state["version"], "count": env["count"],
            "duration": env["count"] * DOWNSAMPLE / SR, "bpm": bpm,
            "bpm_source": bpm_source}


@app.post("/api/detect_bpm")
async def redetect_bpm():
    if state["audio_path"] is None:
        raise HTTPException(400, "no audio loaded")
    audio, _ = sf.read(state["audio_path"])
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)
    bpm = detect_bpm(audio.T, SR)
    state["bpm"] = bpm
    print(f"[bpm] re-detect: {bpm}")
    return {"bpm": bpm, "bpm_source": "detect"}


class LoraEntry(BaseModel):
    name: str
    strength: float = 1.0


# ---- Prompt enhancer (SA3's bundled Qwen3.5-2B reprompter) ----
# Loaded lazily on first call — ~4.2GB download then BF16 in unified memory.

_ENHANCER_MODEL_ID = "Qwen/Qwen3.5-2B"

class EnhancePromptBody(BaseModel):
    prompt: str = ""
    preset: str = "Auto"           # "Auto" | "Music" | "Instrument" | "SFX" | "Classifier"
    max_new_tokens: int = 128
    temperature: float = 1.11

@app.post("/api/enhance_prompt")
async def enhance_prompt(body: EnhancePromptBody):
    """Wraps stable_audio_3.interface.reprompt.reprompt(...).
    Empty input returns a random example from the Music system prompt."""
    try:
        from stable_audio_3.interface.reprompt import reprompt as _reprompt_fn
    except Exception as e:
        raise HTTPException(500, f"prompt enhancer unavailable: {e}")
    def _run():
        return _reprompt_fn(
            body.prompt, body.preset, "",
            _ENHANCER_MODEL_ID, body.max_new_tokens, body.temperature,
        )
    try:
        raw, processed, category = await asyncio.to_thread(_run)
    except Exception as e:
        raise HTTPException(500, f"enhance failed: {e}")
    return {"prompt": processed or raw or body.prompt, "raw": raw, "category": category}


class GenBody(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    mask: list[int] = []
    settings: dict = {}
    loras: list[LoraEntry] = []


@app.post("/api/cancel")
async def cancel():
    _cancel_event.set()
    return {"status": "cancelling"}


@app.post("/api/undo")
async def undo_audio():
    if not _audio_undo:
        raise HTTPException(400, "nothing to undo")
    if state["audio_path"]:
        idx = len(_audio_undo) + len(_audio_redo)
        dst = HISTORY_DIR / f"snap_{idx}.wav"
        shutil.copy2(state["audio_path"], dst)
        _audio_redo.append((str(dst), state.get("bpm")))
    entry = _audio_undo.pop()
    env = _restore_snapshot(entry)
    return {"version": state["version"], "count": env["count"],
            "duration": env["count"] * DOWNSAMPLE / SR, "bpm": state["bpm"],
            "can_undo": len(_audio_undo) > 0, "can_redo": len(_audio_redo) > 0}


@app.post("/api/redo")
async def redo_audio():
    if not _audio_redo:
        raise HTTPException(400, "nothing to redo")
    if state["audio_path"]:
        idx = len(_audio_undo) + len(_audio_redo)
        dst = HISTORY_DIR / f"snap_{idx}.wav"
        shutil.copy2(state["audio_path"], dst)
        _audio_undo.append((str(dst), state.get("bpm")))
    entry = _audio_redo.pop()
    env = _restore_snapshot(entry)
    return {"version": state["version"], "count": env["count"],
            "duration": env["count"] * DOWNSAMPLE / SR, "bpm": state["bpm"],
            "can_undo": len(_audio_undo) > 0, "can_redo": len(_audio_redo) > 0}


@app.post("/api/generate")
async def generate(body: GenBody):
    _snapshot_current()
    _cancel_event.clear()

    s = body.settings
    steps = int(s.get("steps", 8))
    cfg = float(s.get("cfg", 1.0))
    seed = int(s.get("seed", 42))
    if seed == -1:
        seed = int(np.random.randint(0, 999999))
    noise = float(s.get("noise", 1.0))
    duration = float(s.get("duration", 30.0))
    sampler_type = s.get("sampler_type")
    apg_scale = float(s.get("apg_scale", 1.0))

    # advanced params
    scale_phi = float(s.get("scale_phi", 0.0))
    cfg_interval = s.get("cfg_interval", [0.0, 1.0])
    cfg_norm_threshold = float(s.get("cfg_norm_threshold", 0.0))
    exit_layer_ix_raw = s.get("exit_layer_ix")
    exit_layer_ix = int(exit_layer_ix_raw) if exit_layer_ix_raw else None
    duration_padding_sec = float(s.get("duration_padding_sec", 6.0))

    # inference speedup params
    kv_cache_enabled = bool(s.get("kvCache", False))
    tome_ratio = float(s.get("tomeRatio", 0.0))

    # dist_shift construction
    dist_shift_type = s.get("dist_shift_type", "default")
    dist_shift = None
    if dist_shift_type == "none":
        dist_shift = IdentityDistributionShift()
    elif dist_shift_type == "logsnr":
        dist_shift = LogSNRShift(
            anchor_logsnr=float(s.get("dist_shift_anchor_logsnr", -6.2)),
            rate=float(s.get("dist_shift_rate", 0.0)),
            logsnr_end=float(s.get("dist_shift_logsnr_end", 2.0)),
        )
    elif dist_shift_type == "flux":
        dist_shift = FluxDistributionShift(
            alpha_min=float(s.get("dist_shift_alpha_min", 1.0)),
            alpha_max=float(s.get("dist_shift_alpha_max", 1.0)),
        )
    elif dist_shift_type == "full":
        dist_shift = DistributionShift(
            base_shift=float(s.get("dist_shift_base_shift", 0.5)),
            max_shift=float(s.get("dist_shift_max_shift", 1.15)),
        )

    has_source = state["audio_path"] is not None
    has_mask = any(body.mask) if body.mask else False
    n_regen = sum(body.mask) if body.mask else 0
    print(f"[generate] source={has_source} mask_len={len(body.mask) if body.mask else 0} regen_latents={n_regen} mode={('inpaint' if has_source and has_mask else 'vary' if has_source else 't2a')}")

    neg_prompt = body.negative_prompt or None
    # Conditioning cache: skip T5/Gemma re-encoding (~1-2s) when the same prompt
    # repeats. Cache key is (prompt, neg, file_duration), so multiple inpaints
    # against the same track hit the cache. Falls back to None tensors if
    # encoding fails for any reason — sa.generate() will then re-encode itself.
    eff_duration = duration
    if has_source and state["audio_path"]:
        try:
            info = sf.info(state["audio_path"])
            eff_duration = info.frames / info.samplerate
        except Exception:
            pass
    t_cond = time.time()
    try:
        pos_cond, neg_cond, was_cached = _get_conditioning(body.prompt, neg_prompt, eff_duration)
        print(f"[backend] conditioning {'(cached)' if was_cached else '(encoded)'} {time.time()-t_cond:.2f}s")
    except Exception as e:
        print(f"[backend] conditioning cache miss → falling back to prompt path: {e}")
        pos_cond, neg_cond = None, None
    kwargs = dict(steps=steps, cfg_scale=cfg, seed=seed, apg_scale=apg_scale,
                  duration_padding_sec=duration_padding_sec,
                  return_latents=True,
                  chunked_decode=False)
    if pos_cond is not None:
        kwargs["conditioning_tensors"] = pos_cond
        if neg_cond is not None:
            kwargs["negative_conditioning_tensors"] = neg_cond
    else:
        kwargs["prompt"] = body.prompt
        kwargs["negative_prompt"] = neg_prompt
    if sampler_type:
        kwargs["sampler_type"] = sampler_type
    if dist_shift is not None:
        kwargs["dist_shift"] = dist_shift
    if scale_phi != 0.0:
        kwargs["scale_phi"] = scale_phi
    if cfg_interval != [0.0, 1.0]:
        kwargs["cfg_interval"] = tuple(cfg_interval)
    if cfg_norm_threshold > 0:
        kwargs["cfg_norm_threshold"] = cfg_norm_threshold
    if exit_layer_ix is not None:
        kwargs["exit_layer_ix"] = exit_layer_ix

    audio_mask = None
    if has_source and has_mask:
        audio, _ = sf.read(state["audio_path"])
        audio_t = torch.from_numpy(audio.T).float().to(DEVICE)
        actual_lat = audio.shape[0] // DOWNSAMPLE
        mask_lat = np.asarray(body.mask, dtype=np.float32)
        if len(mask_lat) > actual_lat:
            mask_lat = mask_lat[:actual_lat]
        elif len(mask_lat) < actual_lat:
            mask_lat = np.pad(mask_lat, (0, actual_lat - len(mask_lat)), constant_values=0)
        inv = 1.0 - mask_lat
        audio_mask = np.repeat(inv, DOWNSAMPLE)
        audio_mask = audio_mask[:audio.shape[0]]
        print(f"[inpaint] mask aligned: {len(mask_lat)} latents, {int(mask_lat.sum())} regen, {audio.shape[0]} samples")
        kwargs["duration"] = audio.shape[0] / SR
        kwargs["sample_size"] = audio.shape[0]
        kwargs["inpaint_audio"] = (SR, audio_t)
        kwargs["inpaint_mask"] = torch.from_numpy(audio_mask).unsqueeze(0).to(DEVICE)
    elif has_source:
        audio, _ = sf.read(state["audio_path"])
        audio_t = torch.from_numpy(audio.T).float().to(DEVICE)
        kwargs["duration"] = audio.shape[0] / SR
        kwargs["sample_size"] = audio.shape[0]
        kwargs["init_audio"] = (SR, audio_t)
        kwargs["init_noise_level"] = noise
    else:
        kwargs["duration"] = duration
        kwargs["sample_size"] = int(duration * SR)

    loras_list = [l.model_dump() for l in body.loras]

    def _run_generate():
        _apply_loras(loras_list)
        _dit = sa.model.model
        try:
            if _cancel_event.is_set():
                return None

            # Apply inference speedups
            if kv_cache_enabled:
                enable_kv_cache(_dit)
            if tome_ratio > 0.0:
                apply_tome(_dit, ratio=tome_ratio)

            t0 = time.time()
            result = sa.generate(**kwargs)

            lat_np = result.detach().to(torch.float32).cpu().numpy()
            print(f"[backend] DIT {time.time()-t0:.1f}s, latents shape {lat_np.shape}")
            t1 = time.time()
            wav_np = _decode_latents(lat_np)
            print(f"[backend] AE {time.time()-t1:.1f}s, decode_fp32={_decode_fp32} overlap={_decode_overlap}")

            if _cancel_event.is_set():
                return None

            target_dur = float(kwargs.get("duration", duration))
            max_samples = int(target_dur * SR)
            print(f"[truncate] mode={'inpaint' if (has_source and has_mask) else 'vary' if has_source else 't2a'} target_dur={target_dur:.2f}s max_samples={max_samples} wav_len={wav_np.shape[-1]}")
            if wav_np.shape[-1] > max_samples:
                wav_np = wav_np[:, :max_samples]

            if has_source and has_mask and audio_mask is not None:
                orig, _ = sf.read(state["audio_path"])
                orig_t = orig.T
                if orig_t.ndim == 1:
                    orig_t = np.stack([orig_t, orig_t], axis=0)
                T = min(orig_t.shape[-1], wav_np.shape[-1], len(audio_mask))
                m = audio_mask[:T].astype(np.float32)
                m2 = np.stack([m, m], axis=0)
                wav_np = wav_np[:, :T]
                orig_t = orig_t[:, :T]
                XF = 256
                m_eased = m2.copy()
                edges = np.where(np.abs(np.diff(m)) > 0)[0]
                for e in edges:
                    lo = max(0, e - XF // 2)
                    hi = min(T, e + XF // 2)
                    w = np.linspace(0, 1, hi - lo)
                    if (m[e] > m[e + 1]) if (e + 1 < T) else False:
                        m_eased[:, lo:hi] = np.minimum(m_eased[:, lo:hi], 1 - w)
                    else:
                        m_eased[:, lo:hi] = np.maximum(m_eased[:, lo:hi], w)
                wav_np = m_eased * orig_t + (1.0 - m_eased) * wav_np

            return wav_np
        finally:
            # Clean up speedup patches
            clear_kv_cache(_dit)
            disable_kv_cache(_dit)
            remove_tome(_dit)
            _unload_loras(loras_list)

    wav_np = await asyncio.to_thread(_run_generate)

    if wav_np is None:
        raise HTTPException(status_code=499, detail="generation cancelled")

    env = persist_audio(wav_np)
    bpm = detect_bpm(wav_np, SR)
    state["bpm"] = bpm
    return {"version": state["version"], "count": env["count"],
            "duration": env["count"] * DOWNSAMPLE / SR, "bpm": bpm,
            "seed": seed}


def read_bpm_tag(file_path):
    """Read BPM from audio file metadata (ID3 TBPM, Vorbis, etc). Returns float or None."""
    try:
        import mutagen
        f = mutagen.File(file_path, easy=True)
        if f is None:
            return None
        bpm_str = None
        if hasattr(f, "get"):
            for key in ("bpm", "TBPM", "tempo"):
                val = f.get(key)
                if val:
                    bpm_str = val[0] if isinstance(val, list) else val
                    break
        if not bpm_str:
            f2 = mutagen.File(file_path)
            if f2 and hasattr(f2, "tags") and f2.tags:
                for key in ("TBPM", "TXXX:BPM", "TXXX:bpm"):
                    frame = f2.tags.get(key)
                    if frame:
                        bpm_str = str(frame.text[0]) if hasattr(frame, "text") else str(frame)
                        break
        if bpm_str:
            bpm = float(bpm_str)
            if 20 < bpm < 400:
                return round(bpm, 1)
    except Exception as e:
        print(f"[bpm] tag read failed: {e}")
    return None


def detect_bpm(audio_np, sr=44100):
    """Detect BPM via onset strength autocorrelation. audio_np: (channels, T) or (T,)."""
    from scipy.signal import find_peaks
    mono = audio_np.mean(axis=0) if audio_np.ndim == 2 else audio_np
    # compute spectral flux onset strength
    hop = 512
    n_fft = 2048
    n_frames = (len(mono) - n_fft) // hop + 1
    if n_frames < 16:
        return 120.0
    onset = np.zeros(n_frames)
    prev_spec = np.zeros(n_fft // 2 + 1)
    for i in range(n_frames):
        frame = mono[i * hop : i * hop + n_fft] * np.hanning(n_fft)
        spec = np.abs(np.fft.rfft(frame))
        diff = spec - prev_spec
        onset[i] = np.sum(np.maximum(0, diff))
        prev_spec = spec
    # autocorrelation in BPM range 40-220
    min_lag = int(60.0 / 220 * sr / hop)
    max_lag = int(60.0 / 40 * sr / hop)
    max_lag = min(max_lag, len(onset) // 2)
    if min_lag >= max_lag:
        return 120.0
    onset_norm = onset - onset.mean()
    corr = np.correlate(onset_norm, onset_norm, mode='full')
    corr = corr[len(onset_norm) - 1:]  # positive lags only
    corr_range = corr[min_lag:max_lag]
    if len(corr_range) == 0:
        return 120.0
    peaks, props = find_peaks(corr_range, distance=int(0.3 * sr / hop))
    if len(peaks) == 0:
        best_lag = min_lag + np.argmax(corr_range)
    else:
        best_idx = np.argmax(corr_range[peaks])
        best_lag = min_lag + peaks[best_idx]
    bpm = 60.0 * sr / hop / best_lag
    # snap to reasonable range, handle octave errors
    if bpm > 180:
        bpm /= 2
    elif bpm < 60:
        bpm *= 2
    return round(bpm, 1)


class TempoBody(BaseModel):
    factor: float = 1.0  # >1 = faster, <1 = slower
    target_bpm: float | None = None


@app.post("/api/tempo")
async def tempo_change(body: TempoBody):
    if state["audio_path"] is None:
        raise HTTPException(400, "no audio loaded")
    _snapshot_current()
    factor = body.factor
    if factor <= 0.1 or factor > 10.0:
        raise HTTPException(400, "factor must be in (0.1, 10.0]")
    if abs(factor - 1.0) < 0.001:
        return {"version": state["version"], "count": 0, "duration": 0}

    src = state["audio_path"]
    dst = str(DATA_DIR / "tempo_out.wav")
    proc = await asyncio.create_subprocess_exec(
        "rubberband", "--fine", "--tempo", str(factor), src, dst,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise HTTPException(500, f"rubberband failed: {stderr.decode()[:200]}")

    audio, sr_out = sf.read(dst)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)
    env = persist_audio(audio.T)
    if body.target_bpm and body.target_bpm > 0:
        bpm = round(body.target_bpm, 1)
    else:
        bpm = detect_bpm(audio.T, SR)
    state["bpm"] = bpm
    return {"version": state["version"], "count": env["count"],
            "duration": env["count"] * DOWNSAMPLE / SR, "bpm": bpm}


@app.post("/api/clear")
async def clear():
    state["audio_path"] = None
    state["version"] += 1
    return {"version": state["version"]}


@app.get("/api/state")
async def get_state():
    return {"has_audio": state["audio_path"] is not None, "version": state["version"],
            "model_loaded": sa is not None, "backend": BACKEND, "model": _current_model,
            "bpm": state.get("bpm")}


def _mask_secrets(safe: dict) -> dict:
    if safe.get("hf_token"):
        safe["hf_token"] = "hf_****" + safe["hf_token"][-4:]
    if safe.get("openrouter_api_key"):
        safe["openrouter_api_key"] = "sk-or-****" + safe["openrouter_api_key"][-4:]
    return safe

@app.get("/api/settings")
async def get_settings():
    safe = {**_settings, "first_run": not _SETTINGS_FILE.exists()}
    return _mask_secrets(safe)

class SettingsBody(BaseModel):
    models_dir: str | None = None
    lora_dir: str | None = None
    lora_train_dir: str | None = None
    sa3_root: str | None = None
    hf_token: str | None = None
    openrouter_api_key: str | None = None
    captioner_model: str | None = None
    captioner_parallel: int | None = None
    captioner_examples: str | None = None
    lora_adapter: str | None = None

@app.post("/api/settings")
async def update_settings(body: SettingsBody):
    changed = {k: v for k, v in body.model_dump().items() if v is not None}
    # Don't overwrite real secrets with their masked echo
    if changed.get("hf_token", "").startswith("hf_****"):
        del changed["hf_token"]
    if changed.get("openrouter_api_key", "").startswith("sk-or-****"):
        del changed["openrouter_api_key"]
    _settings.update(changed)
    _save_settings(_settings)
    _apply_settings()
    return _mask_secrets({**_settings})


import psutil
_proc = psutil.Process()
psutil.cpu_percent(interval=None)

@app.get("/api/speedups")
async def get_speedups():
    """Return available inference speedup options and their current state."""
    return {
        "kv_cache": {
            "available": True,
            "description": "Cache cross-attention K/V projections across diffusion steps",
            "param": "kvCache",
            "type": "bool",
        },
        "tome": {
            "available": True,
            "description": "Token Merging: reduce self-attention sequence length per block",
            "param": "tomeRatio",
            "type": "float",
            "range": [0.0, 0.5],
            "default": 0.0,
        },
    }


def _load_training_losses():
    """Parse loss from PL metrics CSVs in training dirs. Returns {step: loss}."""
    losses = {}
    if not LORA_TRAIN_DIR.exists():
        return losses
    import csv
    for train_dir in LORA_TRAIN_DIR.iterdir():
        if not train_dir.is_dir():
            continue
        logs_dir = train_dir / "checkpoints" / "lightning_logs"
        if not logs_dir.exists():
            continue
        for version_dir in sorted(logs_dir.iterdir(), reverse=True):
            csv_path = version_dir / "metrics.csv"
            if not csv_path.exists():
                continue
            try:
                with open(csv_path) as f:
                    for row in csv.DictReader(f):
                        step = int(row.get("step", -1))
                        loss = float(row.get("train/loss", 0))
                        losses[(train_dir.name, step)] = round(loss, 4)
            except Exception:
                pass
    return losses

@app.get("/api/loras")
async def list_loras():
    if not LORA_DIR.exists(): return {"dir": str(LORA_DIR), "files": []}
    import re
    losses = _load_training_losses()
    files = []
    for p in sorted(LORA_DIR.iterdir()):
        if not p.is_file() or p.suffix != ".safetensors":
            continue
        entry = {"name": p.name}
        m = re.match(r"(.+?)-step(\d+)\.safetensors$", p.name)
        if m:
            train_name, step = m.group(1), int(m.group(2))
            loss = losses.get((train_name, step - 1)) or losses.get((train_name, step))
            if loss is not None:
                entry["loss"] = loss
                entry["step"] = step
        files.append(entry)
    return {"dir": str(LORA_DIR), "files": files}


# -------- memory tokens --------

MEMTOK_DIR = Path(os.environ.get("SA3_MEMTOK_DIR", str(Path.home() / "models/sa3-memory-tokens")))


def _get_memory_tokens():
    """Get the live memory_tokens parameter, or None."""
    try:
        return sa.model.model.transformer.memory_tokens
    except AttributeError:
        return None


@app.get("/api/memory_tokens")
async def get_memory_tokens():
    mt = _get_memory_tokens()
    if mt is None:
        return {"available": False}
    presets = []
    if MEMTOK_DIR.exists():
        presets = sorted(p.stem for p in MEMTOK_DIR.glob("*.safetensors"))
    return {
        "available": True,
        "count": mt.shape[0],
        "dim": mt.shape[1],
        "strength": _memory_token_strength,
        "presets": presets,
    }


class MemtokBody(BaseModel):
    strength: float | None = None
    preset: str | None = None
    action: str = "set"   # "set" | "save" | "load" | "reset"
    name: str | None = None


@app.post("/api/memory_tokens")
async def set_memory_tokens(body: MemtokBody):
    global _memory_token_strength
    mt = _get_memory_tokens()
    if mt is None:
        raise HTTPException(400, "model has no memory tokens")

    if body.action == "reset":
        if _default_memory_tokens is not None:
            mt.data.copy_(_default_memory_tokens)
        _memory_token_strength = 1.0
        return {"strength": 1.0, "status": "reset"}

    if body.action == "save":
        name = body.name or "custom"
        MEMTOK_DIR.mkdir(parents=True, exist_ok=True)
        from safetensors.torch import save_file
        save_file({"memory_tokens": mt.data.clone().cpu()}, str(MEMTOK_DIR / f"{name}.safetensors"))
        print(f"[memtok] saved {name}")
        return {"status": "saved", "name": name}

    if body.action == "load":
        preset = body.preset or body.name
        if not preset:
            raise HTTPException(400, "preset name required")
        path = MEMTOK_DIR / f"{preset}.safetensors"
        if not path.exists():
            raise HTTPException(404, f"preset not found: {preset}")
        data = load_file(str(path))
        mt.data.copy_(data["memory_tokens"].to(mt.device, mt.dtype))
        _default_memory_tokens.copy_(mt.data)
        _memory_token_strength = 1.0
        print(f"[memtok] loaded {preset}")
        return {"status": "loaded", "preset": preset, "strength": 1.0}

    if body.strength is not None:
        s = max(0.0, min(3.0, body.strength))
        if _default_memory_tokens is not None:
            mt.data.copy_(_default_memory_tokens * s)
        _memory_token_strength = s
        return {"strength": s, "status": "ok"}

    return {"strength": _memory_token_strength, "status": "ok"}


@app.get("/api/browse_folder")
async def browse_folder(start: str = "~"):
    """Open a native folder picker dialog and return the selected path."""
    import subprocess
    start_path = str(Path(start).expanduser())
    try:
        result = subprocess.run(
            ["zenity", "--file-selection", "--directory", f"--filename={start_path}/"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and result.stdout.strip():
            return {"path": result.stdout.strip()}
        return {"path": None}
    except FileNotFoundError:
        try:
            result = subprocess.run(
                ["kdialog", "--getexistingdirectory", start_path],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0 and result.stdout.strip():
                return {"path": result.stdout.strip()}
            return {"path": None}
        except FileNotFoundError:
            raise HTTPException(status_code=501, detail="No dialog tool found (zenity or kdialog)")


# -------- LoRA data folder management --------

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}


def _lora_data_folder(name: str, folder: str | None) -> Path:
    """Resolve the active dataset folder for a LoRA. Override wins; otherwise
    the default convention is LORA_TRAIN_DIR/<name>/data."""
    if folder:
        return Path(folder).expanduser().resolve()
    if not name:
        raise HTTPException(400, "lora name required when no folder override is set")
    return (LORA_TRAIN_DIR / name / "data").resolve()


def _scan_lora_folder(d: Path) -> dict:
    files = []
    if d.is_dir():
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in _AUDIO_EXTS:
                txt = d / (p.stem + ".txt")
                cap_text = ""
                if txt.exists():
                    try:
                        # cap to a sane length so list responses stay small
                        cap_text = txt.read_text(encoding="utf-8")[:500]
                    except Exception:
                        cap_text = ""
                files.append({
                    "name": p.name,
                    "stem": p.stem,
                    "ext": p.suffix.lower().lstrip("."),
                    "captioned": txt.exists(),
                    "caption": cap_text,
                })
    return {
        "folder": str(d),
        "exists": d.is_dir(),
        "files": files,
        "total": len(files),
        "captioned": sum(1 for f in files if f["captioned"]),
    }


class LoraDataQuery(BaseModel):
    name: str = ""
    folder: str | None = None


@app.post("/api/lora_data/list")
async def lora_data_list(body: LoraDataQuery):
    if not body.name and not body.folder:
        return {"folder": "", "exists": False, "files": [], "total": 0, "captioned": 0}
    d = _lora_data_folder(body.name, body.folder)
    return _scan_lora_folder(d)


@app.post("/api/lora_data/upload")
async def lora_data_upload(
    name: str = Form(""),
    folder: str = Form(""),
    files: list[UploadFile] = File(...),
):
    d = _lora_data_folder(name, folder or None)
    d.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        # accept audio and paired .txt files
        ext = Path(f.filename).suffix.lower()
        if ext not in _AUDIO_EXTS and ext != ".txt":
            continue
        target = d / f.filename
        target.write_bytes(await f.read())
        saved.append(f.filename)
    return {"saved": saved, **_scan_lora_folder(d)}


def _render_audio_thumb(audio_path: Path, out_path: Path, size: int = 96):
    """Render a small square spectrogram thumbnail of an audio file.
    Uses PIL + numpy directly (no matplotlib) — matplotlib's figure/axes
    overhead alone is ~50-100ms per call, which dwarfs the actual STFT for
    tiny outputs. With PIL the whole thing is ~3-10ms."""
    import soundfile as sf
    from scipy.signal import stft as _stft
    from PIL import Image
    from matplotlib import cm  # only used for the colormap LUT — no figures
    audio, sr = sf.read(str(audio_path))
    if audio.ndim == 2: audio = audio.mean(axis=1)
    if len(audio) < 64: return False
    n_fft = 512
    hop = max(1, len(audio) // (size * 2))
    f, _t, Z = _stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    Pdb = 10.0 * np.log10(P + 1e-12)
    Pdb = np.clip(Pdb, -55, Pdb.max())
    if Pdb.max() < Pdb.min() + 1e-6:
        Pdb = np.zeros_like(Pdb)
    else:
        Pdb -= Pdb.min(); Pdb /= Pdb.max(); Pdb = Pdb ** 0.55
    # log-frequency squish into `size` rows (top = high freq)
    log_f = np.geomspace(30, min(16000, sr / 2), size)
    spec_log = np.empty((size, Pdb.shape[1]), dtype=np.float32)
    for j in range(Pdb.shape[1]):
        spec_log[:, j] = np.interp(log_f, f, Pdb[:, j])
    # resample columns to `size` so output is square
    if spec_log.shape[1] != size:
        idx = np.linspace(0, spec_log.shape[1] - 1, size)
        spec_log = spec_log[:, idx.astype(np.int32)]
    # flip so y=0 (top) is high frequency
    spec_log = spec_log[::-1]
    # apply magma colormap via PIL
    magma = cm.get_cmap("magma")
    rgba = (magma(np.clip(spec_log, 0, 1)) * 255).astype(np.uint8)
    img = Image.fromarray(rgba[:, :, :3], mode="RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG", optimize=False, compress_level=1)
    return True


@app.get("/api/lora_data/thumb")
async def lora_data_thumb(name: str = "", folder: str = "", file: str = ""):
    if not file:
        raise HTTPException(400, "file required")
    if not name and not folder:
        raise HTTPException(400, "name or folder required")
    d = _lora_data_folder(name, folder or None)
    src = d / file
    if not src.is_file() or src.suffix.lower() not in _AUDIO_EXTS:
        raise HTTPException(404, "audio file not found")
    thumb_dir = d / "_thumbs"
    thumb_path = thumb_dir / (Path(file).stem + ".png")
    # Serve cached thumb if it's newer than the audio file
    if thumb_path.exists() and thumb_path.stat().st_mtime >= src.stat().st_mtime:
        return FileResponse(thumb_path, media_type="image/png")
    try:
        ok = _render_audio_thumb(src, thumb_path)
        if not ok: raise HTTPException(500, "thumb render failed")
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(500, f"thumb render error: {e}")
    return FileResponse(thumb_path, media_type="image/png")


@app.get("/api/lora_data/caption")
async def lora_data_get_caption(name: str = "", folder: str = "", file: str = ""):
    if not file: raise HTTPException(400, "file required")
    if not name and not folder: raise HTTPException(400, "name or folder required")
    d = _lora_data_folder(name, folder or None)
    txt = d / (Path(file).stem + ".txt")
    if not txt.exists(): return {"text": "", "exists": False}
    try:
        return {"text": txt.read_text(encoding="utf-8"), "exists": True}
    except Exception as e:
        raise HTTPException(500, f"read failed: {e}")


class CaptionBody(BaseModel):
    name: str = ""
    folder: str | None = None
    file: str
    text: str


@app.post("/api/lora_data/caption")
async def lora_data_set_caption(body: CaptionBody):
    if not body.file: raise HTTPException(400, "file required")
    if not body.name and not body.folder: raise HTTPException(400, "name or folder required")
    d = _lora_data_folder(body.name, body.folder)
    txt = d / (Path(body.file).stem + ".txt")
    cleaned = body.text.strip()
    if cleaned == "":
        # Empty caption → delete the .txt sidecar
        if txt.exists():
            try: txt.unlink()
            except Exception as e: raise HTTPException(500, f"delete failed: {e}")
        return _scan_lora_folder(d)
    d.mkdir(parents=True, exist_ok=True)
    try:
        txt.write_text(cleaned, encoding="utf-8")
    except Exception as e:
        raise HTTPException(500, f"write failed: {e}")
    return _scan_lora_folder(d)


class LoraDataClearBody(BaseModel):
    name: str = ""
    folder: str | None = None


@app.post("/api/lora_data/clear")
async def lora_data_clear(body: LoraDataClearBody):
    if not body.name and not body.folder:
        return _scan_lora_folder(_lora_data_folder(body.name, body.folder))
    d = _lora_data_folder(body.name, body.folder)
    if d.is_dir():
        for p in d.iterdir():
            if p.is_file():
                try: p.unlink()
                except Exception as e: print(f"[lora_data] failed to remove {p}: {e}")
    return _scan_lora_folder(d)


class TrainSettingsGetBody(BaseModel):
    name: str = ""

class TrainSettingsSetBody(BaseModel):
    name: str = ""
    settings: dict = {}


def _train_settings_path(name: str) -> Path:
    return LORA_TRAIN_DIR / name / "train_settings.json"


@app.post("/api/lora_data/train_settings/get")
async def train_settings_get(body: TrainSettingsGetBody):
    if not body.name: return {}
    p = _train_settings_path(body.name)
    if not p.exists(): return {}
    try: return json.loads(p.read_text())
    except Exception: return {}


@app.post("/api/lora_data/train_settings/set")
async def train_settings_set(body: TrainSettingsSetBody):
    if not body.name: raise HTTPException(400, "name required")
    p = _train_settings_path(body.name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body.settings or {}, indent=2))
    return {"ok": True}


@app.post("/api/lora_data/clear_captions")
async def lora_data_clear_captions(body: LoraDataClearBody):
    """Delete every .txt sidecar that pairs with an audio file in the folder."""
    if not body.name and not body.folder:
        raise HTTPException(400, "name or folder required")
    d = _lora_data_folder(body.name, body.folder)
    if d.is_dir():
        audio_stems = {p.stem for p in d.iterdir()
                       if p.is_file() and p.suffix.lower() in _AUDIO_EXTS}
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() == ".txt" and p.stem in audio_stems:
                try: p.unlink()
                except Exception as e: print(f"[lora_data] failed to remove {p}: {e}")
    return _scan_lora_folder(d)


class ProfileBody(BaseModel):
    name: str = ""              # lora name — used to locate pre-encoded data
    adapter_type: str = "dora-rows"
    rank: int = 16
    batch_size: int = 1
    n_steps: int = 5


@app.post("/api/train_lora/profile")
async def train_lora_profile(body: ProfileBody):
    """Run N actual training steps to measure ms/step + peak RAM.
    Requires pre-encoded latents at LORA_TRAIN_DIR/<name>/."""
    import tempfile, shutil
    if not body.name:
        body.name = "_scratch"

    encoded_dir = LORA_TRAIN_DIR / body.name / "_encoded"
    has_real_latents = encoded_dir.is_dir() and bool(list(encoded_dir.glob("*.npy")))

    dummy_dir = None
    if not has_real_latents:
        # No pre-encoded data: synthesize a tiny dummy dataset matching the latent
        # shape produced by pre_encode_mlx.py (256-dim × T_lat frames). 30s clips
        # yield T_lat≈324; we use 4 files so batch sampling has variety.
        dummy_dir = Path(tempfile.mkdtemp(prefix="sa3_profile_dummy_"))
        T_lat = 324
        rng = np.random.default_rng(0)
        for i in range(4):
            lat = rng.standard_normal((256, T_lat)).astype(np.float16)
            np.save(str(dummy_dir / f"{i:010d}.npy"), lat)
            meta = {"path": "dummy", "relpath": f"dummy_{i}.wav",
                    "seconds_total": 30.0, "padding_mask": [1] * T_lat}
            (dummy_dir / f"{i:010d}.json").write_text(json.dumps(meta))
        encoded_dir = dummy_dir

    out_dir = Path(tempfile.mkdtemp(prefix="sa3_profile_"))
    script = str(Path(__file__).resolve().parent.parent / "mlx_sa3" / "train_lora_mlx.py")
    cmd = [
        sys.executable, script,
        "--encoded-dir", str(encoded_dir),
        "--output-dir", str(out_dir),
        "--rank", str(body.rank),
        "--adapter-type", body.adapter_type,
        "--steps", str(max(2, body.n_steps)),
        "--batch-size", str(max(1, body.batch_size or 1)),
        "--checkpoint-every", "9999999",   # don't checkpoint during profile
    ]

    env = os.environ.copy()
    env["SA3_ROOT"] = _settings["sa3_root"]
    env["SA3_MODELS_DIR"] = str(MODELS_DIR)
    env["SA3_LORA_DIR"] = str(LORA_DIR)
    if _settings.get("hf_token"):
        env["HF_TOKEN"] = _settings["hf_token"]

    # Track peak memory in the parent process group via psutil
    import psutil as _ps
    peak_bytes = 0
    cancelled = False
    t0 = time.time()
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    async def _watch_mem():
        nonlocal peak_bytes
        try:
            p = _ps.Process(proc.pid)
        except _ps.NoSuchProcess:
            return
        while proc.returncode is None:
            try:
                rss = p.memory_info().rss
                for c in p.children(recursive=True):
                    try: rss += c.memory_info().rss
                    except _ps.NoSuchProcess: pass
                if rss > peak_bytes: peak_bytes = rss
            except _ps.NoSuchProcess:
                break
            await asyncio.sleep(0.5)

    watcher = asyncio.create_task(_watch_mem())
    try:
        await asyncio.wait_for(proc.wait(), timeout=600)
    except asyncio.TimeoutError:
        cancelled = True
        proc.kill()
    watcher.cancel()
    elapsed = time.time() - t0
    stdout = (await proc.stdout.read()).decode("utf-8", errors="replace") if proc.stdout else ""
    stderr = (await proc.stderr.read()).decode("utf-8", errors="replace") if proc.stderr else ""

    # cleanup temp dirs
    try: shutil.rmtree(out_dir, ignore_errors=True)
    except Exception: pass
    if dummy_dir is not None:
        try: shutil.rmtree(dummy_dir, ignore_errors=True)
        except Exception: pass

    if proc.returncode != 0:
        # Surface the final exception line (Python tracebacks: "Traceback…"
        # header is useless; the actual `ExceptionType: message` is the last
        # non-empty line). Fall back to the tail of stderr if nothing parses.
        non_empty = [l for l in stderr.splitlines() if l.strip()]
        exc_line = next(
            (l.strip() for l in reversed(non_empty)
             if ":" in l and not l.lstrip().startswith(("File ", "Traceback"))),
            None,
        )
        if not exc_line:
            exc_line = "\n".join(non_empty[-6:]) if non_empty else "training script failed"
        # Also dump the full stderr server-side so the user can grep it.
        print(f"[profile] FAILED (rc={proc.returncode})\n{stderr}", flush=True)
        raise HTTPException(500, f"profile run failed (rc={proc.returncode}): {exc_line[:400]}")

    # parse the script's final json line for elapsed_s / it_s if present
    script_elapsed = None; script_it_s = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"): continue
        try:
            j = json.loads(line)
            if "elapsed_s" in j: script_elapsed = float(j["elapsed_s"])
            if "it_s" in j:      script_it_s = float(j["it_s"])
            break
        except Exception: continue

    n = max(2, body.n_steps)
    if script_it_s and script_it_s > 0:
        ms_per_step = 1000.0 / script_it_s
    elif script_elapsed and script_elapsed > 0:
        ms_per_step = script_elapsed * 1000.0 / n
    else:
        ms_per_step = elapsed * 1000.0 / n  # fallback: wall clock (includes startup)

    return {
        "ms_per_step": ms_per_step,
        "peak_ram_gb": peak_bytes / (1024**3),
        "wall_sec": elapsed,
        "n_steps": n,
        "script_it_s": script_it_s,
        "stub": False,
    }


# ── auto-captioning ─────────────────────────────────────────────────────────
from backend import captioner as _cap

CAPTION_MODELS = {
    "pro":   "google/gemini-3.1-pro-preview",
    "flash": "google/gemini-3.5-flash",
}

class CaptionEstimateBody(BaseModel):
    name: str = ""
    folder: str | None = None
    only_uncaptioned: bool = True
    model: str = "pro"      # "pro" or "flash"

@app.post("/api/lora_data/autocaption/estimate")
async def autocaption_estimate(body: CaptionEstimateBody):
    d = _lora_data_folder(body.name, body.folder)
    scan = _scan_lora_folder(d)
    files = scan["files"]
    if body.only_uncaptioned:
        files = [f for f in files if not f["captioned"]]
    model = CAPTION_MODELS.get(body.model, CAPTION_MODELS["pro"])
    est = _cap.estimate_cost(len(files), model=model)
    return {"n_files": len(files), "model": model, **est}


class CaptionStartBody(BaseModel):
    name: str = ""
    folder: str | None = None
    only_uncaptioned: bool = True
    examples: list[str] | None = None
    parallel: int = 32
    model: str = "pro"


_caption_task = None
@app.post("/api/lora_data/autocaption/start")
async def autocaption_start(body: CaptionStartBody):
    global _caption_task
    if _cap.status.running:
        raise HTTPException(409, "captioning already in progress")
    api_key = _settings.get("openrouter_api_key", "")
    if not api_key:
        raise HTTPException(400, "OpenRouter API key not set in Settings")
    d = _lora_data_folder(body.name, body.folder)
    scan = _scan_lora_folder(d)
    files = [Path(scan["folder"]) / f["name"] for f in scan["files"]
             if not body.only_uncaptioned or not f["captioned"]]
    if not files:
        return {"started": False, "reason": "no files to caption"}
    examples = body.examples or _cap.DEFAULT_EXAMPLES
    model = CAPTION_MODELS.get(body.model, CAPTION_MODELS["pro"])
    parallel = max(1, min(128, int(body.parallel)))
    # Launch on the running loop in a fire-and-forget task
    loop = asyncio.get_running_loop()
    _caption_task = loop.create_task(
        _cap.run_caption_batch(api_key, Path(scan["folder"]), files, examples,
                               parallel, model, lora_name=body.name)
    )
    return {"started": True, "total": len(files), "parallel": parallel, "model": model}


@app.get("/api/lora_data/autocaption/status")
async def autocaption_status():
    return _cap.status.snapshot()


@app.post("/api/lora_data/autocaption/cancel")
async def autocaption_cancel():
    # Flip running=false immediately so the UI clears without waiting for
    # in-flight requests to drain; the worker pool watches `cancelled` and
    # short-circuits any new work.
    _cap.status.cancelled = True
    _cap.status.running = False
    return {"cancelled": True}


# Persist examples per-folder so the UI can recall the user's edits
@app.post("/api/lora_data/captioner_examples/get")
async def captioner_examples_get(body: LoraDataQuery):
    d = _lora_data_folder(body.name, body.folder)
    cfg = d.parent / "captioner.json"
    if cfg.exists():
        try:
            j = json.loads(cfg.read_text())
            return {"examples": j.get("examples") or _cap.DEFAULT_EXAMPLES,
                    "model":    j.get("model",    "pro"),
                    "parallel": j.get("parallel", 32)}
        except Exception: pass
    return {"examples": _cap.DEFAULT_EXAMPLES, "model": "pro", "parallel": 32}


class CaptionerCfgBody(BaseModel):
    name: str = ""
    folder: str | None = None
    examples: list[str]
    model: str = "pro"
    parallel: int = 32

@app.post("/api/lora_data/captioner_examples/set")
async def captioner_examples_set(body: CaptionerCfgBody):
    d = _lora_data_folder(body.name, body.folder)
    d.parent.mkdir(parents=True, exist_ok=True)
    cfg = d.parent / "captioner.json"
    cfg.write_text(json.dumps({
        "examples": body.examples, "model": body.model, "parallel": body.parallel,
    }, indent=2))
    return {"ok": True}


class LoraDataRenameBody(BaseModel):
    old_name: str
    new_name: str


@app.post("/api/lora_data/rename")
async def lora_data_rename(body: LoraDataRenameBody):
    """Rename or load: if dst doesn't exist, move src → dst (so an in-progress
    `_scratch` becomes a named LoRA, carrying its files). If dst already exists,
    don't error — just switch to it (load its files), leaving src in place.
    The frontend treats this as 'change the name field and the dataset follows.'"""
    if not body.old_name or not body.new_name:
        raise HTTPException(400, "old_name and new_name required")
    if "/" in body.new_name or "\\" in body.new_name or body.new_name.startswith("."):
        raise HTTPException(400, "invalid lora name")
    if body.old_name == body.new_name:
        return _scan_lora_folder(_lora_data_folder(body.new_name, None))
    src = (LORA_TRAIN_DIR / body.old_name).resolve()
    dst = (LORA_TRAIN_DIR / body.new_name).resolve()
    if dst.exists():
        # Switching to an existing LoRA. Don't move src; just return dst's data.
        return _scan_lora_folder(_lora_data_folder(body.new_name, None))
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
    return _scan_lora_folder(_lora_data_folder(body.new_name, None))


class LoraDataDeleteBody(BaseModel):
    name: str = ""
    folder: str | None = None
    file: str


@app.post("/api/lora_data/delete")
async def lora_data_delete(body: LoraDataDeleteBody):
    d = _lora_data_folder(body.name, body.folder)
    target = d / body.file
    # guard against path traversal
    try:
        target.resolve().relative_to(d.resolve())
    except ValueError:
        raise HTTPException(400, "invalid file path")
    if target.exists():
        target.unlink()
    txt = d / (Path(body.file).stem + ".txt")
    if txt.exists() and txt != target:
        txt.unlink()
    return _scan_lora_folder(d)


# -------- LoRA training --------

class TrainLoraBody(BaseModel):
    folder: str
    name: str
    caption: str = ""
    rank: int = 16
    adapter_type: str = "dora-rows"
    steps: int = 1000
    lr: float = 1e-4
    batch_size: int = 1
    checkpoint_every: int = 100
    exclude: list[str] | None = None
    use_compile: bool = False
    train_conditioner: bool = False
    dist_shift: bool = True
    grad_checkpoint: bool = True

class PreEncodeBody(BaseModel):
    folder: str
    name: str
    caption: str = ""

_lora_train_proc = None
_lora_train_last_result = None
# Continuous stdout drain for the lora-train subprocess. _lora_train_progress
# holds the latest parseable JSON status line; _lora_train_buf accumulates the
# full transcript so the completion path can read it without racing the drainer.
_lora_train_progress: dict = {}
_lora_train_buf: list[str] = []
_lora_train_steps_all: list[dict] = []     # full history of step events; survives tab switches
_lora_train_started_ts: float = 0.0        # wall time when the subprocess was spawned
_lora_train_reader_task = None


async def _drain_lora_train_stream():
    global _lora_train_progress
    proc = _lora_train_proc
    while True:
        try: line = await proc.stdout.readline()
        except Exception: return
        if not line: return
        text = line.decode("utf-8", errors="replace")
        _lora_train_buf.append(text)
        t = text.strip()
        if t.startswith("{") and t.endswith("}"):
            try:
                ev = json.loads(t)
            except Exception:
                continue
            _lora_train_progress = ev
            # Persist every step event so the frontend can rebuild the chart
            # history on remount (e.g. after a tab switch).
            if ev.get("status") == "step":
                _lora_train_steps_all.append(ev)
                # Cap so a runaway long run doesn't blow up RAM. 8k step events
                # ≈ ~500KB; still cheap to ship over a JSON poll.
                if len(_lora_train_steps_all) > 8000:
                    del _lora_train_steps_all[:len(_lora_train_steps_all) - 8000]
_pre_encode_proc = None
_pre_encode_last_result = None
# Background-reader state. The reader task drains stdout/stderr lines into
# these as they arrive; the /status endpoint just snapshots `progress`.
_pre_encode_progress: dict = {}
_pre_encode_stdout_buf: list[str] = []
_pre_encode_stderr_buf: list[str] = []
_pre_encode_reader_task = None


async def _drain_pre_encode_streams():
    """Continuously read both stdout and stderr from the pre-encoder subprocess.
    Keeps `_pre_encode_progress` up to date with the latest parseable JSON line."""
    global _pre_encode_progress
    proc = _pre_encode_proc

    async def _read_stream(stream, buf, is_stdout):
        global _pre_encode_progress
        while True:
            line = await stream.readline()
            if not line: return
            text = line.decode("utf-8", errors="replace")
            buf.append(text)
            if is_stdout:
                t = text.strip()
                if t.startswith("{") and t.endswith("}"):
                    try:    _pre_encode_progress = json.loads(t)
                    except Exception: pass

    await asyncio.gather(
        _read_stream(proc.stdout, _pre_encode_stdout_buf, True),
        _read_stream(proc.stderr, _pre_encode_stderr_buf, False),
    )

@app.post("/api/pre_encode")
async def start_pre_encode(body: PreEncodeBody):
    global _pre_encode_proc
    if _pre_encode_proc and _pre_encode_proc.returncode is None:
        raise HTTPException(409, "Pre-encoding already in progress")

    folder = Path(body.folder).expanduser().resolve()
    if not folder.is_dir():
        raise HTTPException(400, f"folder not found: {folder}")

    output_dir = str(LORA_TRAIN_DIR / body.name)
    LORA_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    ae_model = "same-l"
    if _current_model and "small" in _current_model:
        ae_model = "same-s"

    env = os.environ.copy()
    env["SA3_ROOT"] = _settings["sa3_root"]
    env["SA3_MODELS_DIR"] = str(MODELS_DIR)
    if _settings.get("hf_token"):
        env["HF_TOKEN"] = _settings["hf_token"]

    if HAS_MLX:
        # MLX pre-encoding path (Apple Silicon)
        script = str(Path(__file__).resolve().parent.parent / "mlx_sa3" / "pre_encode_mlx.py")
        enc_dir = str(Path(output_dir) / "_encoded")
        cmd = [
            sys.executable, script,
            "--audio-dir", str(folder),
            "--output-dir", enc_dir,
        ]
    else:
        # PyTorch pre-encoding path (CUDA)
        script = str(Path(__file__).resolve().parent / "pre_encode.py")
        cmd = [
            sys.executable, script,
            "--audio-folder", str(folder),
            "--output-dir", output_dir,
            "--caption", body.caption or body.name,
            "--ae-model", ae_model,
        ]

    _pre_encode_proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    # Reset shared reader state and spawn the drain task.
    global _pre_encode_progress, _pre_encode_stdout_buf, _pre_encode_stderr_buf, _pre_encode_reader_task
    _pre_encode_progress = {}
    _pre_encode_stdout_buf = []
    _pre_encode_stderr_buf = []
    _pre_encode_reader_task = asyncio.create_task(_drain_pre_encode_streams())
    return {"status": "started", "name": body.name, "output_dir": output_dir}


@app.get("/api/pre_encode/status")
async def pre_encode_status():
    global _pre_encode_proc, _pre_encode_last_result
    if _pre_encode_proc is None:
        if _pre_encode_last_result is not None:
            result = _pre_encode_last_result
            _pre_encode_last_result = None
            return result
        return {"status": "idle"}
    if _pre_encode_proc.returncode is not None:
        # Make sure the reader task has finished draining both streams.
        if _pre_encode_reader_task is not None:
            try: await asyncio.wait_for(_pre_encode_reader_task, timeout=2.0)
            except Exception: pass
        stdout = "".join(_pre_encode_stdout_buf)
        stderr = "".join(_pre_encode_stderr_buf)
        last = dict(_pre_encode_progress) if _pre_encode_progress else {}
        rc = _pre_encode_proc.returncode
        ok = rc == 0
        _pre_encode_proc = None
        if not ok:
            print(f"[pre_encode] FAILED (rc={rc})")
            if stderr:
                print(f"[pre_encode] stderr: {stderr[:2000]}")
        _pre_encode_last_result = {
            "status": "done" if ok else "error",
            "result": last,
            "error": stderr[:2000] if not ok else None,
        }
        return _pre_encode_last_result
    return {"status": "running", "progress": dict(_pre_encode_progress)}


@app.post("/api/pre_encode/cancel")
async def cancel_pre_encode():
    global _pre_encode_proc
    if _pre_encode_proc is None or _pre_encode_proc.returncode is not None:
        return {"cancelled": False, "reason": "not running"}
    try: _pre_encode_proc.kill()
    except Exception: pass
    return {"cancelled": True}


class ClearEncodedBody(BaseModel):
    name: str

@app.post("/api/pre_encode/clear")
async def clear_encoded(body: ClearEncodedBody):
    """Delete the cached latents directory for a given lora name."""
    if not body.name: raise HTTPException(400, "name required")
    enc_dir = LORA_TRAIN_DIR / body.name / "_encoded"
    if enc_dir.is_dir():
        import shutil
        shutil.rmtree(enc_dir, ignore_errors=True)
    return {"cleared": True, "name": body.name}


@app.get("/api/lora_training/{name}/has_encoded")
async def check_encoded(name: str):
    enc_dir = LORA_TRAIN_DIR / name / "_encoded"
    if enc_dir.is_dir():
        n = len(list(enc_dir.glob("*.npy")))
        return {"has_encoded": n > 0, "latents": n}
    return {"has_encoded": False, "latents": 0}


@app.post("/api/train_lora")
async def start_lora_training(body: TrainLoraBody):
    global _lora_train_proc, _training_unloaded_model
    if _lora_train_proc and _lora_train_proc.returncode is None:
        raise HTTPException(409, "LoRA training already in progress")

    folder = Path(body.folder).expanduser().resolve()
    if not folder.is_dir():
        raise HTTPException(400, f"folder not found: {folder}")

    output_dir = str(LORA_TRAIN_DIR / body.name)
    LORA_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    # Train against whatever model is currently loaded (defaults to medium).
    model_name = _current_model or "medium"

    # unload inference model to free VRAM for training
    _training_unloaded_model = _current_model
    _unload_model()

    # resolve auto batch size from free VRAM (~3GB per batch element for full LoRA training)
    batch_size = body.batch_size
    if batch_size <= 0 and DEVICE == "cuda":
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        batch_size = max(1, min(8, int(free_gb / 3.0)))
        print(f"[lora_train] auto batch: {free_gb:.1f}GB free → batch_size={batch_size}")
    elif batch_size <= 0:
        batch_size = 1

    # total_steps ÷ batch = optimizer steps (keeps wall time constant regardless of batch)
    optimizer_steps = max(1, body.steps // batch_size)
    checkpoint_every = max(1, body.checkpoint_every // batch_size)
    print(f"[lora_train] total_steps={body.steps} ÷ batch={batch_size} → optimizer_steps={optimizer_steps}")

    env = os.environ.copy()
    env["SA3_ROOT"] = _settings["sa3_root"]
    env["SA3_MODELS_DIR"] = str(MODELS_DIR)
    env["SA3_LORA_DIR"] = str(LORA_DIR)
    if _settings.get("hf_token"):
        env["HF_TOKEN"] = _settings["hf_token"]

    # Check for pre-encoded latents
    enc_dir = Path(output_dir) / "_encoded"
    has_encoded = enc_dir.is_dir() and list(enc_dir.glob("*.npy"))

    use_mlx_train = HAS_MLX
    if use_mlx_train:
        # MLX training path (Apple Silicon)
        script = str(Path(__file__).resolve().parent.parent / "mlx_sa3" / "train_lora_mlx.py")
        if not has_encoded:
            raise HTTPException(400,
                "MLX training requires pre-encoded latents. "
                "Run pre-encoding first or transfer pre-encoded data from a CUDA machine.")
        cmd = [
            sys.executable, script,
            "--model-name", model_name,
            "--encoded-dir", str(enc_dir),
            "--output-dir", output_dir,
            "--caption", body.caption or body.name,
            "--rank", str(body.rank),
            "--adapter-type", body.adapter_type,
            "--steps", str(optimizer_steps),
            "--lr", str(body.lr),
            "--batch-size", str(batch_size),
            "--checkpoint-every", str(checkpoint_every),
        ]
        if body.dist_shift:
            cmd.append("--dist-shift")
        if body.grad_checkpoint:
            cmd.append("--grad-checkpoint")
        if body.train_conditioner:
            cmd.append("--train-conditioner")
        if body.exclude:
            cmd.extend(["--exclude"] + body.exclude)
        print(f"[lora_train] using MLX training backend ({body.adapter_type})")
    else:
        # PyTorch training path (CUDA)
        script = str(Path(__file__).resolve().parent / "train_lora.py")
        cmd = [
            sys.executable, script,
            "--model-name", model_name,
            "--audio-folder", str(folder),
            "--output-dir", output_dir,
            "--caption", body.caption or body.name,
            "--rank", str(body.rank),
            "--adapter-type", body.adapter_type,
            "--steps", str(optimizer_steps),
            "--lr", str(body.lr),
            "--batch-size", str(batch_size),
            "--checkpoint-every", str(checkpoint_every),
        ]
        if body.exclude:
            cmd.extend(["--exclude"] + body.exclude)
        if body.use_compile:
            cmd.append("--compile")

    _lora_train_proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    global _lora_train_progress, _lora_train_buf, _lora_train_steps_all, _lora_train_started_ts, _lora_train_reader_task
    _lora_train_progress = {}
    _lora_train_buf = []
    _lora_train_steps_all = []
    _lora_train_started_ts = time.time()
    _lora_train_reader_task = asyncio.create_task(_drain_lora_train_stream())
    return {"status": "started", "name": body.name, "output_dir": output_dir,
            "batch_size": batch_size, "optimizer_steps": optimizer_steps,
            "backend": "mlx" if use_mlx_train else "cuda"}


@app.get("/api/train_lora/status")
async def lora_train_status():
    global _lora_train_proc, _lora_train_last_result, _training_unloaded_model
    if _lora_train_proc is None:
        if _lora_train_last_result is not None:
            result = _lora_train_last_result
            _lora_train_last_result = None
            return result
        return {"status": "idle"}
    if _lora_train_proc.returncode is not None:
        if _lora_train_reader_task is not None:
            try: await asyncio.wait_for(_lora_train_reader_task, timeout=2.0)
            except Exception: pass
        stdout = "".join(_lora_train_buf)
        lines = [l.strip() for l in stdout.splitlines() if l.strip()]
        last = dict(_lora_train_progress) if _lora_train_progress else {}
        rc = _lora_train_proc.returncode
        ok = rc == 0
        _lora_train_proc = None
        if not ok:
            print(f"[lora_train] FAILED (rc={rc})")
            err_lines = [l for l in lines if "error" in l.lower() or "traceback" in l.lower() or "exception" in l.lower()]
            print(f"[lora_train] stdout errors: {err_lines[-3:] if err_lines else lines[-5:]}")
        error_detail = "\n".join(lines[-10:]) if not ok else None
        _lora_train_last_result = {
            "status": "done" if ok else "error",
            "result": last,
            "error": error_detail,
        }
        # reload the inference model that was unloaded for training
        if _training_unloaded_model:
            reload_name = _training_unloaded_model
            _training_unloaded_model = None
            print(f"[backend] reloading {reload_name} after training...")
            try:
                _load_model(reload_name)
                _lora_train_last_result["model_reloaded"] = True
            except Exception as e:
                print(f"[backend] reload failed: {e}")
                _lora_train_last_result["model_reloaded"] = False
                _lora_train_last_result["reload_error"] = str(e)
        return _lora_train_last_result
    # Still running. We ship the FULL step history so a remount (tab switch /
    # refresh) can rebuild charts in one shot. ~8k events caps at ~500KB.
    body = {
        "status": "running",
        "progress": dict(_lora_train_progress),
        "steps_all": list(_lora_train_steps_all),
        "started_ts": _lora_train_started_ts,
    }
    return body


@app.post("/api/train_lora/cancel")
async def lora_train_cancel():
    global _lora_train_proc
    if _lora_train_proc is None or _lora_train_proc.returncode is not None:
        return {"cancelled": False, "reason": "not running"}
    try: _lora_train_proc.kill()
    except Exception: pass
    return {"cancelled": True}


VALID_MODELS = {"medium", "medium-base", "small-music", "small-sfx"}


class ModelBody(BaseModel):
    model: str


@app.post("/api/model")
async def switch_model(body: ModelBody):
    name = body.model
    if name not in VALID_MODELS:
        raise HTTPException(400, f"unknown model: {name}")
    if name == _current_model:
        return {"model": name, "status": "already_loaded"}

    try:
        await asyncio.to_thread(_load_model, name)
    except Exception as e:
        raise HTTPException(500, f"model load failed: {e}")

    return {"model": name, "status": "loaded", "backend": BACKEND,
            "mlx_ae": _use_mlx_ae}


class PrecisionBody(BaseModel):
    precision: str


@app.post("/api/precision")
async def set_precision(body: PrecisionBody):
    global _use_fp16
    want_fp16 = body.precision == "fp16"
    if want_fp16 == _use_fp16:
        return {"precision": "fp16" if _use_fp16 else "fp32"}
    if DEVICE != "cuda":
        raise HTTPException(400, "precision switching requires CUDA")
    if want_fp16:
        sa.model.half()
        sa.model_half = True
    else:
        sa.model.float()
        sa.model_half = False
    _use_fp16 = want_fp16
    torch.cuda.empty_cache()
    print(f"[backend] precision switched to {'fp16' if _use_fp16 else 'fp32'}")
    return {"precision": "fp16" if _use_fp16 else "fp32"}


class DecodeSettingsBody(BaseModel):
    decode_fp32: bool | None = None
    decode_overlap: int | None = None


@app.get("/api/decode_settings")
async def get_decode_settings():
    return {"decode_fp32": _decode_fp32, "decode_overlap": _decode_overlap}


@app.post("/api/decode_settings")
async def set_decode_settings(body: DecodeSettingsBody):
    global _decode_fp32, _decode_overlap
    if body.decode_fp32 is not None:
        _decode_fp32 = body.decode_fp32
    if body.decode_overlap is not None:
        _decode_overlap = max(0, min(128, body.decode_overlap))
    print(f"[backend] decode settings: fp32={_decode_fp32} overlap={_decode_overlap}")
    return {"decode_fp32": _decode_fp32, "decode_overlap": _decode_overlap}


@app.get("/api/stats")
async def get_stats():
    cpu = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    ram_used_gb = (vm.total - vm.available) / 1e9
    ram_total_gb = vm.total / 1e9
    gpu_alloc_gb = 0.0
    try:
        if DEVICE == "cuda" and torch.cuda.is_available():
            gpu_alloc_gb = torch.cuda.memory_allocated() / 1e9
        elif DEVICE == "mps" and hasattr(torch, "mps"):
            gpu_alloc_gb = torch.mps.current_allocated_memory() / 1e9
    except Exception: pass
    return {
        "cpu": round(cpu, 1),
        "ram_used": round(ram_used_gb, 1),
        "ram_total": round(ram_total_gb, 1),
        "gpu_alloc": round(gpu_alloc_gb, 2),
        "precision": "fp16" if _use_fp16 else "fp32",
        "backend": BACKEND,
        "model": _current_model,
        "model_loaded": sa is not None,
    }


@app.get("/api/audio")
async def get_audio():
    if not state["audio_path"]:
        raise HTTPException(404, "no audio")
    return FileResponse(state["audio_path"], media_type="audio/wav")


@app.get("/api/envelope.json")
async def get_env():
    p = DATA_DIR / "envelope.json"
    if not p.exists():
        return {"count": 0, "data": [], "downsample": DOWNSAMPLE, "sr": SR}
    return FileResponse(p, media_type="application/json")


@app.get("/api/spec.png")
async def get_spec():
    p = DATA_DIR / "current_spec.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.get("/api/overview.png")
async def get_overview():
    p = DATA_DIR / "current_overview.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.get("/api/noise_spec.png")
async def get_noise_spec():
    p = DATA_DIR / "noise_spec.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


render_noise_spec_once()
print(f"[backend] ready (backend={BACKEND}, device={DEVICE})")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5174, log_level="info")
