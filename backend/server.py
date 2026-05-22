"""SA3 Inpainter backend. FastAPI on :5174.

Loads the SA3 medium model once at startup (~30s), exposes JSON API for the
Svelte frontend.

Dual backend: auto-detects CUDA (torch AE) vs MPS (MLX AE).
Set SA3_BACKEND=cuda or SA3_BACKEND=mlx to force.
"""
import asyncio
import gc
import os, sys, json, time, threading, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
import torch
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import stft

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from stable_audio_3.factory import create_diffusion_cond_from_config
from stable_audio_3 import StableAudioModel
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

LOCAL_MEDIUM = os.environ.get("SA3_MODEL_DIR", str(Path.home() / "models/stable-audio-3-medium"))
DATA_DIR = Path("/tmp/sa3-inpainter"); DATA_DIR.mkdir(exist_ok=True)
SR = 44100
DOWNSAMPLE = 4096
BANDS = [(0, 250), (250, 2500), (2500, 22050)]

# -------- model state --------

sa = None               # StableAudioModel instance
mlx_ae = None           # MLX AE, only for medium on MPS
_use_mlx_ae = False     # whether current model uses MLX AE decode path
_current_model = None   # name of loaded model
_use_fp16 = os.environ.get("SA3_FP16", "1" if DEVICE == "cuda" else "0") == "1"
_cancel_event = threading.Event()
_loaded_lora_name = None


def _load_model(name, local_path=None):
    """Load a model by name. local_path overrides for manual loading."""
    global sa, mlx_ae, _use_mlx_ae, _current_model, _loaded_lora_name

    # cleanup old model
    sa = None
    mlx_ae = None
    _use_mlx_ae = False
    _loaded_lora_name = None
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

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
        # MLX AE only available for medium models loaded from local path;
        # from_pretrained models fall back to torch AE on MPS (slower but works)

    _current_model = name
    print(f"[backend] {name} ready (mlx_ae={'yes' if _use_mlx_ae else 'no'})")


# initial load
_load_model("medium", local_path=LOCAL_MEDIUM)
if _use_fp16 and DEVICE == "cuda":
    print("[backend] fp16 enabled")


# -------- decode helpers --------

def _decode_latents(lat_np):
    """Decode latent numpy array → waveform numpy array.
    lat_np: (1, 256, T_lat) float32
    Returns: (channels, T_audio) float32 numpy
    """
    if _use_mlx_ae and mlx_ae is not None:
        lat_mx = mx.array(lat_np)
        if lat_np.shape[-1] > 128:
            wav_m = decode_chunked(mlx_ae, lat_mx, chunk_size=128, overlap=32)
        else:
            wav_m = mlx_ae.decode(lat_mx)
        mx.eval(wav_m)
        return np.array(wav_m)[0]
    else:
        lat_t = torch.from_numpy(lat_np).to(DEVICE)
        if _use_fp16 and DEVICE == "cuda":
            lat_t = lat_t.half()
        with torch.inference_mode():
            wav_t = sa.same.decode(lat_t)
        return wav_t.float().cpu().numpy()[0]


def render_noise_spec_once():
    out_path = DATA_DIR / "noise_spec.png"
    if out_path.exists(): return
    T_lat = int(30 * SR / DOWNSAMPLE) + 1
    rng = np.random.default_rng(7)
    lat = rng.standard_normal((1, 256, T_lat)).astype(np.float32) * 0.3
    wav_np = _decode_latents(lat)
    render_spec_png(wav_np, out_path)

state = {"audio_path": None, "version": 0}
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
    hop = DOWNSAMPLE
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
    fig = plt.figure(figsize=(20, 6), dpi=100)
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
    peaks /= peaks.max() + 1e-9
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
            sa.load_lora(str(lora_path))
            _loaded_lora_name = name
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
    raw = DATA_DIR / "upload.wav"
    with open(raw, "wb") as f: f.write(await file.read())
    audio, sr = sf.read(raw)
    if audio.ndim == 1: audio = np.stack([audio, audio], axis=-1)
    if sr != SR:
        import torchaudio
        a = torch.from_numpy(audio.T).float()
        a = torchaudio.transforms.Resample(sr, SR)(a)
        audio = a.numpy().T
    env = persist_audio(audio.T)
    return {"version": state["version"], "count": env["count"],
            "duration": env["count"] * DOWNSAMPLE / SR}


class LoraEntry(BaseModel):
    name: str
    strength: float = 1.0


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


@app.post("/api/generate")
async def generate(body: GenBody):
    _cancel_event.clear()

    s = body.settings
    steps = int(s.get("steps", 8))
    cfg = float(s.get("cfg", 1.0))
    seed = int(s.get("seed", 42))
    noise = float(s.get("noise", 1.0))
    duration = float(s.get("duration", 30.0))
    sampler_type = s.get("sampler_type")
    apg_scale = float(s.get("apg_scale", 1.0))

    has_source = state["audio_path"] is not None
    has_mask = any(body.mask) if body.mask else False
    n_regen = sum(body.mask) if body.mask else 0
    print(f"[generate] source={has_source} mask_len={len(body.mask) if body.mask else 0} regen_latents={n_regen} mode={('inpaint' if has_source and has_mask else 'vary' if has_source else 't2a')}")

    neg_prompt = body.negative_prompt or None
    kwargs = dict(prompt=body.prompt, negative_prompt=neg_prompt, steps=steps,
                  cfg_scale=cfg, seed=seed, apg_scale=apg_scale,
                  return_latents=_use_mlx_ae,
                  chunked_decode=(not _use_mlx_ae))
    if sampler_type:
        kwargs["sampler_type"] = sampler_type

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
        try:
            if _cancel_event.is_set():
                return None

            t0 = time.time()
            result = sa.generate(**kwargs)

            if _use_mlx_ae:
                lat_np = result.detach().to(torch.float32).cpu().numpy()
                print(f"[backend] DIT {time.time()-t0:.1f}s, latents shape {lat_np.shape}")
                t1 = time.time()
                wav_np = _decode_latents(lat_np)
                print(f"[backend] AE {time.time()-t1:.1f}s")
            else:
                wav_np = result.detach().to(torch.float32).cpu().numpy()[0]
                print(f"[backend] DIT+AE {time.time()-t0:.1f}s, wav shape {wav_np.shape}")

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
            _unload_loras(loras_list)

    wav_np = await asyncio.to_thread(_run_generate)

    if wav_np is None:
        raise HTTPException(status_code=499, detail="generation cancelled")

    env = persist_audio(wav_np)
    return {"version": state["version"], "count": env["count"],
            "duration": env["count"] * DOWNSAMPLE / SR}


@app.post("/api/clear")
async def clear():
    state["audio_path"] = None
    state["version"] += 1
    return {"version": state["version"]}


@app.get("/api/state")
async def get_state():
    return {"has_audio": state["audio_path"] is not None, "version": state["version"],
            "model_loaded": sa is not None, "backend": BACKEND, "model": _current_model}


import psutil
_proc = psutil.Process()
psutil.cpu_percent(interval=None)
LORA_DIR = Path(os.environ.get("SA3_LORA_DIR", str(Path.home() / "loras")))

@app.get("/api/loras")
async def list_loras():
    if not LORA_DIR.exists(): return {"dir": str(LORA_DIR), "files": []}
    files = sorted(p.name for p in LORA_DIR.iterdir() if p.is_file() and p.suffix == ".safetensors")
    return {"dir": str(LORA_DIR), "files": files}


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
