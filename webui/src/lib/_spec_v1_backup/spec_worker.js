// STFT + magma colormap rendering worker.
// Two ops:
//   - "scan": one-time pass over the full PCM to find a global dB max for
//     consistent normalization across renders. Posts back { op: "scan", dbRef }.
//   - "render": compute STFT for a slice and write an RGBA ImageData buffer.
//     One STFT frame per output column → no quantization at any zoom.

import FFT from "fft.js";

// 256-entry magma colormap (matplotlib's magma, sampled).
const MAGMA = new Uint8Array([
  0,0,4, 1,0,5, 1,1,6, 1,1,8, 2,1,10, 2,2,12, 2,2,14, 3,3,16,
  4,3,18, 4,4,20, 5,4,23, 6,5,25, 7,5,27, 8,6,29, 9,7,31, 10,7,34,
  11,8,36, 12,9,38, 13,10,41, 14,11,43, 16,11,46, 17,12,49, 18,13,51, 20,13,54,
  21,14,57, 23,14,59, 24,15,62, 26,15,65, 28,16,68, 29,16,71, 31,17,74, 33,17,77,
  35,18,80, 36,18,83, 38,19,86, 40,19,89, 42,19,92, 44,19,95, 46,20,98, 49,20,101,
  51,20,104, 53,20,106, 55,21,109, 57,21,112, 59,21,114, 62,22,117, 64,22,119, 66,22,121,
  68,22,124, 71,22,126, 73,22,128, 75,23,130, 78,23,132, 80,23,134, 82,23,135, 84,23,137,
  87,23,139, 89,24,140, 91,24,142, 93,24,143, 96,24,144, 98,24,146, 100,24,147, 103,24,148,
  105,25,149, 107,25,150, 109,25,151, 112,25,152, 114,25,153, 116,25,154, 118,26,154, 121,26,155,
  123,26,156, 125,26,156, 127,26,157, 130,26,157, 132,27,158, 134,27,158, 137,27,158, 139,27,159,
  141,27,159, 143,28,159, 146,28,159, 148,28,159, 150,29,159, 153,29,159, 155,29,159, 157,29,159,
  159,30,158, 162,30,158, 164,30,158, 166,30,157, 169,31,157, 171,31,156, 173,32,156, 176,32,155,
  178,32,155, 180,33,154, 183,33,153, 185,33,153, 187,34,152, 190,34,151, 192,35,150, 194,35,150,
  196,36,149, 199,36,148, 201,37,147, 203,38,146, 205,38,145, 207,39,144, 209,40,143, 212,40,142,
  214,41,141, 216,42,140, 218,43,139, 219,44,138, 221,45,137, 223,46,136, 225,47,135, 227,48,134,
  228,50,132, 230,51,131, 232,52,130, 233,54,129, 235,55,128, 236,56,127, 238,58,125, 239,60,124,
  240,61,123, 241,63,122, 243,64,121, 244,66,120, 245,68,118, 246,69,117, 247,71,116, 248,73,115,
  249,75,114, 250,77,112, 250,79,111, 251,81,110, 251,83,109, 252,85,108, 252,87,107, 253,89,106,
  253,91,105, 253,93,104, 254,95,103, 254,97,102, 254,99,101, 254,101,100, 254,103,99, 254,105,98,
  255,108,97, 255,110,97, 255,112,96, 255,114,95, 255,116,94, 254,118,94, 254,120,93, 254,122,92,
  254,124,92, 254,126,91, 254,128,91, 254,130,90, 253,132,89, 253,134,89, 253,136,88, 253,138,88,
  252,140,87, 252,142,87, 252,144,86, 251,146,86, 251,149,85, 250,151,85, 250,153,84, 250,155,84,
  249,157,84, 249,159,83, 248,161,83, 248,163,82, 247,165,82, 247,167,81, 246,169,81, 246,172,81,
  245,174,80, 245,176,80, 244,178,80, 244,180,79, 243,182,79, 242,184,79, 242,186,78, 241,188,78,
  241,190,78, 240,193,78, 240,195,77, 239,197,77, 238,199,77, 238,201,77, 237,203,77, 237,205,76,
  236,208,76, 235,210,76, 235,212,76, 234,214,76, 234,216,76, 233,218,76, 232,221,76, 232,223,76,
  231,225,77, 231,227,77, 230,229,77, 230,232,77, 229,234,78, 229,236,78, 229,238,79, 228,240,80,
  228,242,80, 228,244,81, 227,247,82, 227,249,84, 227,251,85, 228,253,87, 228,255,89, 229,255,91,
  230,255,93, 232,255,95, 233,255,98, 235,255,100, 237,255,103, 239,255,106, 241,255,110, 244,255,113,
  246,255,116, 249,255,120, 251,255,123, 254,255,127, 255,255,130, 255,255,134, 255,255,138, 255,255,142,
  255,255,146, 255,255,150, 255,255,154, 255,255,158, 255,255,162, 255,255,166, 255,255,170, 255,255,174,
  255,255,178, 255,255,182, 255,255,186, 255,255,190, 255,255,194, 255,255,198, 255,255,201, 255,255,205,
]);

function magma(v, out, off) {
  let i = Math.max(0, Math.min(255, (v * 255) | 0));
  i *= 3;
  out[off]   = MAGMA[i];
  out[off+1] = MAGMA[i+1];
  out[off+2] = MAGMA[i+2];
  out[off+3] = 255;
}

const winCache = new Map();
const winSumSqCache = new Map();
const winDerivCache = new Map();
function hann(n) {
  let w = winCache.get(n);
  if (w) return w;
  w = new Float32Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
    w[i] = v;
    sum += v;
  }
  winCache.set(n, w);
  winSumSqCache.set(n, sum * sum);
  // Analytic derivative of the Hann window — used as a "frequency-derivative
  // window" for Auger-Flandrin reassignment. dw/dn = (π/(N-1)) sin(2πn/(N-1)).
  const dw = new Float32Array(n);
  const k = Math.PI / (n - 1);
  for (let i = 0; i < n; i++) dw[i] = k * Math.sin((2 * Math.PI * i) / (n - 1));
  winDerivCache.set(n, dw);
  return w;
}
function hannSumSq(n) { if (!winSumSqCache.has(n)) hann(n); return winSumSqCache.get(n); }
function hannDeriv(n) { if (!winDerivCache.has(n)) hann(n); return winDerivCache.get(n); }

const fftCache = new Map();
function getFft(n) {
  let f = fftCache.get(n);
  if (f) return f;
  f = new FFT(n);
  fftCache.set(n, f);
  return f;
}

function frameMaxDb(audio, center, nFft, fft, win, input, output, winSumSq) {
  const s = center - (nFft >> 1);
  for (let i = 0; i < nFft; i++) {
    const idx = s + i;
    const v = (idx >= 0 && idx < audio.length) ? audio[idx] : 0;
    input[i*2] = v * win[i];
    input[i*2+1] = 0;
  }
  fft.transform(output, input);
  const nBins = nFft / 2 + 1;
  let maxDb = -1e9;
  for (let i = 0; i < nBins; i++) {
    const re = output[i*2], im = output[i*2+1];
    const power = (re*re + im*im) / winSumSq;
    const db = 10 * Math.log10(power + 1e-12);
    if (db > maxDb) maxDb = db;
  }
  return maxDb;
}

self.onmessage = (e) => {
  const m = e.data;
  if (m.op === "scan") {
    const { audio, nFft, samples } = m;
    const fft = getFft(nFft);
    const win = hann(nFft);
    const winSumSq = hannSumSq(nFft);
    const input = fft.createComplexArray();
    const output = fft.createComplexArray();
    let dbRef = -1e9;
    for (let k = 0; k < samples; k++) {
      const center = (nFft >> 1) + Math.floor((k / Math.max(1, samples - 1)) * (audio.length - nFft));
      const d = frameMaxDb(audio, center, nFft, fft, win, input, output, winSumSq);
      if (d > dbRef) dbRef = d;
    }
    self.postMessage({ op: "scan", dbRef });
    return;
  }

  if (m.op === "render") {
    const { token, target, audio, sampleRate, width, height,
            dbMax, fMin = 20, fMax = 18000, dbFloor = -55, gamma = 0.55,
            nFftPasses = [8192, 2048],
            tonalRatio = 4,
            cifLimit = 0.5,
            smoothAttenDb = 12,
          } = m;
    try {
      const logMin = Math.log2(fMin);
      const logMax = Math.log2(fMax);
      const acc = new Float32Array(width * height);     // additive accumulator
      const smoothAtten = Math.pow(10, -smoothAttenDb / 10);
      const haveMax = typeof dbMax === "number" && isFinite(dbMax);
      let frameMaxFallback = -1e9;

      // Smooth crossover weights between FFT passes (no hard band edges):
      //   centre = √(N_a × N_b) octaves, width = 1.5 octaves.
      // For N=[8192, 2048] crossover happens around f ≈ ~270 Hz with a smooth
      // sigmoid fade so bass-only and treble-only regions get all the weight
      // from the appropriate window, and the middle blends.
      // Tonal peaks have nFft-independent scaled power, so add-merging weighted
      // contributions that sum to 1 preserves full brightness across freq.
      function passWeight(pIdx, fHz) {
        if (nFftPasses.length === 1) return 1;
        // For our 2-pass case: pIdx 0 = large window (bass), pIdx 1 = small (treble).
        // Use a log-frequency sigmoid centered at the geometric-mean "appropriate" freq.
        // 8192-tap window at 44.1kHz handles down to ~20Hz cleanly; 2048 down to ~80Hz.
        // Set crossover at ~250-500Hz with a soft slope.
        const f = Math.max(1, fHz);
        const x = (Math.log2(f) - Math.log2(350)) / 1.5;   // octave-centered
        const sig = 1 / (1 + Math.exp(-4 * x));            // 0..1, monotonic with f
        return pIdx === 0 ? 1 - sig : sig;
      }

      for (let pIdx = 0; pIdx < nFftPasses.length; pIdx++) {
        const nFft = nFftPasses[pIdx];
        const fft = getFft(nFft);
        const win = hann(nFft);
        const dw  = hannDeriv(nFft);
        const winSumSq = hannSumSq(nFft);
        const input  = fft.createComplexArray();
        const inputD = fft.createComplexArray();
        const output  = fft.createComplexArray();
        const outputD = fft.createComplexArray();
        const nBins = nFft / 2 + 1;
        const binPerHz = nFft / sampleRate;

        // Precompute per-y: linear bin index AND this pass's weight at that freq
        const yToBin = new Float32Array(height);
        const yWeight = new Float32Array(height);
        for (let y = 0; y < height; y++) {
          const t = y / (height - 1);
          const f = Math.pow(2, logMax - t * (logMax - logMin));
          yToBin[y] = f * binPerHz;
          yWeight[y] = passWeight(pIdx, f);
        }

        const mags = new Float32Array(nBins);
        // Noise floor cutoffs scaled per pass (smaller N → higher per-bin noise)
        const nScale = nFftPasses[0] / nFft;
        const smoothFloor   = Math.pow(10, (dbFloor +  0) / 10) * nScale;
        const reassignFloor = Math.pow(10, (dbFloor + 20) / 10) * nScale;

        for (let x = 0; x < width; x++) {
          const center = Math.round((x + 0.5) * (audio.length / width));
          const s = center - (nFft >> 1);
          for (let i = 0; i < nFft; i++) {
            const idx = s + i;
            const v = (idx >= 0 && idx < audio.length) ? audio[idx] : 0;
            input[i*2]  = v * win[i]; input[i*2+1]  = 0;
            inputD[i*2] = v * dw[i];  inputD[i*2+1] = 0;
          }
          fft.transform(output,  input);
          fft.transform(outputD, inputD);

          for (let i = 0; i < nBins; i++) {
            const re = output[i*2], im = output[i*2+1];
            mags[i] = (re*re + im*im) / winSumSq;
            if (!haveMax) {
              const db = 10 * Math.log10(mags[i] + 1e-12);
              if (db > frameMaxFallback) frameMaxFallback = db;
            }
          }

          // (A) SMOOTH BACKDROP — weighted add (no hard cutoff)
          for (let y = 0; y < height; y++) {
            const w = yWeight[y];
            if (w < 1e-3) continue;
            const bf = yToBin[y];
            const i = bf | 0;
            const frac = bf - i;
            let p = (i + 1 < nBins) ? mags[i] * (1 - frac) + mags[i+1] * frac
                                    : mags[Math.min(i, nBins - 1)];
            if (p < smoothFloor) continue;
            // also scale by nScale^-1 so noise floor matches across passes
            acc[y * width + x] += p * smoothAtten * w / nScale;
          }

          // (B) REASSIGNED PEAKS — single pixel, tonal bins, weighted add
          for (let k = 2; k < nBins - 2; k++) {
            const power = mags[k];
            if (power < reassignFloor) continue;
            const neighborAvg = (mags[k-2] + mags[k-1] + mags[k+1] + mags[k+2]) * 0.25;
            if (power < tonalRatio * neighborAvg) continue;

            const re  = output[k*2],  im  = output[k*2+1];
            const reD = outputD[k*2], imD = outputD[k*2+1];
            const mag2 = re*re + im*im;
            const cif = -((imD*re - reD*im) / mag2) / (2 * Math.PI);
            if (Math.abs(cif) > cifLimit) continue;
            const binHat = k + cif * nFft;
            if (binHat <= 0 || binHat >= nBins) continue;
            const fHat = binHat * sampleRate / nFft;
            if (fHat < fMin || fHat > fMax) continue;
            const w = passWeight(pIdx, fHat);
            if (w < 1e-3) continue;
            const tNorm = (logMax - Math.log2(fHat)) / (logMax - logMin);
            const yHat = Math.round(tNorm * (height - 1));
            if (yHat < 0 || yHat >= height) continue;
            // Tonal-invariant: add weighted; weights sum to 1 → full brightness
            acc[yHat * width + x] += power * w;
          }
        }
      }

      const top = haveMax ? dbMax : frameMaxFallback;
      const range = top - dbFloor;
      const outRgba = new Uint8ClampedArray(width * height * 4);
      for (let i = 0; i < acc.length; i++) {
        const p = acc[i];
        if (p <= 0) { magma(0, outRgba, i * 4); continue; }
        const db = 10 * Math.log10(p + 1e-12);
        let v = (db - dbFloor) / range;
        if (v < 0) v = 0; else if (v > 1) v = 1;
        v = v > 0 ? Math.pow(v, gamma) : 0;
        magma(v, outRgba, i * 4);
      }

      self.postMessage({ op: "render", token, target, buf: outRgba.buffer, width, height }, [outRgba.buffer]);
    } catch (err) {
      self.postMessage({ op: "render", token, target, error: String(err) });
    }
  }
};
