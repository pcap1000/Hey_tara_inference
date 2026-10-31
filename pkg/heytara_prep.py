"""Wake-word preprocessing: WAV bytes -> fixed-size log-mel features (numpy/scipy only: no torch,
so worker processes stay light).

Pipeline per clip:
  decode -> trim silence -> drop glitches -> loudness-normalise -> right-align in a fixed window
  (over a low noise floor instead of digital silence) -> [optional augmentation] -> log-mel (dB) -> store.

Storage: float16 dB by default (no range clipping, ~0.03 dB resolution), or uint8 (4x smaller than float32).
Window length, mel count, storage type and chunk size can be set with environment variables
HEYTARA_WIN_S, HEYTARA_N_MELS, HEYTARA_STORE, HEYTARA_MEL_CHUNK (read at import time, so set them before
importing and before starting worker processes).
"""
import io
import os
import zlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from scipy.fft import rfft
from scipy.signal import fftconvolve


# ------------------------------------------------------------------ config
SR = 16000
WIN_S = float(os.environ.get("HEYTARA_WIN_S", 2.0))            # model input window
WIN = int(SR * WIN_S)                                           # 32000 samples for 2.0 s
TAIL_S = 0.2                                                    # silence kept after the word (streaming-like)
TAIL = int(SR * TAIL_S)

HOP, N_FFT, WIN_LEN, FMIN, FMAX = 160, 512, 400, 20.0, 7600.0
N_MELS = int(os.environ.get("HEYTARA_N_MELS", 40))
N_FRAMES = 1 + WIN // HOP                                       # 201 for 2.0 s
STORE_DEFAULT = os.environ.get("HEYTARA_STORE", "float16")      # "float16" | "uint8" | "float32"
DB_MIN, DB_MAX = -80.0, 40.0                                    # only used by the uint8 store
MEL_CHUNK = int(os.environ.get("HEYTARA_MEL_CHUNK", 128))       # clips per FFT chunk (memory vs speed)

TARGET_RMS_DB = -20.0             # speech loudness after normalisation
PEAK_LIMIT = 0.95
FLOOR_DB = -75.0                  # noise floor (dBFS) replacing digital silence

TRIM_REL_DB = 35.0                # speech = frames within 35 dB of the loudest frame ...
TRIM_ABS_DB = -60.0               # ... and above -60 dBFS
LEAD_MARGIN_S = 0.05
TRAIL_MARGIN_S = 0.10
MIN_SPEECH_S, MAX_SPEECH_S, MAX_DUR_S = 0.20, 2.50, 4.00        # outlier filters (TTS glitches)

# offline augmentation: train copies, a noisy test copy, and a harder noisy test copy
AUG_TRAIN = dict(p_reverb=0.30, p_noise=0.50, snr=(0.0, 20.0), p_real_bg=0.5, rt60=(0.15, 0.60))
AUG_NOISY = dict(p_reverb=0.50, p_noise=1.00, snr=(5.0, 15.0), p_real_bg=0.5, rt60=(0.15, 0.60))
AUG_HARD = dict(p_reverb=0.70, p_noise=1.00, snr=(-5.0, 5.0), p_real_bg=0.5, rt60=(0.30, 0.80))


# ------------------------------------------------------------------ helpers
def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-9)


def decode(b):
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        raise ValueError(f"unexpected sample rate {sr}")
    return x


def frame_db(x, frame=HOP):
    n = len(x) // frame
    if n == 0:
        return np.array([-120.0])
    f = x[: n * frame].reshape(n, frame)
    return 10 * np.log10(np.mean(f.astype(np.float64) ** 2, axis=1) + 1e-12)


def trim(x):
    """Energy VAD. Returns (segment with margins, info)."""
    db = frame_db(x)
    thr = max(db.max() - TRIM_REL_DB, TRIM_ABS_DB)
    act = np.where(db > thr)[0]
    info = dict(orig_dur=len(x) / SR)
    if len(act) == 0:
        info.update(lead_s=np.nan, trail_s=np.nan, speech_s=0.0)
        return x, info
    a, b = act[0] * HOP, (act[-1] + 1) * HOP
    info.update(lead_s=a / SR, trail_s=(len(x) - b) / SR, speech_s=(b - a) / SR)
    a = max(0, a - int(LEAD_MARGIN_S * SR))
    b = min(len(x), b + int(TRAIL_MARGIN_S * SR))
    return x[a:b], info


def normalise(seg):
    """Scale speech to TARGET_RMS_DB, then peak-limit. Returns (seg, gain_db)."""
    gain_db = TARGET_RMS_DB - rms_db(seg)
    y = seg * 10 ** (gain_db / 20)
    pk = np.abs(y).max()
    if pk > PEAK_LIMIT:
        y = y * (PEAK_LIMIT / pk)
        gain_db += 20 * np.log10(PEAK_LIMIT / pk)
    return y, gain_db


def place(seg, rng):
    """Right-align seg in a WIN-sample window (word ends TAIL before the end); fill with a noise floor.
    Over-long segments keep their END (what a streaming detector sees). Returns (window, cropped)."""
    room = WIN - TAIL
    cropped = len(seg) > room
    if cropped:
        seg = seg[-room:]
    w = rng.normal(0, 10 ** (FLOOR_DB / 20), WIN).astype(np.float32)
    end = WIN - TAIL
    w[end - len(seg): end] += seg
    return w, cropped


def colored_noise(n, beta, rng):
    """1/f^beta noise (0 white, 1 pink, 2 brown), unit RMS."""
    spec = np.fft.rfft(rng.normal(size=n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    y = np.fft.irfft(spec / f ** (beta / 2), n)
    return (y / (np.sqrt(np.mean(y ** 2)) + 1e-9)).astype(np.float32)


def make_rir(rt60, rng):
    L = int(rt60 * SR)
    t = np.arange(L) / SR
    tail = rng.normal(size=L) * 10 ** (-3 * t / rt60)           # -60 dB at rt60
    tail *= np.sqrt(rng.uniform(0.1, 0.5) / np.sum(tail ** 2))  # tail-to-direct energy ratio
    h = tail
    h[0] += 1.0
    return h.astype(np.float32)


def reverb(x, rt60, rng):
    y = fftconvolve(x, make_rir(rt60, rng))[: len(x)]
    return (y * (np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-9))).astype(np.float32)


def bank_noise(bank, n, rng):
    """Random crop of a real background clip (tiled if short), unit RMS."""
    data, offs = bank
    i = rng.integers(len(offs) - 1)
    c = data[offs[i]: offs[i + 1]].astype(np.float32) / 32768.0
    if len(c) < n:
        c = np.tile(c, int(np.ceil(n / len(c))))
    s = rng.integers(0, len(c) - n + 1)
    c = c[s: s + n]
    return c / (np.sqrt(np.mean(c ** 2)) + 1e-9)


def augment(w, speech_rms, cfg, bank, rng):
    """Reverb then additive noise on the fixed window. Returns (window, aug_info)."""
    info = dict(aug_reverb=False, aug_noise="none", snr_db=np.nan, rt60=np.nan)
    if rng.random() < cfg["p_reverb"]:
        info["rt60"] = float(rng.uniform(*cfg["rt60"]))
        w = reverb(w, info["rt60"], rng)
        info["aug_reverb"] = True
    if rng.random() < cfg["p_noise"]:
        if bank is not None and rng.random() < cfg["p_real_bg"]:
            n, info["aug_noise"] = bank_noise(bank, len(w), rng), "real_bg"
        else:
            n, info["aug_noise"] = colored_noise(len(w), rng.uniform(0, 2), rng), "colored"
        info["snr_db"] = float(rng.uniform(*cfg["snr"]))
        w = w + n * (speech_rms / 10 ** (info["snr_db"] / 20))
    pk = np.abs(w).max()
    if pk > 0.99:
        w = w * (0.99 / pk)
    return w.astype(np.float32), info


# ------------------------------------------------------------------ features
def _mel_filterbank():
    """HTK-scale triangular filterbank, shape (N_FFT//2+1, N_MELS) (same as torchaudio's default)."""
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    freqs = np.linspace(0, SR / 2, N_FFT // 2 + 1)
    pts = mel2hz(np.linspace(hz2mel(FMIN), hz2mel(FMAX), N_MELS + 2))
    slopes = pts[None, :] - freqs[:, None]
    down = -slopes[:, :-2] / np.diff(pts)[:-1]
    up = slopes[:, 2:] / np.diff(pts)[1:]
    return np.maximum(0.0, np.minimum(down, up)).astype(np.float32)


_FB = _mel_filterbank()
_STFT_WIN = np.zeros(N_FFT, np.float32)          # periodic Hann of length WIN_LEN, centred in the FFT frame
_STFT_WIN[(N_FFT - WIN_LEN) // 2:(N_FFT - WIN_LEN) // 2 + WIN_LEN] = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN_LEN) / WIN_LEN)
_FRAME_IDX = np.arange(N_FRAMES)[:, None] * HOP + np.arange(N_FFT)[None, :]


def log_mel_db(wins):
    """(B, WIN) float32 -> (B, N_MELS, N_FRAMES) dB float32. Matches torchaudio
    MelSpectrogram(center=True, reflect padding, power=2) to <1e-3 dB."""
    wins = np.asarray(wins, np.float32)
    out = np.empty((len(wins), N_MELS, N_FRAMES), np.float32)
    for a in range(0, len(wins), MEL_CHUNK):
        x = np.pad(wins[a:a + MEL_CHUNK], ((0, 0), (N_FFT // 2, N_FFT // 2)), mode="reflect")
        frames = x[:, _FRAME_IDX] * _STFT_WIN                                # (b, T, N_FFT) float32
        spec = rfft(frames, axis=-1)                                         # complex64 (float32 in -> float32 out)
        mel = np.matmul(spec.real ** 2 + spec.imag ** 2, _FB)                # (b, T, N_MELS)
        out[a:a + MEL_CHUNK] = (10 * np.log10(np.maximum(mel, 1e-10))).transpose(0, 2, 1)
    return out


def quantise(db, store=None):
    """dB float32 -> stored array. float16: dB as-is (floor is -100 dB from the clamp above), uint8: 0.47 dB steps over -80..+40."""
    store = store or STORE_DEFAULT
    if store == "uint8":
        return np.clip(np.round((db - DB_MIN) / (DB_MAX - DB_MIN) * 255), 0, 255).astype(np.uint8)
    if store == "float16":
        return db.astype(np.float16)
    return db.astype(np.float32)


def dequantise(q):
    """Stored array (any supported dtype) -> dB float32."""
    q = np.asarray(q)
    if q.dtype == np.uint8:
        return q.astype(np.float32) / 255 * (DB_MAX - DB_MIN) + DB_MIN
    return q.astype(np.float32)


# ------------------------------------------------------------------ per-clip
def process_clip(b, rng, cfg=None, bank=None):
    """WAV bytes -> (window float32 or None, meta dict). cfg=None means no augmentation."""
    x = decode(b)
    meta = dict(pre_rms_db=float(rms_db(x)), pre_peak=float(np.abs(x).max()),
                clipped_frac=float((np.abs(x) >= 0.999).mean()))
    seg, info = trim(x)
    meta.update(info)
    if info["speech_s"] < MIN_SPEECH_S:
        return None, {**meta, "drop": "too_short_or_silent"}
    if info["speech_s"] > MAX_SPEECH_S or info["orig_dur"] > MAX_DUR_S:
        return None, {**meta, "drop": "too_long"}
    seg, gain_db = normalise(seg)
    meta["gain_db"] = float(gain_db)
    w, meta["cropped"] = place(seg, rng)
    if cfg is not None:
        w, a = augment(w, 10 ** (TARGET_RMS_DB / 20), cfg, bank, rng)
        meta.update(a)
    return w, meta


def load_bank(path):
    z = np.load(path)
    return z["data"], z["offs"]


# ------------------------------------------------------------------ shard worker
KEEP_COLS = ["filename", "split", "label", "text", "category", "engine", "voice_id", "gender", "accent",
             "speed_name", "duration_s"]


def process_shard(path, out_dir, variants, seed, bank_path=None, batch=256, store=None):
    """Process one parquet shard.
    variants: dict name -> aug cfg or None, e.g. {"clean": None, "noisy": AUG_NOISY}. Each variant gets its own
    random stream (seeded by its name), so two variants with the same cfg are independent realisations.
    Writes feat_<variant>_<shard>.npy and meta_<variant>_<shard>.parquet (row-aligned), plus dropped_<shard>.parquet."""
    name = os.path.basename(path).replace(".parquet", "")
    bank = load_bank(bank_path) if bank_path else None
    pf = pq.ParquetFile(path)
    feats = {v: [] for v in variants}
    metas = {v: [] for v in variants}
    dropped = []
    gidx = 0
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg).to_pandas()
        wins = {v: [] for v in variants}
        augs = {v: [] for v in variants}
        kept = []
        for _, r in df.iterrows():
            base = {c: r[c] for c in KEEP_COLS if c in df.columns}
            base["shard"] = name
            rng = np.random.default_rng([seed, gidx, zlib.crc32(name.encode())])
            w, meta = process_clip(r["audio"]["bytes"], rng, None, bank)
            if w is None:
                dropped.append({**base, **meta})
                gidx += 1
                continue
            kept.append({**base, **meta})
            for v, cfg in variants.items():
                if cfg is None:
                    wins[v].append(w)
                else:                                   # augment the same normalised window, independent stream per variant
                    a_rng = np.random.default_rng([seed, gidx, 7, zlib.crc32(v.encode())])
                    aw, a = augment(w, 10 ** (TARGET_RMS_DB / 20), cfg, bank, a_rng)
                    wins[v].append(aw)
                    augs[v].append(a)
            gidx += 1
        for v in variants:
            if wins[v]:
                arr = np.stack(wins[v])
                feats[v].append(np.concatenate([quantise(log_mel_db(arr[i:i + batch]), store)
                                                for i in range(0, len(arr), batch)]))
                m = pd.DataFrame(kept)
                if augs[v]:
                    m = pd.concat([m, pd.DataFrame(augs[v]).reset_index(drop=True)], axis=1)
                metas[v].append(m)
    for v in variants:
        np.save(os.path.join(out_dir, f"feat_{v}_{name}.npy"), np.concatenate(feats[v]))
        pd.concat(metas[v], ignore_index=True).to_parquet(os.path.join(out_dir, f"meta_{v}_{name}.parquet"))
    pd.DataFrame(dropped).to_parquet(os.path.join(out_dir, f"dropped_{name}.parquet"))
    return name, sum(len(f) for f in feats[next(iter(variants))]), len(dropped)


# ------------------------------------------------------------------ real recordings: streaming windows
def stream_windows(b, stride_s=0.25):
    """WAV bytes of a longer, noisy real clip -> (K, WIN) float32 windows (or None if silent).
    The word position is unknown, so instead of trimming we slide the model window over the clip
    (as a deployed detector does) and the caller takes the MAX score over windows."""
    x = decode(b)
    db = frame_db(x)
    act = db > max(db.max() - TRIM_REL_DB, TRIM_ABS_DB)
    if act.sum() == 0:
        return None
    gain_db = TARGET_RMS_DB - 10 * np.log10(np.mean(10 ** (db[act] / 10)) + 1e-12)
    y = x * 10 ** (gain_db / 20)
    if np.abs(y).max() > PEAK_LIMIT:
        y = y * (PEAK_LIMIT / np.abs(y).max())
    if len(y) < WIN:                                   # short clip: right-align over a noise floor
        w = np.random.default_rng(0).normal(0, 10 ** (FLOOR_DB / 20), WIN).astype(np.float32)
        w[WIN - len(y):] += y
        return w[None]
    starts = list(range(0, len(y) - WIN + 1, int(stride_s * SR)))
    if starts[-1] != len(y) - WIN:
        starts.append(len(y) - WIN)
    return np.stack([y[s:s + WIN] for s in starts]).astype(np.float32)
