# detection.py
import os
import numpy as np
import cv2
from utils import extract_frames, download_youtube_video

# =========================
#   YOUR STACKOVERFLOW DCT
# =========================
# (unchanged)
from scipy.fftpack import dct, idct

def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

# =========================
#        YOUR DFT2D
# =========================
# (unchanged)
import PIL
import cmath

def DFT2D(image):
    data = np.asarray(image)
    M, N = image.size  # (img x, img y)
    dft2d = np.zeros((M, N), dtype=complex)
    for k in range(M):
        for l in range(N):
            sum_matrix = 0.0
            for m in range(M):
                for n in range(N):
                    e = cmath.exp(- 2j * np.pi * ((k * m) / M + (l * n) / N))
                    # use green channel (index 1) like your example
                    sum_matrix += data[m, n, 1] * e
            dft2d[k, l] = sum_matrix
    return dft2d

# ======================================================
#   HELPERS (do not change DCT/DFT math; just pre/post)
# ======================================================

# 1) Face ROI (linear crop doesn’t alter the transform math)
_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
def face_roi(gray):
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    faces = _face.detectMultiScale(gray, 1.2, 5, minSize=(80,80))
    if len(faces):
        x,y,w,h = max(faces, key=lambda b:b[2]*b[3])
        pad = int(0.15*max(w,h))
        y0,y1 = max(0,y-pad), min(gray.shape[0], y+h+pad)
        x0,x1 = max(0,x-pad), min(gray.shape[1], x+w+pad)
        return gray[y0:y1, x0:x1]
    # fallback: center crop if no face
    H,W = gray.shape
    s = min(H,W)
    return gray[(H-s)//2:(H+s)//2, (W-s)//2:(W+s)//2]

# 2) Compression guard: H.264-like blockiness vs total gradient energy
def compression_blockiness(gray):
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    H,W = gray.shape
    step = 16 if (H>=32 and W>=32) else 8
    grid = 0.0
    for y in range(step, H, step):
        grid += np.sum(np.abs(gray[y,:]-gray[y-1,:]))
    for x in range(step, W, step):
        grid += np.sum(np.abs(gray[:,x]-gray[:,x-1]))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradE = float(np.sum(np.abs(gx))+np.sum(np.abs(gy))) + 1e-8
    return float(grid/gradE)

# 3) News graphics/ticker guard (lower-thirds are horizontal-edge heavy)
def ticker_graphics_score(gray):
    """
    Heuristic ~[0..1], higher => more likely broadcast graphics present.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(gray, (3,3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    Ex = float(np.sum(np.abs(gx))) + 1e-8
    Ey = float(np.sum(np.abs(gy))) + 1e-8
    horiz_bias = Ex / (Ex + Ey)        # >0.5 ⇒ more horizontal structure
    edges = cv2.Canny((g*1.0).astype(np.uint8), 80, 160)
    row_density = np.mean(np.sum(edges>0, axis=1) / (edges.shape[1] + 1e-8))
    s = 0.6*horiz_bias + 0.4*np.tanh(3.0*row_density)
    return float(np.clip(s, 0.0, 1.0))

# 4) Central low/high split + slope proxy (shared by FFT and your DFT2D)
def _center_low_high_stats(mag, frac=0.25):
    h,w = mag.shape
    cy,cx = h//2, w//2
    r = int(min(cy,cx)*frac)
    low = mag[cy-r:cy+r, cx-r:cx+r]
    lowE = float(np.sum(low)); totE=float(np.sum(mag))
    highE = max(totE-lowE, 0.0)
    high_ratio = highE/(lowE+1e-8)
    logm = np.log(mag+1e-8); gy,gx = np.gradient(logm)
    slope = float(np.mean(np.abs(gy))+np.mean(np.abs(gx)))
    return lowE, highE, high_ratio, slope

# 5) Robust reducers
def _agg(series):
    v = np.asarray(series, dtype=np.float32)
    mean = float(np.mean(v)) if v.size else 0.0
    jit  = float(np.mean(np.abs(np.diff(v)))) if v.size>1 else 0.0
    return mean, jit

# ======================================================
#      HIGH-PRECISION ARTIFACT CUES (evidence triggers)
# ======================================================

def radial_peakiness(mag, nbins=24):
    """
    Radial power spectral density peakiness: max_bin / median_bin.
    >~2.2 is suspicious on clean clips (tune as needed).
    """
    h, w = mag.shape
    cy, cx = h//2, w//2
    y = np.arange(h) - cy
    x = np.arange(w) - cx
    rr = np.sqrt((y[:,None]**2) + (x[None,:]**2))
    rmax = rr.max()
    bins = np.linspace(0, rmax, nbins+1)
    psd = []
    for i in range(nbins):
        mask = (rr >= bins[i]) & (rr < bins[i+1])
        if np.any(mask):
            psd.append(float(np.mean(mag[mask])))
    if len(psd) < 3:
        return 1.0
    psd = np.array(psd)
    med = np.median(psd[1:-1]) + 1e-8
    return float(np.max(psd) / med)

def grid_peak_score(mag, k=4):
    """
    Energy concentration at grid harmonics (8x8 style).
    Return sum(top-k sampled peaks) / total (coarse).
    """
    h, w = mag.shape
    step_y = max(h//16, 2)
    step_x = max(w//16, 2)
    sub = mag[::step_y, ::step_x].copy()
    flat = sub.ravel()
    if flat.size < k+5:
        return 0.0
    idx = np.argpartition(flat, -k)[-k:]
    peak = float(np.sum(flat[idx]))
    total = float(np.sum(flat)) + 1e-8
    return float(peak / total)

def benford_dct_deviation(Cabs):
    """
    Benford deviation on |DCT| leading digits (excluding zeros/DC).
    Return sum of abs diff from Benford distribution (higher => suspicious).
    """
    benford = np.array([np.log10(1+1/d) for d in range(1,10)], dtype=np.float32)
    v = Cabs.flatten()
    v = v[v > 1e-8]
    if v.size < 256:
        return 0.0
    ld = (v / (10**np.floor(np.log10(v)))).astype(np.float32)  # leading digit value 1..9 region
    counts = np.zeros(9, dtype=np.float32)
    for d in range(1,10):
        counts[d-1] = np.sum((ld >= d) & (ld < d+1))
    p = counts / (counts.sum() + 1e-8)
    return float(np.sum(np.abs(p - benford)))

# ======================================================
#   FEATURE WRAPPERS (call your DCT/DFT exactly as-is)
# ======================================================

def dft_features_from_roi(roi_256):
    """
    Call your DFT2D on a 50x50 RGB version of the ROI to keep runtime reasonable.
    """
    small = cv2.resize(roi_256, (50,50), interpolation=cv2.INTER_AREA)
    rgb   = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
    pil   = PIL.Image.fromarray(rgb)
    dft   = DFT2D(pil)                        # your unchanged O(N^4)
    dmag  = np.abs(dft).astype(np.float32)
    _, _, dft_hr, dft_sl = _center_low_high_stats(dmag, frac=0.25)
    return float(dft_hr), float(dft_sl), dmag

def dct_features_from_roi(roi_256):
    """
    Call your dct2 on a normalized 256x256 ROI.
    """
    g = roi_256.astype(np.float32)
    g = (g - g.mean())/(g.std()+1e-8)
    C = dct2(g)                           # your DCT
    Cabs = np.abs(C)
    DC = float(Cabs[0,0]); AC=float(np.sum(Cabs)-DC)
    dct_dc_ac = DC/(AC+1e-8)
    h,w = Cabs.shape; r1=int(0.12*min(h,w)); r2=int(0.35*min(h,w))
    eL=float(np.sum(Cabs[:r1,:r1])); eM=float(np.sum(Cabs[r1:r2, r1:r2])); eH=float(np.sum(Cabs[r2:, r2:]))
    dct_hl = eH/(eL+1e-8); dct_ml = eM/(eL+1e-8)
    return float(dct_dc_ac), float(dct_hl), float(dct_ml), Cabs

# =========================
#  PIPELINE (sampling + ROI)
# =========================

def frequency_analysis(frames, max_frames_for_dft=3, every_n_for_dft=20):
    """
    - Crop to face ROI, resize to 256x256 for FFT and DCT (linear crop doesn't alter transform math).
    - Call your slow DFT2D on only K frames (default 3), spaced out.
    - Always compute fast FFT features + DCT features per frame.
    - ALSO compute artifact cues (peakiness, gridness, Benford).
    """
    if not frames: return []

    feats = []
    N = len(frames)
    dft_ids = list(range(0, N, every_n_for_dft))[:max_frames_for_dft]

    for i, frame in enumerate(frames):
        gray = frame if frame.ndim==2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi  = face_roi(gray)
        roi  = cv2.resize(roi, (256,256), interpolation=cv2.INTER_AREA)

        # Fast FFT on ROI
        F = np.fft.fftshift(np.fft.fft2(roi.astype(np.float32)))
        mag = np.abs(F) + 1e-8
        _, _, fast_hr, fast_sl = _center_low_high_stats(mag, frac=0.25)

        # DCT (your dct2)
        dct_dc_ac, dct_hl, dct_ml, Cabs = dct_features_from_roi(roi)

        # Sparse DFT2D (your function)
        dft_hr = dft_sl = None
        dmag_for_artifacts = None
        if i in dft_ids:
            dft_hr, dft_sl, dmag_for_artifacts = dft_features_from_roi(roi)

        # Guards
        blockiness = compression_blockiness(roi)
        gfx = ticker_graphics_score(roi)

        # Artifact cues (use FFT magnitude; if DFT magnitude exists, max of both)
        peak_fft = radial_peakiness(mag)
        grid_fft = grid_peak_score(mag)
        peak = peak_fft
        grid = grid_fft
        if dmag_for_artifacts is not None:
            peak = max(peak, radial_peakiness(dmag_for_artifacts))
            grid = max(grid, grid_peak_score(dmag_for_artifacts))
        benf = benford_dct_deviation(Cabs)

        feats.append({
            "fast_hr": float(fast_hr),
            "fast_sl": float(fast_sl),
            "dct_dc_ac": float(dct_dc_ac),
            "dct_hl": float(dct_hl),
            "dct_ml": float(dct_ml),
            "block": float(blockiness),
            "gfx": float(gfx),
            "dft_hr": float(dft_hr) if dft_hr is not None else None,
            "dft_sl": float(dft_sl) if dft_sl is not None else None,
            "peak": float(peak),
            "grid": float(grid),
            "benf": float(benf)
        })
    return feats

# =========================
#  FUSION (probability)
# =========================

def fake_score(features):
    """
    Compression/graphics/jitter-aware fusion + artifact evidence triggers.
    DFT/DCT math is unchanged; only fusion/weights adapt to context.
    """
    if not features: return 0.5

    get = lambda k: [f[k] for f in features if f[k] is not None]  # for sparse DFT
    get_all = lambda k: [f[k] for f in features]

    def agg_all(name):
        v = np.asarray(get_all(name), dtype=np.float32)
        if v.size == 0: return 0.0, 0.0
        return float(np.mean(v)), float(np.mean(np.abs(np.diff(v)))) if v.size>1 else 0.0

    def agg_some(name):
        v = np.asarray(get(name), dtype=np.float32)
        if v.size == 0: return 0.0, 0.0
        return float(np.mean(v)), float(np.mean(np.abs(np.diff(v)))) if v.size>1 else 0.0

    fast_hr_m, fast_hr_j = agg_all("fast_hr")
    fast_sl_m, _         = agg_all("fast_sl")
    dct_dcac_m, _        = agg_all("dct_dc_ac")
    dct_hl_m,  dct_hl_j  = agg_all("dct_hl")
    dct_ml_m,  _         = agg_all("dct_ml")
    block_m,   _         = agg_all("block")
    gfx_m,     _         = agg_all("gfx")

    dft_hr_m, _ = agg_some("dft_hr")
    dft_sl_m, _ = agg_some("dft_sl")

    # NEW artifact cues
    peak_m, _  = agg_all("peak")
    grid_m, _  = agg_all("grid")
    benf_m, _  = agg_all("benf")

    # Adaptive spectral trust (compression/news lower-thirds)
    comp_weight = 1.0 / (1.0 + 6.0 * max(block_m, 0.0))
    gfx_weight  = 1.0 / (1.0 + 4.0 * max(gfx_m,   0.0))
    spectral_weight = comp_weight * gfx_weight

    # Jitter bonus (real videos): stronger
    jitter_term = np.tanh(6.0 * (dct_hl_j + fast_hr_j))
    T = np.tanh

    # Base spectral score (conservative)
    spectral = 0.0
    spectral += 0.18 * T(dct_hl_m) + 0.08 * T(dct_ml_m) + 0.04 * T(dct_dcac_m - 0.5)
    spectral += 0.16 * T(fast_hr_m) + 0.05 * T(abs(fast_sl_m))
    spectral += 0.08 * T(dft_hr_m)  + 0.04 * T(abs(dft_sl_m))
    spectral *= spectral_weight

    # Guards (reduce score on compressed/news & reward natural jitter)
    guard = 0.0
    guard -= 0.35 * np.tanh(3.0 * block_m)
    guard -= 0.20 * np.tanh(3.0 * gfx_m)
    guard -= 0.28 * jitter_term

    # Evidence boost (only when compression is LOW), strict thresholds
    low_comp = float(block_m < 0.18)
    evidence = 0.0
    evidence += low_comp * 0.35 * T(max(0.0, peak_m - 2.2) * 1.5)
    evidence += low_comp * 0.25 * T(max(0.0, grid_m - 0.12) * 8.0)
    evidence += low_comp * 0.25 * T(max(0.0, benf_m - 0.20) * 6.0)

    raw = spectral + guard + evidence

    prob = 1.0 / (1.0 + np.exp(-raw))
    # keep probabilities reasonable
    prob = 0.90 * prob + 0.05
    return float(np.clip(prob, 0.0, 1.0))

# =========================
#           CLI
# =========================

if __name__ == "__main__":
    # default to newest ~/Downloads *.mp4 if user hits Enter
    downloads = os.path.expanduser("~/Downloads")
    candidates = [os.path.join(downloads, f) for f in os.listdir(downloads) if f.lower().endswith(".mp4")]
    candidates.sort(key=os.path.getmtime, reverse=True)
    default_path = candidates[0] if candidates else ""
    prompt = f"Enter video file path or YouTube URL [{default_path}]: " if default_path else "Enter video file path or YouTube URL: "
    inp = input(prompt).strip()
    video_path = inp or default_path

    if video_path.startswith("http"):
        print("Downloading…")
        video_path = download_youtube_video(video_path, output_folder="Downloads") or ""

    if not video_path or not os.path.exists(video_path):
        print("File not found.")
        raise SystemExit(1)

    frames = extract_frames(video_path, every_n_frames=10, resize=(256, 256))
    if not frames:
        print("Couldn’t extract frames.")
        raise SystemExit(1)

    # You can tune speed by passing max_frames_for_dft=0,1,2...
    feats = frequency_analysis(frames, max_frames_for_dft=3, every_n_for_dft=20)
    score = fake_score(feats)
    print(f"Fake probability: {score:.3f}")
    if score < 0.40:
        print("→ Likely REAL")
    elif score > 0.60:
        print("→ Likely FAKE")
    else:
        print("→ UNCERTAIN (needs higher-res or different clip)")
