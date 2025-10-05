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
#   SPEED/ROBUSTNESS HELPERS (do not change DCT/DFT math)
# ======================================================

# 1) face ROI (helps accuracy without touching transform math)
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
    # fallback: center crop
    H,W = gray.shape
    s = min(H,W)
    return gray[(H-s)//2:(H+s)//2, (W-s)//2:(W+s)//2]

# 2) compression guard: H.264-like blockiness vs total grad energy
def compression_blockiness(gray):
    H,W = gray.shape
    step = 16 if (H>=32 and W>=32) else 8
    grid = 0.0
    for y in range(step, H, step): grid += np.sum(np.abs(gray[y,:]-gray[y-1,:]))
    for x in range(step, W, step): grid += np.sum(np.abs(gray[:,x]-gray[:,x-1]))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradE = float(np.sum(np.abs(gx))+np.sum(np.abs(gy))) + 1e-8
    return float(grid/gradE)

# 3) central low/high split + slope proxy
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

# 4) robust reducers
def _agg(series):
    v = np.asarray(series, dtype=np.float32)
    mean = float(np.mean(v)) if v.size else 0.0
    jit  = float(np.mean(np.abs(np.diff(v)))) if v.size>1 else 0.0
    return mean, jit

# ======================================================
#   DFT/DCT FEATURE WRAPPERS (call your functions as-is)
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
    return float(dft_hr), float(dft_sl)

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
    return float(dct_dc_ac), float(dct_hl), float(dct_ml)

# =========================
#  PIPELINE (sampling + ROI)
# =========================

def frequency_analysis(frames, max_frames_for_dft=3, every_n_for_dft=20):
    """
    - Crop to face ROI, resize to 256x256 for FFT and DCT (linear crop doesn't alter DFT/DCT math).
    - Call your slow DFT2D on only K frames (default 3), spaced out.
    - Always compute fast FFT features + DCT features per frame.
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
        dct_dc_ac, dct_hl, dct_ml = dct_features_from_roi(roi)

        # Sparse DFT2D (your function)
        dft_hr = dft_sl = None
        if i in dft_ids:
            dft_hr, dft_sl = dft_features_from_roi(roi)

        # Compression guard
        blockiness = compression_blockiness(roi)

        feats.append({
            "fast_hr": float(fast_hr),
            "fast_sl": float(fast_sl),
            "dct_dc_ac": float(dct_dc_ac),
            "dct_hl": float(dct_hl),
            "dct_ml": float(dct_ml),
            "block": float(blockiness),
            "dft_hr": float(dft_hr) if dft_hr is not None else None,
            "dft_sl": float(dft_sl) if dft_sl is not None else None
        })
    return feats

# =========================
#  FUSION (probability)
# =========================

def fake_score(features):
    """
    Fuse DFT + DCT + FFT + guards into a probability.
    NOTE: We did not change your DFT/DCT math—only when we call them and how we fuse.
    """
    if not features: return 0.5

    get = lambda k: [f[k] for f in features if f[k] is not None]
    get_all = lambda k: [f[k] for f in features]

    fast_hr_m, fast_hr_j = _agg(get_all("fast_hr"))
    fast_sl_m, _         = _agg(get_all("fast_sl"))
    dct_dcac_m, _        = _agg(get_all("dct_dc_ac"))
    dct_hl_m,  dct_hl_j  = _agg(get_all("dct_hl"))
    dct_ml_m,  _         = _agg(get_all("dct_ml"))
    block_m,   _         = _agg(get_all("block"))

    dft_hr_vals = get("dft_hr")
    dft_sl_vals = get("dft_sl")
    dft_hr_m = float(np.mean(dft_hr_vals)) if dft_hr_vals else 0.0
    dft_sl_m = float(np.mean(dft_sl_vals)) if dft_sl_vals else 0.0

    T = np.tanh
    raw = 0.0
    # DCT (robust compression-domain cues)
    raw += 0.20 * T(dct_hl_m) + 0.10 * T(dct_ml_m)
    raw += 0.10 * T(dct_dcac_m - 0.5)
    # Fast FFT
    raw += 0.20 * T(fast_hr_m) + 0.05 * T(abs(fast_sl_m))
    # Sparse DFT2D (your function)
    raw += 0.10 * T(dft_hr_m) + 0.05 * T(abs(dft_sl_m))
    # Guards: compression + temporal jitter reduce score
    raw -= 0.25 * T(3.0 * block_m)
    raw -= 0.20 * T(5.0 * (dct_hl_j + fast_hr_j))

    prob = 1.0 / (1.0 + np.exp(-raw))
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

    frames = extract_frames(video_path, every_n_frames=100, resize=(256, 256))
    if not frames:
        print("Couldn’t extract frames.")
        raise SystemExit(1)

    # You can tune speed by passing max_frames_for_dft=0,1,2...
    feats = frequency_analysis(frames, max_frames_for_dft=3, every_n_for_dft=20)
    score = fake_score(feats)
    print(f"Fake probability: {score:.3f}")

