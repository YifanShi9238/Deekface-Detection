import os
import numpy as np
import cv2
from scipy.fftpack import dct, idct
import argparse
from pathlib import Path

# ---------- DCT 2D ----------
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

# ---------- Face ROI / Guards ----------
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
    H,W = gray.shape[:2]
    s = min(H,W)
    return gray[(H-s)//2:(H+s)//2, (W-s)//2:(W+s)//2]

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

def ticker_graphics_score(gray):
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(gray, (3,3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    Ex = float(np.sum(np.abs(gx))) + 1e-8
    Ey = float(np.sum(np.abs(gy))) + 1e-8
    horiz_bias = Ex / (Ex + Ey)
    edges = cv2.Canny((g*1.0).astype(np.uint8), 80, 160)
    row_density = np.mean(np.sum(edges>0, axis=1) / (edges.shape[1] + 1e-8))
    s = 0.6*horiz_bias + 0.4*np.tanh(3.0*row_density)
    return float(np.clip(s, 0.0, 1.0))

# ---------- Spectral / Artifacts ----------
def center_low_high_stats(mag, frac=0.25):
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

def radial_peakiness(mag, nbins=24):
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
    benford = np.array([np.log10(1+1/d) for d in range(1,10)], dtype=np.float32)
    v = Cabs.flatten()
    v = v[v > 1e-8]
    if v.size < 256:
        return 0.0
    ld = (v / (10**np.floor(np.log10(v)))).astype(np.float32)
    counts = np.zeros(9, dtype=np.float32)
    for d in range(1,10):
        counts[d-1] = np.sum((ld >= d) & (ld < d+1))
    p = counts / (counts.sum() + 1e-8)
    return float(np.sum(np.abs(p - benford)))

# ---------- Core feature & score ----------
def image_features(img_bgr):
    gray = img_bgr if img_bgr.ndim==2 else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    roi  = face_roi(gray)
    roi  = cv2.resize(roi, (256,256), interpolation=cv2.INTER_AREA)

    F = np.fft.fftshift(np.fft.fft2(roi.astype(np.float32)))
    mag = np.abs(F) + 1e-8
    _, _, fast_hr, fast_sl = center_low_high_stats(mag, frac=0.25)

    g = roi.astype(np.float32)
    g = (g - g.mean())/(g.std()+1e-8)
    C = dct2(g)
    Cabs = np.abs(C)
    DC = float(Cabs[0,0]); AC=float(np.sum(Cabs)-DC)
    dct_dc_ac = DC/(AC+1e-8)
    h,w = Cabs.shape; r1=int(0.12*min(h,w)); r2=int(0.35*min(h,w))
    eL=float(np.sum(Cabs[:r1,:r1])); eM=float(np.sum(Cabs[r1:r2, r1:r2])); eH=float(np.sum(Cabs[r2:, r2:]))
    dct_hl = eH/(eL+1e-8); dct_ml = eM/(eL+1e-8)

    block = compression_blockiness(roi)
    gfx   = ticker_graphics_score(roi)

    peak = radial_peakiness(mag)
    grid = grid_peak_score(mag)
    benf = benford_dct_deviation(Cabs)

    return {
        "fast_hr": fast_hr,
        "fast_sl": fast_sl,
        "dct_dc_ac": dct_dc_ac,
        "dct_hl": dct_hl,
        "dct_ml": dct_ml,
        "block": block,
        "gfx": gfx,
        "peak": peak,
        "grid": grid,
        "benf": benf
    }

def image_fake_score(feat):
    block_m = feat["block"]; gfx_m = feat["gfx"]
    comp_weight = 1.0 / (1.0 + 6.0 * max(block_m, 0.0))
    gfx_weight  = 1.0 / (1.0 + 4.0 * max(gfx_m,   0.0))
    spectral_weight = comp_weight * gfx_weight
    T = np.tanh
    spectral = 0.20*T(feat["dct_hl"]) + 0.08*T(feat["dct_ml"]) + 0.04*T(feat["dct_dc_ac"]-0.5)
    spectral += 0.18*T(feat["fast_hr"]) + 0.05*T(abs(feat["fast_sl"]))
    spectral *= spectral_weight
    guard = -0.35*np.tanh(3.0*block_m) - 0.20*np.tanh(3.0*gfx_m)
    low_comp = float(block_m < 0.18)
    evidence = 0.40*low_comp*T(max(0.0,feat["peak"]-2.1)*1.6)
    evidence += 0.30*low_comp*T(max(0.0,feat["grid"]-0.11)*8.0)
    evidence += 0.30*low_comp*T(max(0.0,feat["benf"]-0.18)*6.0)
    raw = spectral + guard + evidence
    prob = 1.0/(1.0+np.exp(-raw))
    prob = 0.9*prob + 0.05
    return float(np.clip(prob,0.0,1.0))

# ---------- Compatibility Wrappers for app.py ----------
def frequency_analysis(frames):
    """
    Adapter for Streamlit app.
    Takes list of frames (images) and returns list of features like detection.py.
    """
    feats = []
    for frame in frames:
        feats.append(image_features(frame))
    return feats

def fake_score(features):
    """
    Adapter for Streamlit app.
    Takes feature list (usually 1 image) and averages scores.
    """
    if not features:
        return 0.5
    scores = [image_fake_score(f) for f in features]
    return float(np.mean(scores))
