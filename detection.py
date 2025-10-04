import os
import numpy as np
import cv2
from utils import extract_frames, download_youtube_video

def frequency_analysis(frames):
    features = []
    if not frames:
        return features
    for frame in frames:
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f = np.fft.fftshift(np.fft.fft2(frame.astype(np.float32)))
        mag = np.abs(f) + 1e-8
        h, w = frame.shape
        cy, cx = h // 2, w // 2
        r = int(min(cy, cx) * 0.25)
        low = mag[cy-r:cy+r, cx-r:cx+r]
        low_energy = float(np.sum(low))
        total_energy = float(np.sum(mag))
        high_energy = total_energy - low_energy
        high_ratio = high_energy / (low_energy + 1e-8)
        logm = np.log(mag)
        gy, gx = np.gradient(logm)
        slope = float(np.mean(np.abs(gy)) + np.mean(np.abs(gx)))
        features.append({"high_freq_ratio": high_ratio, "slope": slope})
    return features

def fake_score(features):
    if not features: return 0.5
    ratios = np.array([f["high_freq_ratio"] for f in features], dtype=np.float32)
    slopes = np.array([f["slope"] for f in features], dtype=np.float32)
    ratio_std = float(np.std(ratios))
    slope_mean = float(np.mean(slopes))
    realness = ratio_std * 0.6 + slope_mean * 0.4
    return float(np.clip(1 - realness / (realness + 1), 0, 1))

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

    feats = frequency_analysis(frames)
    score = fake_score(feats)
    print(f"Fake probability: {score:.3f}")
