import numpy as np
import cv2

def frequency_analysis(frames):
    """Compute FFT-based features per frame."""
    features = []
    if not frames:
        return features

    for frame in frames:
        # ensure grayscale 2D
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2D FFT -> shift -> magnitude
        f = np.fft.fftshift(np.fft.fft2(frame.astype(np.float32)))
        mag = np.abs(f) + 1e-8  # avoid log(0)

        h, w = frame.shape
        cy, cx = h // 2, w // 2
        r = int(min(cy, cx) * 0.25)  # low-freq radius

        low = mag[cy - r: cy + r, cx - r: cx + r]
        low_energy = float(np.sum(low))
        total_energy = float(np.sum(mag))
        high_energy = total_energy - low_energy
        high_ratio = high_energy / (low_energy + 1e-8)

        # spectral slope proxy
        logm = np.log(mag)
        gy, gx = np.gradient(logm)
        slope = float(np.mean(np.abs(gy)) + np.mean(np.abs(gx)))

        features.append({
            "high_freq_ratio": high_ratio,
            "slope": slope
        })

    return features


def fake_score(features):
    """Naive fake probability from frequency irregularities."""
    if not features:
        return 0.5  # neutral if no data

    ratios = np.array([f["high_freq_ratio"] for f in features], dtype=np.float32)
    slopes = np.array([f["slope"] for f in features], dtype=np.float32)

    ratio_std = float(np.std(ratios))
    slope_mean = float(np.mean(slopes))

    realness = ratio_std * 0.6 + slope_mean * 0.4
    fake_prob = float(np.clip(1.0 - realness / (realness + 1.0), 0.0, 1.0))
    return fake_prob


if __name__ == "__main__":
    import os
    from utils import extract_frames, download_youtube

    video_path = input("Enter video file path or YouTube URL: ").strip()

    if video_path.startswith("http"):
        print("Downloading...")
        video_path = download_youtube(video_path) or ""
    if not os.path.exists(video_path):
        print("File not found or download failed.")
        exit(1)

    frames = extract_frames(video_path, every_n_frames=10, resize=(256,256))
    if not frames:
        print("Couldn’t extract frames from the video.")
        exit(1)

    feats = frequency_analysis(frames)
    score = fake_score(feats)
    print(f"Fake probability: {score:.3f}")


