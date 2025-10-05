import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
from pathlib import Path

# --- Import teammates' functions safely ---
try:
    from utils import download_youtube_video, extract_frames   # Person A
    from detection import frequency_analysis, fake_score       # Person B
except Exception:
    download_youtube_video = None
    extract_frames = None
    frequency_analysis = None
    fake_score = None

# --- Streamlit Config ---
st.set_page_config(page_title="Frequency-Based Fake Detection", layout="wide")

# --- Title / Header ---
st.title("🎥 Frequency-Based Fake Detection")
st.markdown(
    "Upload a video or paste a YouTube link. We analyze frequency patterns in frames "
    "to estimate the likelihood of deepfake content."
)

# --- Sidebar ---
st.sidebar.header("⚙️ Options")
youtube_url = st.sidebar.text_input("YouTube URL")
uploaded_file = st.sidebar.file_uploader("Or upload a video file", type=["mp4", "mov"])
demo_mode = st.sidebar.checkbox("Use Demo Mode (no backend required)", value=True)
st.sidebar.info("👩‍💻 Built at HackUTA 2025\n\nTeam: Matteo + A + B + C")

# ---------- Compatibility wrappers (support teammate variants) ----------
def _call_download(url: str):
    if not download_youtube_video:
        return None
    try:
        # current variant
        return download_youtube_video(url, output_folder="Downloads")
    except TypeError:
        try:
            # alt variant with out_dir/filename
            return download_youtube_video(url, out_dir="Downloads", filename="input.mp4")
        except TypeError:
            # plain single-arg
            return download_youtube_video(url)

def _call_extract(video_path):
    """Call extract_frames regardless of teammate signature and normalize result."""
    if not extract_frames:
        return []

    p = str(video_path)
    result = None

    # Try our preferred signature
    try:
        result = extract_frames(p, every_n_frames=10, resize=(256, 256), max_frames=90)
    except TypeError:
        # Try teammate's fps_interval variant
        try:
            result = extract_frames(p, fps_interval=10)
        except TypeError:
            # Fallback positional-only
            result = extract_frames(p)

    # ---- Normalize to a list[np.ndarray] ----
    frames = []
    if result is None:
        frames = []
    elif isinstance(result, dict) and "frames" in result:
        frames = result["frames"]
    elif isinstance(result, tuple):
        # assume (frames, *extras)
        frames = result[0]
    else:
        frames = result

    # If numpy array of shape (N,H,W) or (N,H,W,C), convert to list
    if isinstance(frames, np.ndarray):
        if frames.ndim >= 3:
            frames = [frames[i] for i in range(frames.shape[0])]
        else:
            frames = [frames]

    # Ensure grayscale 2D arrays
    fixed = []
    for f in frames:
        if f is None:
            continue
        arr = np.asarray(f)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            import cv2
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        fixed.append(arr)
    return fixed

# ---------- Run Button ----------
if st.sidebar.button("Run Detection"):
    with st.spinner("Processing video..."):

        # Progress for UX polish
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress.progress(i + 1)

        # --- Demo path or missing modules ---
        if demo_mode or not (download_youtube_video and extract_frames and frequency_analysis and fake_score):
            score = 0.72
            frames = [np.random.randint(0, 255, (256, 256), dtype=np.uint8)]
            F = np.fft.fftshift(np.fft.fft2(frames[0].astype(np.float32)))
            spectrum = np.abs(F)

        else:
            # Acquire video path
            if youtube_url:
                video_path = _call_download(youtube_url)
            elif uploaded_file:
                # Save uploaded file locally
                with open("temp.mp4", "wb") as f:
                    f.write(uploaded_file.read())
                video_path = "temp.mp4"
            else:
                st.error("Please upload a video or provide a YouTube link.")
                st.stop()

            if not video_path or not os.path.exists(str(video_path)):
                st.error("Could not obtain or find a valid video file.")
                st.stop()

            # Extract frames (normalized to list)
            frames = _call_extract(video_path)

            # ---- Robust emptiness check (avoid ambiguous truth value) ----
            if frames is None or len(frames) == 0:
                st.error("Could not extract frames from the video.")
                st.stop()

            # Run detection
            feats = frequency_analysis(frames)
            score = fake_score(feats)

            # Spectrum from the first frame for visualization
            F = np.fft.fftshift(np.fft.fft2(frames[0].astype(np.float32)))
            spectrum = np.abs(F)

    # ---------- Results ----------
    st.success("Detection complete!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sample Frame")
        st.image(frames[0], clamp=True, caption="Extracted Frame (grayscale)")

    with col2:
        st.subheader("Frequency Spectrum (Log Magnitude)")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(np.log(spectrum + 1e-8), cmap="inferno")
        ax.axis("off")
        st.pyplot(fig)

    # Fake Score
    st.subheader("Detection Result")
    st.metric(label="Fake Probability", value=f"{score*100:.1f}%")
    st.progress(float(min(max(score, 0.0), 1.0)))

    if score > 0.5:
        st.error("Likely Deepfake")
    else:
        st.success("Likely Real")

    # Frequency Energy Breakdown (illustrative)
    st.subheader("Frequency Energy Breakdown")
    h, w = spectrum.shape
    low_q = spectrum[:h//2, :w//2]
    low_energy = float(np.sum(low_q))
    high_energy = float(np.sum(spectrum) - low_energy)
    energy_df = pd.DataFrame({"Low Frequency": [low_energy], "High Frequency": [high_energy]})
    st.bar_chart(energy_df.T)
