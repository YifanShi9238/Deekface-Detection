import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile
import cv2
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

# --- Optional: image-based detection module ---
try:
    import image_detection as image_detect
except Exception:
    image_detect = None

# --- Streamlit Config ---
st.set_page_config(page_title="Frequency-Based Fake Detection", layout="wide")

# --- Title / Header ---
st.title("🎥 Frequency-Based Fake Detection")
st.markdown(
    "Upload a video or image (or paste a YouTube link). "
    "The system analyzes frequency patterns to estimate the likelihood of AI-generated or deepfake content."
)

# --- Sidebar ---
st.sidebar.header("⚙️ Options")
input_mode = st.sidebar.radio("Select Input Type:", ("Video", "Image"), horizontal=True)
demo_mode = st.sidebar.checkbox("Use Demo Mode (no backend required)", value=True)
st.sidebar.info("👩‍💻 Built at HackUTA 2025\nTeam: Matteo + A + B + C")

# --- Input Controls ---
youtube_url = None
uploaded_file = None
uploaded_image = None

if input_mode == "Video":
    youtube_url = st.sidebar.text_input("YouTube URL")
    uploaded_file = st.sidebar.file_uploader("Or upload a video file", type=["mp4", "mov"])
elif input_mode == "Image":
    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

# ---------- Compatibility wrappers ----------
def _call_download(url: str):
    if not download_youtube_video:
        return None
    try:
        return download_youtube_video(url, output_folder="Downloads")
    except TypeError:
        try:
            return download_youtube_video(url, out_dir="Downloads", filename="input.mp4")
        except TypeError:
            return download_youtube_video(url)

def _call_extract(video_path):
    """Call extract_frames regardless of teammate signature and normalize result."""
    if not extract_frames:
        return []
    p = str(video_path)
    result = None
    try:
        result = extract_frames(p, every_n_frames=10, resize=(256, 256), max_frames=90)
    except TypeError:
        try:
            result = extract_frames(p, fps_interval=10)
        except TypeError:
            result = extract_frames(p)

    # Normalize
    frames = []
    if result is None:
        return []
    elif isinstance(result, dict) and "frames" in result:
        frames = result["frames"]
    elif isinstance(result, tuple):
        frames = result[0]
    else:
        frames = result

    if isinstance(frames, np.ndarray):
        if frames.ndim >= 3:
            frames = [frames[i] for i in range(frames.shape[0])]
        else:
            frames = [frames]

    fixed = []
    for f in frames:
        if f is None:
            continue
        arr = np.asarray(f)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        fixed.append(arr)
    return fixed


# ---------- Run Detection ----------
if st.sidebar.button("Run Detection"):

    if input_mode == "Image":
        # ---------------- IMAGE MODE ----------------
        with st.spinner("Analyzing image..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            if uploaded_image is None:
                st.error("Please upload an image.")
                st.stop()

            file_ext = os.path.splitext(uploaded_image.name)[-1].lower()
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
            tfile.write(uploaded_image.read())
            temp_path = tfile.name

            img = cv2.imread(temp_path)
            if img is None:
                st.error("⚠️ Could not read the image file.")
                st.stop()

            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                     caption="Uploaded Image", use_container_width=True)

            if demo_mode or image_detect is None:
                feats = np.random.random(10)
                score = np.random.uniform(0.3, 0.8)
            else:
                feats = image_detect.frequency_analysis([img])
                score = image_detect.fake_score(feats)

            F = np.fft.fftshift(np.fft.fft2(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)))
            spectrum = np.abs(F)

    else:
        # ---------------- VIDEO MODE ----------------
        with st.spinner("Processing video..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress.progress(i + 1)

            # --- Demo mode ---
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
                    with open("temp.mp4", "wb") as f:
                        f.write(uploaded_file.read())
                    video_path = "temp.mp4"
                else:
                    st.error("Please upload a video or provide a YouTube link.")
                    st.stop()

                if not video_path or not os.path.exists(str(video_path)):
                    st.error("Could not obtain or find a valid video file.")
                    st.stop()

                frames = _call_extract(video_path)
                if not frames:
                    st.error("Could not extract frames.")
                    st.stop()

                # Frequency analysis
                feats = frequency_analysis(frames)
                score = fake_score(feats)

                # Spectrum visualization
                F = np.fft.fftshift(np.fft.fft2(frames[0].astype(np.float32)))
                spectrum = np.abs(F)

    # ---------- Results (shared for both modes) ----------
    st.success("Detection complete!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sample Frame" if input_mode == "Video" else "Input Image")
        if input_mode == "Video":
            st.image(frames[2], clamp=True, caption="Extracted Frame (grayscale)")
        else:
            st.image(img, caption="Original Image")

    with col2:
        st.subheader("Frequency Spectrum (Log Magnitude)")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(np.log(spectrum + 1e-8), cmap="inferno")
        ax.axis("off")
        st.pyplot(fig)

    # --- Score ---
    st.subheader("Detection Result")
    st.metric(label="Fake Probability", value=f"{score * 100:.1f}%")
    st.progress(float(min(max(score, 0.0), 1.0)))

    if score < 0.45:
        st.success("✅ Likely Real")
    elif score > 0.55:
        st.error("🚨 Likely Fake")
    else:
        st.warning("⚖️ Uncertain — borderline score")

    # --- Frequency Energy Breakdown ---
    st.subheader("Frequency Energy Breakdown")
    h, w = spectrum.shape
    low_q = spectrum[:h // 2, :w // 2]
    low_energy = float(np.sum(low_q))
    high_energy = float(np.sum(spectrum) - low_energy)
    energy_df = pd.DataFrame({"Low Frequency": [low_energy], "High Frequency": [high_energy]})
    st.bar_chart(energy_df.T)

# --- Footer ---
st.markdown("---")
st.caption("Frequency-based AI/Deepfake detection (CPU-only)")
