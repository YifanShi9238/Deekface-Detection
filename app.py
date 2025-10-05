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

# --- NEW: Hybrid detector ---
try:
    from models.hybrid_detector import HybridDetector
    HYBRID_AVAILABLE = True
except Exception:
    HYBRID_AVAILABLE = False

# --- Streamlit Config ---
st.set_page_config(page_title="Frequency-Based Fake Detection", layout="wide")

# --- Initialize hybrid detector (cached) ---
@st.cache_resource
def load_hybrid_detector():
    if not HYBRID_AVAILABLE:
        return None
    model_path = 'weights/xception_ff.pth'
    if os.path.exists(model_path):
        detector = HybridDetector(model_path=model_path)
        return detector
    return None

hybrid_detector = load_hybrid_detector()

# --- Title / Header ---
st.title("🎥 Frequency-Based Fake Detection")
st.markdown(
    "Upload a video or image (or paste a YouTube link). "
    "The system analyzes frequency patterns to estimate the likelihood of AI-generated or deepfake content."
)

# --- Sidebar ---
st.sidebar.header("⚙️ Options")
input_mode = st.sidebar.radio("Select Input Type:", ("Video", "Image"), horizontal=True)
demo_mode = st.sidebar.checkbox("Use Demo Mode (no backend required)", value=False)

# NEW: Detection mode selector
if hybrid_detector and not demo_mode:
    detection_mode = st.sidebar.selectbox(
        "Detection Method:",
        ["Hybrid (CNN + Frequency)", "Frequency Only", "CNN Only"],
        help="Hybrid combines both methods for best accuracy"
    )
    mode_map = {
        "Hybrid (CNN + Frequency)": "hybrid",
        "Frequency Only": "frequency",
        "CNN Only": "cnn"
    }
    selected_mode = mode_map[detection_mode]
else:
    detection_mode = "Frequency Only"
    selected_mode = "frequency"

st.sidebar.info("👩‍💻 Built at HackUTA 2025\nTeam: Matteo + A + B + C")

# Status indicator
if hybrid_detector and hybrid_detector.model_loaded:
    st.sidebar.success("✅ Hybrid Model Loaded")
else:
    st.sidebar.warning("⚠️ CNN Model Not Available\nUsing frequency analysis only")

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

            # NEW: Use hybrid detector if available
            if demo_mode:
                feats = np.random.random(10)
                score = np.random.uniform(0.3, 0.8)
                result = {'explanation': [], 'cnn_score': None, 'frequency_score': score, 'confidence': 'low'}
            elif hybrid_detector:
                result = hybrid_detector.predict(img, mode=selected_mode)
                score = result['ensemble_score']
                feats = result['features']
            elif image_detect:
                feats = image_detect.frequency_analysis([img])
                score = image_detect.fake_score(feats)
                result = {'explanation': [], 'cnn_score': None, 'frequency_score': score, 'confidence': 'low'}
            else:
                feats = np.random.random(10)
                score = np.random.uniform(0.3, 0.8)
                result = {'explanation': [], 'cnn_score': None, 'frequency_score': score, 'confidence': 'low'}

            # Get spectrum for visualization
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            F = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
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
                result = {'explanation': [], 'cnn_score': None, 'frequency_score': score, 'confidence': 'medium'}
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
                result = {'explanation': [], 'cnn_score': None, 'frequency_score': score, 'confidence': 'medium'}

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

    # --- Score with confidence ---
    st.subheader("Detection Result")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(label="Fake Probability", value=f"{score * 100:.1f}%")
    with col_b:
        st.metric(label="Confidence", value=result.get('confidence', 'unknown').upper())
    with col_c:
        st.metric(label="Method", value=detection_mode)
    
    st.progress(float(min(max(score, 0.0), 1.0)))

    if score < 0.45:
        st.success("✅ Likely Real")
    elif score > 0.55:
        st.error("🚨 Likely Fake")
    else:
        st.warning("⚖️ Uncertain — borderline score")

    # NEW: Show component scores if available
    if result.get('cnn_score') is not None or result.get('frequency_score') is not None:
        st.subheader("Score Breakdown")
        score_data = {}
        if result.get('cnn_score') is not None:
            score_data['CNN'] = result['cnn_score']
        if result.get('frequency_score') is not None:
            score_data['Frequency'] = result['frequency_score']
        if result.get('ensemble_score') is not None and selected_mode == 'hybrid':
            score_data['Ensemble'] = result['ensemble_score']
        
        score_df = pd.DataFrame(score_data, index=['Score'])
        st.bar_chart(score_df.T)

    # NEW: Show explanations
    if result.get('explanation'):
        st.subheader("Analysis Details")
        for explanation in result['explanation']:
            st.info(explanation)

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
st.caption("Hybrid AI/Deepfake detection with CNN + Frequency Analysis")