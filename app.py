import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# --- Import teammates' functions (use try/except so app won't break if not ready) ---
try:
    from utils import download_youtube_video, extract_frames   # Person A
    from detection import frequency_analysis, fake_score       # Person B
except ImportError:
    download_youtube_video = None
    extract_frames = None
    frequency_analysis = None
    fake_score = None

# --- Streamlit Config ---
st.set_page_config(page_title="Frequency-Based Fake Detection", layout="wide")

# --- Title / Header ---
st.title("🎥 Frequency-Based Fake Detection")
st.markdown(
    "Upload a video or paste a YouTube link. We'll analyze frequency patterns in the frames "
    "to estimate the likelihood of deepfake content."
)

# --- Sidebar ---
st.sidebar.header("⚙️ Options")
youtube_url = st.sidebar.text_input("YouTube URL")
uploaded_file = st.sidebar.file_uploader("Or upload a video file", type=["mp4", "mov"])
demo_mode = st.sidebar.checkbox("Use Demo Mode (no backend required)", value=True)
st.sidebar.info("👩‍💻 Built at HackUTA 2025\n\nTeam: Matteo + A + B")

# --- Button ---
if st.sidebar.button("Run Detection"):
    with st.spinner("Processing video..."):

        # --- Simulated Progress Bar ---
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.02)  # fake work
            progress.progress(i + 1)

        # --- Demo Mode ---
        if demo_mode or not (download_youtube_video and extract_frames):
            # Fake results
            score = 0.72  # mock probability
            frames = [np.random.randint(0, 255, (256, 256), dtype=np.uint8)]
            spectrum = np.random.rand(256, 256)
        else:
            # --- Real Backend (when teammates finish) ---
            if youtube_url:
                video_path = download_youtube_video(youtube_url, "temp.mp4")
            elif uploaded_file:
                with open("temp.mp4", "wb") as f:
                    f.write(uploaded_file.read())
                video_path = "temp.mp4"
            else:
                st.error("Please upload a video or provide a YouTube link.")
                st.stop()

            frames = extract_frames(video_path, fps_interval=10)
            features = frequency_analysis(frames)
            score = fake_score(features)
            spectrum = features["spectrum"][0]

    # --- Results Section ---
    st.success("Detection complete!")

    # Layout: 2 columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sample Frame")
        st.image(frames[0], channels="GRAY", caption="Extracted Frame")

    with col2:
        st.subheader("Frequency Spectrum (Example)")
        plt.figure(figsize=(4, 4))
        plt.imshow(np.log(np.abs(spectrum) + 1), cmap="inferno")
        plt.axis("off")
        st.pyplot(plt)

    # --- Fake Score ---
    st.subheader("Detection Result")
    st.metric(label="Fake Probability", value=f"{score*100:.1f}%")
    st.progress(min(max(score, 0.0), 1.0))

    if score > 0.5:
        st.error("Likely Deepfake")
    else:
        st.success("Likely Real")

    # --- Extra: Frequency Distribution Chart ---
    st.subheader("Frequency Energy Breakdown")
    mock_data = pd.DataFrame({
        "Low Frequency": [30],
        "High Frequency": [70]
    }) if demo_mode else pd.DataFrame({
        "Low Frequency": [np.random.randint(20, 50)],
        "High Frequency": [np.random.randint(50, 80)]
    })
    st.bar_chart(mock_data.T)
