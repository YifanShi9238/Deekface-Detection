import streamlit as st
import tempfile
import cv2
import numpy as np
import os

import image_detection as image_detect

st.set_page_config(page_title="AI Image Detector", layout="centered")

st.title("🖼️ AI Image / Deepfake Detector")
st.caption("Upload an image — the system will analyze its frequency patterns to estimate if it’s AI-generated.")

# ----------- File Upload -----------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()

    # Save uploaded image to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    tfile.write(uploaded_file.read())
    temp_path = tfile.name

    # ----------- IMAGE ANALYSIS -----------
    img = cv2.imread(temp_path)
    if img is None:
        st.error("⚠️ Could not read the image file.")
    else:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing frequency spectrum..."):
            feats = image_detect.frequency_analysis([img])
            score = image_detect.fake_score(feats)

        # ----------- Results -----------
        st.success(f"Fake Probability: **{score:.3f}**")

        if score < 0.45:
            st.write("✅ **Likely REAL**")
        elif score > 0.55:
            st.write("🚨 **Likely FAKE**")
        else:
            st.write("⚖️ **Uncertain — borderline score**")

else:
    st.info("📤 Please upload an image to begin analysis.")

# Footer
st.markdown("---")
st.caption("Developed by Yifan Shi • Frequency-based AI image detection (CPU-only)")
