import os
import yt_dlp
import cv2
import numpy as np

# Original download logic by Person A, lightly refactored
def download_youtube_video(url, output_folder="Downloads"):
    """Download a YouTube video and return its local file path."""
    from yt_dlp import YoutubeDL
    os.makedirs(output_folder, exist_ok=True)
    ydl_opts = {
        "outtmpl": os.path.join(output_folder, "%(title)s.%(ext)s"),
        "format": "mp4/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "quiet": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info)
    return video_path

def extract_frames(video_path, frame_interval=10, resize_dim=(256, 256)):
    """
    Extract frames from video.
    - frame_interval: number of frames to skip between samples
    - resize_dim: (width, height)
    Returns: list of grayscale frames as numpy arrays
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open video:", video_path)
        return frames

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, resize_dim)
            frames.append(gray)
        frame_count += 1

    cap.release()
    return np.array(frames)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    path = "Downloads/test_video.mp4"
    frames = extract_frames(path, 15)
    plt.imshow(frames[0], cmap="gray")
    plt.show()

