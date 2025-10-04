import os
from pathlib import Path
import cv2

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

# Added helper for downstream modules
def extract_frames(video_path, every_n_frames=10, resize=(256, 256), max_frames=120):
    """Extract grayscale frames from a local video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frames, idx = [], 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n_frames == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if resize:
                gray = cv2.resize(gray, resize)
            frames.append(gray)
        idx += 1
    cap.release()
    return frames

if __name__ == "__main__":
    # quick manual test for download
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    saved_path = download_youtube_video(test_url)
    print(f"Video downloaded to: {saved_path}")
