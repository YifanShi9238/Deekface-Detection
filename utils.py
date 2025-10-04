import os
import yt_dlp

def download_youtube_video(url, output_folder="Downloads"):
    """Download YouTube video and return its local file path."""
    os.makedirs(output_folder, exist_ok=True)
    ydl_opts = {
        'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'),
        'format': 'mp4/bestvideo+bestaudio/best'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info)

    return video_path

def main():
    url = "https://www.youtube.com/watch?v=dOySjaatsy8"  # Example URL
    video_path = download_youtube_video(url)
    print(f"Video downloaded to: {video_path}")


if __name__ == "__main__":
    main()