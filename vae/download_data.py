import json
import subprocess
import os
from tqdm import tqdm

def split_videoID(videoID):
    # Find the indices of the last two underscores
    last_underscore_index = videoID.rfind('_')
    second_last_underscore_index = videoID.rfind('_', 0, last_underscore_index)

    # Extract components based on the found indices
    YouTubeID = videoID[:second_last_underscore_index]
    StartTime = videoID[second_last_underscore_index + 1:last_underscore_index]
    EndTime = videoID[last_underscore_index + 1:]

    return YouTubeID, StartTime, EndTime


def download_videos_to_img_files(json_file: str, data_folder: str):
    # Load the JSON data
    with open(json_file, 'r', encoding="utf-8") as f:
        data = json.load(f)

    # Ensure data is a list
    if not isinstance(data, list):
        data = [data]

    errored = 0

    # Process each item in the JSON data
    progress = tqdm(data, unit="videos", total=len(data), desc=f"{json_file}")
    for item in progress:
        videoID = item['videoID']
        # Parse videoID to get YouTube ID, start time, and end time
        try:
            youtube_id, start_time, end_time = split_videoID(videoID)
        except ValueError:
            #print(f'Invalid videoID format: {videoID}')
            errored = errored + 1
            continue

        # get output directory path
        output_dir = os.path.join(data_folder, videoID)

        if os.path.exists(output_dir):
            continue

        progress.set_description_str(f"Downloading {youtube_id}")
        progress.set_postfix_str(f"{json_file} | errored {errored}")

        # Build YouTube URL
        youtube_url = f'https://www.youtube.com/watch?v={youtube_id}'

        # Download the video using yt-dlp
        # print(f'Downloading video {youtube_id}...')
        download_command = [
            'yt-dlp',
            '-f', 'bestvideo[height<=720][height>=360][ext=mp4]',
            '-N', '2',
            '-o', f'{youtube_id}.mp4',
            youtube_url
        ]
        # silently run the process
        dlcode = subprocess.run(download_command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Continue if download failed
        if dlcode.returncode != 0:
            errored = errored + 1
            continue

        # Create output directory for frames
        os.makedirs(output_dir, exist_ok=True)
        # Trim the video using ffmpeg
        # print(f'Trimming video {youtube_id} from {start_time} to {end_time} seconds...')
        progress.set_description_str(f"Trimming {youtube_id}")
        trim_command = [
            'ffmpeg',
            '-y',  # Overwrite output files without asking
            '-ss', start_time,
            '-to', end_time,
            '-i', f'{youtube_id}.mp4',
            '-r', '30',  # Set the output framerate to a standard 30 fps
            os.path.join(output_dir, 'frame_%04d.jpg')
        ]
        trimcode = subprocess.run(trim_command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if trimcode.returncode != 0:
            errored = errored + 1
            os.remove(output_dir)
            continue

        # Clean up downloaded and trimmed videos to save space
        os.remove(f'{youtube_id}.mp4')

        #print(f'Processing for {videoID} completed.\n')



download_videos_to_img_files("data/vatex_training.json", "data/train")
download_videos_to_img_files("data/vatex_val.json", "data/val")
download_videos_to_img_files("data/vatex_test.json", "data/test")
