"""
Downloads the videos from the Vatex dataset and splits each video into separate images
https://eric-xw.github.io/vatex-website/index.html
"""

import json
import subprocess
import os
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from tqdm import tqdm

def split_videoID(videoID):
    last_underscore_index = videoID.rfind('_')
    second_last_underscore_index = videoID.rfind('_', 0, last_underscore_index)
    YouTubeID = videoID[:second_last_underscore_index]
    StartTime = videoID[second_last_underscore_index + 1:last_underscore_index]
    EndTime = videoID[last_underscore_index + 1:]
    return YouTubeID, StartTime, EndTime

def append_error(videoID: str, errored_file: str, lock: Lock):
    with lock:
        with open(errored_file, "a+", encoding="utf-8") as f:
            f.write(videoID)
            f.write("\n")

def process_video(item, data_folder, errored_file, lock):
    videoID = item['videoID']
    errors = []

    if os.path.exists(errored_file):
        with open(errored_file, "r", encoding="utf-8") as f:
            errors = f.read().splitlines()

    # Skip if the videoID is contained within the error file
    if videoID in errors:
        return 1

    try:
        youtube_id, start_time, end_time = split_videoID(videoID)
    except ValueError:
        append_error(videoID, errored_file, lock)
        return 1

    output_dir = os.path.join(data_folder, videoID)

    if os.path.exists(output_dir):
        return 0

    with TemporaryDirectory() as tmpdir:
        video_file = os.path.join(tmpdir, f'{youtube_id}.mp4')
        youtube_url = f'https://www.youtube.com/watch?v={youtube_id}'
        download_command = [
            'yt-dlp',
            '-f', 'bestvideo[height<=480][height>=360][ext=mp4]',
            '-N', '2',
            '-o', video_file,
            youtube_url
        ]
        dlcode = subprocess.run(download_command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if dlcode.returncode != 0:
            append_error(videoID, errored_file, lock)
            return 1

        os.makedirs(output_dir, exist_ok=True)
        trim_command = [
            'ffmpeg',
            '-y',
            '-ss', start_time,
            '-to', end_time,
            '-i', video_file,
            '-r', '30',
            os.path.join(output_dir, 'frame_%04d.jpg')
        ]
        trimcode = subprocess.run(trim_command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # os.remove(f'{youtube_id}.mp4')

    if trimcode.returncode != 0:
        os.removedirs(output_dir)
        append_error(videoID, errored_file, lock)
        return 1

    return 0

def download_videos_to_img_files(json_file: str, data_folder: str, num_threads: int = 4):
    with open(json_file, 'r', encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    errored_file = os.path.join(data_folder, "errored.txt")
    errored = 0

    lock = Lock()  # For thread-safe file writing
    with ThreadPoolExecutor(max_workers=num_threads) as executor, \
        tqdm(
            total=len(data),
            unit="video",
            desc=f"{json_file}",
            miniters=1,
        ) as pbar:
        for result in executor.map(lambda item: process_video(item, data_folder, errored_file, lock), data):
            errored += result
            pbar.set_postfix_str(f"{json_file} | errored={errored}")
            pbar.update()
            pbar.refresh()
        # results = list(tqdm(
        #     executor.map(lambda item: process_video(item, data_folder, errored_file, lock), data),
        #     total=len(data),
        #     unit="video",
        #     desc=f"{json_file}"
        # ))

    print(f"\nFinished processing {json_file}. Failed to process {errored} videos.")

# Example calls with optional number of threads:
download_videos_to_img_files("data/vatex_training.json", "data/train", num_threads=20)
download_videos_to_img_files("data/vatex_val.json", "data/val", num_threads=20)
download_videos_to_img_files("data/vatex_test.json", "data/test", num_threads=20)
