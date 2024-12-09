"""
This script is used for preprocessing OpenDV-YouTube meta data, from raw video files to image files.
The script is a part of the [`GenAD`](https://arxiv.org/abs/2403.09630) project.
"""

import json
import os, sys
import time
import argparse
from multiprocessing import Pool

from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.easydict import EasyDict
# from utils.frame_extraction import extract_frames
from utils.download import POSSIBLE_EXTS, youtuber_formatize, get_mini_opendv
from utils.ffmpeg_tools import extract_frames_ffmpeg, get_video_fps_duration

gpus = [0,1,2]

def collect_unfinished_videos(config, mini=False):
    configs = EasyDict(json.load(open(config, "r")))
    root = {
        "train": configs.train_img_root,
        "val": configs.val_img_root
    }

    meta_infos = json.load(open(configs.meta_info, "r"))
    if mini:
        meta_infos = get_mini_opendv(meta_infos)
    if os.path.exists(configs.finish_log):
        finish_log = set(open(configs.finish_log, "r").readlines())
        finish_log = {x.strip() for x in finish_log}
    else:
        finish_log = set()
    
    unfinished_videos = []
    print("collecting unfinished videos...")
    for video_meta in tqdm(meta_infos):
        if video_meta["videoid"] in finish_log:
            continue
        video_path = os.path.join(configs.video_root, youtuber_formatize(video_meta["youtuber"]), video_meta['videoid'])
        for ext in POSSIBLE_EXTS:
            if os.path.exists(f"{video_path}.{ext}"):
                break
        if not os.path.exists(f"{video_path}.{ext}"):
            # raise ValueError(f"Video {video_meta['videoid']} not found. maybe something wrong in the download process?")
            continue

        img_path = os.path.join(root[video_meta["split"].lower()], youtuber_formatize(video_meta["youtuber"]), video_meta['videoid'])
        expeced_frames = int((video_meta["length"]-video_meta['start_discard']-video_meta['end_discard'])*configs.frame_rate)
        num_frames = 0
        if os.path.exists(img_path):
            num_frames = len(os.listdir(img_path))
        
        if num_frames==expeced_frames:
            continue
        
        video_info = {
            "video_id": video_meta["videoid"],
            "video_path": f"{video_path}.{ext}",
            "output_dir": img_path,
            "freq": configs.frame_rate,
            "start_discard": video_meta["start_discard"],
            "end_discard": video_meta["end_discard"],
            "length": video_meta["length"],
            "gpu_id": gpus[len(unfinished_videos)%len(gpus)]
        }
        unfinished_videos.append(video_info)

    return unfinished_videos, EasyDict(configs)

def extract_frames(video_info):
    video_path = video_info.get("video_path", None)
    output_dir = video_info.get("output_dir", None)
    fps = video_info.get("freq", 10)
    discard_begin = video_info.get("start_discard", 90)
    discard_end = video_info.get("end_discard", 60)
    length = video_info.get("length")
    gpu_id = video_info.get("gpu_id")
    extract_frames_ffmpeg(video_path,output_dir,fps,discard_begin, length - discard_begin - discard_end, gpu_id)


def convert_multiprocess(video_lists, configs):
    video_count = len(video_lists)    
    with Pool(configs.num_workers) as p:
        current_time = time.perf_counter()
        for _ in tqdm(p.imap(extract_frames, video_lists), total=video_count):
            pass
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/video2img.json')
    parser.add_argument('--mini', action='store_true', default=False, help='Convert mini dataset only.')

    args = parser.parse_args()
    video_lists, meta_configs = collect_unfinished_videos(args.config, args.mini)

    convert_multiprocess(video_lists, meta_configs)