"""
This script is used for downloading OpenDV-YouTube raw data.
The script is a part of the [`GenAD`](https://arxiv.org/abs/2403.09630) project.
"""

from multiprocessing import Pool
from tqdm import tqdm
import os, sys
import time
import json
import cv2
import subprocess

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.easydict import EasyDict
from utils.download import youtuber_formatize, POSSIBLE_EXTS, get_video_with_meta, get_mini_opendv

from utils.ffmpeg_tools import get_video_fps_duration

CONFIGS = dict()

def format_file_size(size_in_bytes):
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = size_in_bytes
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"

def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"

def check_status(video_list, configs, video2img_configs):

    valid_videos = []
    downloading_videos = []
    invalid_videos = []
    valid_imgs = []
    invalid_imgs_size = 0

    check_log_path = "./check_log.json"

    print("Checking Started")
    for vid_info in video_list:
        exists = False
        path = os.path.join(configs.root, youtuber_formatize(vid_info["youtuber"]))
        for ext in POSSIBLE_EXTS:
            video_path = "{}/{}.{}".format(path, vid_info["videoid"], ext)
            if os.path.exists(f"{video_path}.part"):
                downloading_videos.append(video_path)
            if os.path.exists(video_path):
                exists = True
                break
        if not exists:
            continue

        video_infos = None
        for v in valid_videos:
            if v['path']==video_path:
                video_infos = v
                break
            
        if video_infos is None:
            video_infos={
                "path" : video_path,
                "size" : os.path.getsize(video_path),
            }
            expected_duration = vid_info["length"]
            print(f"Checking Video [{vid_info['videoid']}] ... ", end="")
            true_duration, fps = get_video_fps_duration(video_path)
            print(f"size={format_file_size(video_infos['size']):>10}, fps={fps:5.2f}, duration={true_duration:8.2f} ",end="")
            if abs(true_duration - expected_duration) > 2:
                print(f"Failed!")
                invalid_videos.append(video_path)
                continue
            print(f"Succeed!",end="")
            video_infos['fps'] = fps
            video_infos['duration'] = true_duration
            valid_videos.append(video_infos)

        img_infos = None
        for v in valid_imgs:
            if v['video_path']==video_path:
                img_infos = v
                break
        if img_infos is None:
            print(f", Checking Images ... ", end="")
            img_infos = {'video_path':video_path, 'size':0}
            if vid_info['split'] == "Train":
                img_root = video2img_configs.train_img_root
            else:
                img_root = video2img_configs.val_img_root
            img_path = os.path.join(img_root, youtuber_formatize(vid_info["youtuber"]), vid_info['videoid'])
            # print(img_path)
            expeced_frames = int((vid_info["length"]-vid_info['start_discard']-vid_info['end_discard'])*video2img_configs['frame_rate'])
            num_frames = 0
            if os.path.exists(img_path):
                num_frames = len(os.listdir(img_path))
                res = subprocess.run(f"du -sb \"{img_path}\" ", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                img_infos['size']=int(res.stdout.split()[0])
            print(f"size={format_file_size(img_infos['size']):>10}, frames={num_frames:6} ",end="")
            if expeced_frames!=num_frames:
                invalid_imgs_size += img_infos['size']
                print(f"Failed!  Expect {expeced_frames}")
                continue
            img_infos['num_frames']=num_frames
            valid_imgs.append(img_infos)
            print("Succeed!")
    
    print("Checking Finished")
    print("")

    with open(check_log_path,"w") as f:
        check_log = {'valid_videos':valid_videos, 'downloading_videos':downloading_videos, 'invalid_videos':invalid_videos,'valid_imgs':valid_imgs}
        json.dump(check_log, f, indent=4, ensure_ascii=False)

    total_size, total_duration = 0,0
    for v in valid_videos:
        total_duration+=v['duration']
        total_size+=v['size']
    total_imgs_size, total_frames = 0,0
    for v in valid_imgs:
        total_imgs_size+=v['size']
        total_frames+=v['num_frames']
    downloading_size, invalid_size =0,0
    for v in downloading_videos:
        if os.path.exists(v+".part"):
            downloading_size+=os.path.getsize(v+".part")
    for v in invalid_videos:
        if os.path.exists(v):
            invalid_size+=os.path.getsize(v)
    print(f"Full Dataset: {len(video_list)}")

    print(f"Downloading Videos: {len(downloading_videos)}")
    print(f"Downloading Videos Size: {format_file_size(downloading_size)}")

    print(f"Invalid Videos: {len(invalid_videos)}")
    print(f"Invalid Videos Size: {format_file_size(invalid_size)}")

    print(f"Valid Videos: {len(valid_videos)}")
    print(f"Valid Videos Size: {format_file_size(total_size)}")
    print(f"Valid Videos Duration: {format_time(int(total_duration))}")

    print(f"Processing Size: {format_file_size(invalid_imgs_size)}")
    print(f"Processed Videos: {len(valid_imgs)}")
    print(f"Total Frames: {total_frames}")
    print(f"Total Frames Size: {format_file_size(total_imgs_size)}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_config", type=str, default="configs/download.json", help="Path to the config file. should be a `json` file.")
    parser.add_argument("--video2img_config", type=str, default="configs/video2img.json", help="Path to the config file. should be a `json` file.")
    parser.add_argument("--mini", action="store_true", default=False, help="Download mini dataset only.")
    args = parser.parse_args()
    
    configs = EasyDict(json.load(open(args.download_config, "r")))
    video2img_configs = EasyDict(json.load(open(args.video2img_config, "r")))

    video_list = json.load(open(configs.pop("video_list"), "r"))
    if args.mini:
        video_list = get_mini_opendv(video_list)

    check_status(video_list, configs, video2img_configs)