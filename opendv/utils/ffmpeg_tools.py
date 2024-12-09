import subprocess
import os

def get_video_fps_duration(video_path):
    cmd = f"CUDA_VISIBLE_DEVICES=3 ffmpeg -hwaccel cuda -i \"{video_path}\" ",
    try:
        result = subprocess.run(cmd, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        output = result.stderr
        
        duration = 0
        fps = 0
        
        for line in output.splitlines():
            if 'Duration' in line:
                # 时长格式为 "00:00:00.00"
                duration_str = line.split('Duration: ')[1].split(',')[0]
                h, m, s = duration_str.split(':')
                s, ms = s.split('.')
                duration = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 100  # 转换为秒
        
        for line in output.splitlines():
            if 'Stream' in line and 'Video' in line:
                if 'fps' in line:
                    fps = line.split('fps')[0].split()[-1]
        
        return float(duration), float(fps)
    
    except Exception as e:
        # print(f"Error: {e}")
        return 0,0

def extract_frames_ffmpeg(video_path, img_folder, fps, start, duration, gpu_id=None):
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    cmd = f"ffmpeg -hwaccel cuda -i \"{video_path}\" -ss {start} -t {duration}  -vf \"fps={fps}\" \"{img_folder}/%07d.jpg\""
    if gpu_id is not None:
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} " + cmd
    # print(cmd)
    try:
        result = subprocess.run(cmd, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        output = result.stderr
    except Exception as e:
        print(f"Error: {e}")