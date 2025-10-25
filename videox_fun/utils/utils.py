import gc
import inspect
import os
import shutil
import subprocess
import time

import cv2
import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image


def filter_kwargs(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

def get_width_and_height_from_image_and_base_resolution(image, base_resolution):
    target_pixels = int(base_resolution) * int(base_resolution)
    original_width, original_height = Image.open(image).size
    ratio = (target_pixels / (original_width * original_height)) ** 0.5
    width_slider = round(original_width * ratio)
    height_slider = round(original_height * ratio)
    return height_slider, width_slider

def color_transfer(sc, dc):
    """
    Transfer color distribution from of sc, referred to dc.

    Args:
        sc (numpy.ndarray): input image to be transfered.
        dc (numpy.ndarray): reference image

    Returns:
        numpy.ndarray: Transferred color distribution on the sc.
    """

    def get_mean_and_std(img):
        x_mean, x_std = cv2.meanStdDev(img)
        x_mean = np.hstack(np.around(x_mean, 2))
        x_std = np.hstack(np.around(x_std, 2))
        return x_mean, x_std

    sc = cv2.cvtColor(sc, cv2.COLOR_RGB2LAB)
    s_mean, s_std = get_mean_and_std(sc)
    dc = cv2.cvtColor(dc, cv2.COLOR_RGB2LAB)
    t_mean, t_std = get_mean_and_std(dc)
    img_n = ((sc - s_mean) * (t_std / s_std)) + t_mean
    np.putmask(img_n, img_n > 255, 255)
    np.putmask(img_n, img_n < 0, 0)
    dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2RGB)
    return dst

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=12, imageio_backend=True, color_transfer_post_process=False):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(Image.fromarray(x))

    if color_transfer_post_process:
        for i in range(1, len(outputs)):
            outputs[i] = Image.fromarray(color_transfer(np.uint8(outputs[i]), np.uint8(outputs[0])))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if imageio_backend:
        if path.endswith("mp4"):
            imageio.mimsave(path, outputs, fps=fps)
        else:
            imageio.mimsave(path, outputs, duration=(1000 * 1/fps))
    else:
        if path.endswith("mp4"):
            path = path.replace('.mp4', '.gif')
        outputs[0].save(path, format='GIF', append_images=outputs, save_all=True, duration=100, loop=0)

def merge_video_audio(video_path: str, audio_path: str):
    """
    Merge the video and audio into a new video, with the duration set to the shorter of the two,
    and overwrite the original video file.

    Parameters:
    video_path (str): Path to the original video file
    audio_path (str): Path to the audio file
    """
    # check
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video file {video_path} does not exist")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"audio file {audio_path} does not exist")

    base, ext = os.path.splitext(video_path)
    temp_output = f"{base}_temp{ext}"

    try:
        # create ffmpeg command
        command = [
            'ffmpeg',
            '-y',  # overwrite
            '-i',
            video_path,
            '-i',
            audio_path,
            '-c:v',
            'copy',  # copy video stream
            '-c:a',
            'aac',  # use AAC audio encoder
            '-b:a',
            '192k',  # set audio bitrate (optional)
            '-map',
            '0:v:0',  # select the first video stream
            '-map',
            '1:a:0',  # select the first audio stream
            '-shortest',  # choose the shortest duration
            temp_output
        ]

        # execute the command
        print("Start merging video and audio...")
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # check result
        if result.returncode != 0:
            error_msg = f"FFmpeg execute failed: {result.stderr}"
            print(error_msg)
            raise RuntimeError(error_msg)

        shutil.move(temp_output, video_path)
        print(f"Merge completed, saved to {video_path}")

    except Exception as e:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        print(f"merge_video_audio failed with error: {e}")

def get_image_to_video_latent(validation_image_start, validation_image_end, video_length, sample_size):
    if validation_image_start is not None and validation_image_end is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]

        if type(validation_image_end) is str and os.path.isfile(validation_image_end):
            image_end = Image.open(validation_image_end).convert("RGB")
            image_end = image_end.resize([sample_size[1], sample_size[0]])
        else:
            image_end = validation_image_end
            image_end = [_image_end.resize([sample_size[1], sample_size[0]]) for _image_end in image_end]

        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start], 
                dim=2
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, :len(image_start)] = start_video
            
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start):] = 255
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0), 
                [1, 1, video_length, 1, 1]
            )
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:] = 255

        if type(image_end) is list:
            image_end = [_image_end.resize(image_start[0].size if type(image_start) is list else image_start.size) for _image_end in image_end]
            end_video = torch.cat(
                [torch.from_numpy(np.array(_image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_end in image_end], 
                dim=2
            )
            input_video[:, :, -len(end_video):] = end_video
            
            input_video_mask[:, :, -len(image_end):] = 0
        else:
            image_end = image_end.resize(image_start[0].size if type(image_start) is list else image_start.size)
            input_video[:, :, -1:] = torch.from_numpy(np.array(image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
            input_video_mask[:, :, -1:] = 0

        input_video = input_video / 255

    elif validation_image_start is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]
        image_end = None
        
        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start], 
                dim=2
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, :len(image_start)] = start_video
            input_video = input_video / 255
            
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start):] = 255
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0), 
                [1, 1, video_length, 1, 1]
            ) / 255
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:, ] = 255
    else:
        image_start = None
        image_end = None
        input_video = torch.zeros([1, 3, video_length, sample_size[0], sample_size[1]])
        input_video_mask = torch.ones([1, 1, video_length, sample_size[0], sample_size[1]]) * 255
        clip_image = None

    del image_start
    del image_end
    gc.collect()

    return  input_video, input_video_mask, clip_image

def get_video_to_video_latent(input_video_path, video_length, sample_size, fps=None, validation_video_mask=None, ref_image=None):
    if input_video_path is not None:
        if isinstance(input_video_path, str):
            cap = cv2.VideoCapture(input_video_path)
            input_video = []

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = 1 if fps is None else max(1,int(original_fps // fps))

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
                    input_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                frame_count += 1

            cap.release()
        else:
            input_video = input_video_path

        input_video = torch.from_numpy(np.array(input_video))[:video_length]
        input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255

        if validation_video_mask is not None:
            validation_video_mask = Image.open(validation_video_mask).convert('L').resize((sample_size[1], sample_size[0]))
            input_video_mask = np.where(np.array(validation_video_mask) < 240, 0, 255)
            
            input_video_mask = torch.from_numpy(np.array(input_video_mask)).unsqueeze(0).unsqueeze(-1).permute([3, 0, 1, 2]).unsqueeze(0)
            input_video_mask = torch.tile(input_video_mask, [1, 1, input_video.size()[2], 1, 1])
            input_video_mask = input_video_mask.to(input_video.device, input_video.dtype)
        else:
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, :] = 255
    else:
        input_video, input_video_mask = None, None

    if ref_image is not None:
        if isinstance(ref_image, str):
            clip_image = Image.open(ref_image).convert("RGB")
        else:
            clip_image = Image.fromarray(np.array(ref_image, np.uint8))
    else:
        clip_image = None

    if ref_image is not None:
        if isinstance(ref_image, str):
            ref_image = Image.open(ref_image).convert("RGB")
            ref_image = ref_image.resize((sample_size[1], sample_size[0]))
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
        else:
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
    return input_video, input_video_mask, ref_image, clip_image

def padding_image(images, new_width, new_height):
    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))

    aspect_ratio = images.width / images.height
    if new_width / new_height > 1:
        if aspect_ratio > new_width / new_height:
            new_img_width = new_width
            new_img_height = int(new_img_width / aspect_ratio)
        else:
            new_img_height = new_height
            new_img_width = int(new_img_height * aspect_ratio)
    else:
        if aspect_ratio > new_width / new_height:
            new_img_width = new_width
            new_img_height = int(new_img_width / aspect_ratio)
        else:
            new_img_height = new_height
            new_img_width = int(new_img_height * aspect_ratio)

    resized_img = images.resize((new_img_width, new_img_height))

    paste_x = (new_width - new_img_width) // 2
    paste_y = (new_height - new_img_height) // 2

    new_image.paste(resized_img, (paste_x, paste_y))

    return new_image

def get_image_latent(ref_image=None, sample_size=None, padding=False):
    if ref_image is not None:
        if isinstance(ref_image, str):
            ref_image = Image.open(ref_image).convert("RGB")
            if padding:
                ref_image = padding_image(ref_image, sample_size[1], sample_size[0])
            ref_image = ref_image.resize((sample_size[1], sample_size[0]))
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
        elif isinstance(ref_image, Image.Image):
            ref_image = ref_image.convert("RGB")
            if padding:
                ref_image = padding_image(ref_image, sample_size[1], sample_size[0])
            ref_image = ref_image.resize((sample_size[1], sample_size[0]))
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
        else:
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255

    return ref_image

def get_image(ref_image=None):
    if ref_image is not None:
        if isinstance(ref_image, str):
            ref_image = Image.open(ref_image).convert("RGB")
        elif isinstance(ref_image, Image.Image):
            ref_image = ref_image.convert("RGB")

    return ref_image

def timer(func):
    def wrapper(*args, **kwargs):
        start_time  = time.time()
        result      = func(*args, **kwargs)
        end_time    = time.time()
        print(f"function {func.__name__} running for {end_time - start_time} seconds")
        return result
    return wrapper

def timer_record(model_name=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            start_time = time.time()
            result      = func(*args, **kwargs)
            torch.cuda.synchronize()
            end_time = time.time()
            import torch.distributed as dist
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    time_sum  = end_time - start_time
                    print('# --------------------------------------------------------- #')
                    print(f'#   {model_name} time: {time_sum}s')
                    print('# --------------------------------------------------------- #')
                    _write_to_excel(model_name, time_sum)
            else:
                time_sum  = end_time - start_time
                print('# --------------------------------------------------------- #')
                print(f'#   {model_name} time: {time_sum}s')
                print('# --------------------------------------------------------- #')
                _write_to_excel(model_name, time_sum)
            return result
        return wrapper
    return decorator

def _write_to_excel(model_name, time_sum):
    import os

    import pandas as pd

    row_env = os.environ.get(f"{model_name}_EXCEL_ROW", "1")  # 默认第1行
    col_env = os.environ.get(f"{model_name}_EXCEL_COL", "1")  # 默认第A列
    file_path = os.environ.get("EXCEL_FILE", "timing_records.xlsx")  # 默认文件名

    try:
        df = pd.read_excel(file_path, sheet_name="Sheet1", header=None)
    except FileNotFoundError:
        df = pd.DataFrame()

    row_idx = int(row_env)
    col_idx = int(col_env)

    if row_idx >= len(df):
        df = pd.concat([df, pd.DataFrame([ [None] * (len(df.columns) if not df.empty else 0) ] * (row_idx - len(df) + 1))], ignore_index=True)

    if col_idx >= len(df.columns):
        df = pd.concat([df, pd.DataFrame(columns=range(len(df.columns), col_idx + 1))], axis=1)

    df.iloc[row_idx, col_idx] = time_sum

    df.to_excel(file_path, index=False, header=False, sheet_name="Sheet1")

def get_autocast_dtype():
    try:
        if not torch.cuda.is_available():
            print("CUDA not available, using float16 by default.")
            return torch.float16

        device = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(device)

        print(f"GPU: {prop.name}, Compute Capability: {prop.major}.{prop.minor}")

        if prop.major >= 8:
            if torch.cuda.is_bf16_supported():
                print("Using bfloat16.")
                return torch.bfloat16
            else:
                print("Compute capability >= 8.0 but bfloat16 not supported, falling back to float16.")
                return torch.float16
        else:
            print("GPU does not support bfloat16 natively, using float16.")
            return torch.float16

    except Exception as e:
        print(f"Error detecting GPU capability: {e}, falling back to float16.")
        return torch.float16