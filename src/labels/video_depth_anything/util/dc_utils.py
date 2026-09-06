# This file is originally from DepthCrafter/depthcrafter/utils.py at main · Tencent/DepthCrafter
# SPDX-License-Identifier: MIT License license
#
# This file may have been modified by ByteDance Ltd. and/or its affiliates on [date of modification]
# Original file is released under [ MIT License license], with the full license text available at [https://github.com/Tencent/DepthCrafter?tab=License-1-ov-file].
import numpy as np
import matplotlib.cm as cm
import imageio
try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except:
    import cv2
    DECORD_AVAILABLE = False

def ensure_even(value):
    return value if value % 2 == 0 else value + 1

def read_video_frames(video_path, process_length, target_fps=-1, max_res=-1):
    if DECORD_AVAILABLE:
        vid = VideoReader(video_path, ctx=cpu(0))
        original_height, original_width = vid.get_batch([0]).shape[1:3]
        height = original_height
        width = original_width
        if max_res > 0 and max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = ensure_even(round(original_height * scale))
            width = ensure_even(round(original_width * scale))

        vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

        fps = vid.get_avg_fps() if target_fps == -1 else target_fps
        stride = round(vid.get_avg_fps() / fps)
        stride = max(stride, 1)
        frames_idx = list(range(0, len(vid), stride))
        if process_length != -1 and process_length < len(frames_idx):
            frames_idx = frames_idx[:process_length]
        frames = vid.get_batch(frames_idx).asnumpy()
    else:
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        if max_res > 0 and max(original_height, original_width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale)
            width = round(original_width * scale)

        fps = original_fps if target_fps < 0 else target_fps

        stride = max(round(original_fps / fps), 1)

        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (process_length > 0 and frame_count >= process_length):
                break
            if frame_count % stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                if max_res > 0 and max(original_height, original_width) > max_res:
                    frame = cv2.resize(frame, (width, height))  # Resize frame
                frames.append(frame)
            frame_count += 1
        cap.release()
        frames = np.stack(frames, axis=0)

    return frames, fps


def save_video(frames, output_video_path, fps=10, is_depths=False, grayscale=False, reverse_colormap=False):
    writer = imageio.get_writer(output_video_path, fps=fps, macro_block_size=1, codec='libx264', ffmpeg_params=['-crf', '18'])
    if is_depths:
        cmap = cm.get_cmap("inferno").reversed() if reverse_colormap else cm.get_cmap("inferno")
        colormap = np.array(cmap.colors)
        
        d_min, d_max = frames.min(), frames.max()
        for i in range(frames.shape[0]):
            depth = frames[i]
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            depth_vis = (colormap[depth_norm] * 255).astype(np.uint8) if not grayscale else depth_norm
            writer.append_data(depth_vis)
    else:
        for i in range(frames.shape[0]):
            writer.append_data(frames[i])

    writer.close()

def _pad_to_shape(img: np.ndarray, target_hw):
    """
    Pad `img` (H[,W][,C]) with zeros so its spatial size matches `target_hw`=(H,W).
    Pads sym­metrically; the extra row / col (for odd diffs) goes on the bottom / right.
    """
    th, tw = target_hw
    h, w = img.shape[:2]

    if h > th or w > tw:
        raise ValueError("Reconstruction is larger than input along at least one axis")

    pad_h = th - h
    pad_w = tw - w
    pad = (
        (pad_h // 2, pad_h - pad_h // 2),   # top, bottom
        (pad_w // 2, pad_w - pad_w // 2),   # left, right
        (0, 0)                              # channels
    ) if img.ndim == 3 else (
        (pad_h // 2, pad_h - pad_h // 2),
        (pad_w // 2, pad_w - pad_w // 2)
    )

    return np.pad(img, pad, mode="constant", constant_values=0)

def save_comparison_video(
        frames_in,               # [N,H,W]  or  [N,H,W,3]  (NumPy or torch‑like)
        frames_out,              # same N, may differ in H
        output_video_path: str,
        fps: int = 10,
        is_depths: bool = False,
        grayscale: bool = False,
        reverse_colormap: bool = False,
        axis: str = "horizontal"      # "horizontal" or "vertical"
):
    """
    Write a comparison video showing input vs. reconstructed frames.
    If the recon has smaller spatial dims it is zero‑padded to match.
    """
    # ── basic checks ───────────────────────────────────────────────────────────
    frames_in  = np.asarray(frames_in)
    frames_out = np.asarray(frames_out)

    assert frames_in.shape[0] == frames_out.shape[0], "Different number of frames"
    assert axis in {"horizontal", "vertical"}, "`axis` must be 'horizontal' or 'vertical'"

    N, H_in, W_in = frames_in.shape[:3]
    writer = imageio.get_writer(
        output_video_path,
        fps=fps,
        macro_block_size=1,
        codec='libx264',
        ffmpeg_params=['-crf', '18']
    )

    # ── prepare depth colormap once ────────────────────────────────────────────
    if is_depths:
        cmap = cm.get_cmap("inferno").reversed() if reverse_colormap else cm.get_cmap("inferno")
        colors = (np.array(cmap.colors)[:, :3] * 255).astype(np.uint8)  # (256,3)
        d_min = min(frames_in.min(), frames_out.min())
        d_max = max(frames_in.max(), frames_out.max())
        eps   = 1e-8

    # ── iterate over frames ────────────────────────────────────────────────────
    for fin, fout in zip(frames_in, frames_out):
        # --- convert to RGB uint8 -------------------------------------------------
        if is_depths:
            def depth_to_rgb(depth):
                depth_norm = np.clip((depth - d_min) / (d_max - d_min + eps), 0, 1)
                idx = (depth_norm * 255).astype(np.uint8)            # (H,W)
                if grayscale:
                    return np.repeat(idx[..., None], 3, axis=-1)     # gray‑RGB
                return colors[idx]                                   # (H,W,3)

            rgb_in  = depth_to_rgb(fin)
            rgb_out = depth_to_rgb(fout)
        else:
            # assume already uint8, channel‑last; if 2‑D, replicate channels
            rgb_in  = fin  if fin.ndim  == 3 else np.repeat(fin[..., None], 3, axis=-1)
            rgb_out = fout if fout.ndim == 3 else np.repeat(fout[..., None], 3, axis=-1)

        # --- pad reconstruction if needed -----------------------------------------
        rgb_out = _pad_to_shape(rgb_out, (H_in, W_in))

        # --- stitch ---------------------------------------------------------------
        combined = (np.concatenate([rgb_in, rgb_out], axis=1 if axis == "horizontal" else 0)
                    .astype(np.uint8))

        writer.append_data(combined)

    writer.close()
