#!/usr/bin/env python3

import dataclasses
import os
import sys
import argparse
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import cv2
import clip
import torch
from PIL import Image
from natsort import natsorted
from tqdm import tqdm
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


# Configurable parameters


CUT_THRESHOLD = 27.0  # PySceneDetect 硬切偵測靈敏度
SEMANTIC_THRESHOLD = 0.12  # 語意距離超過此值觸發捕捉
SAMPLE_FPS = 2.0  # 語意取樣頻率 (每秒幾幀)
MAX_INTERVAL = 6.0  # 最大間隔 (秒)，超過此時間未捕捉則強制捕捉
MAX_CAPTURES = 15  # 每部影片最多捕捉張數
JPEG_QUALITY = 95  # 輸出 JPEG 品質
CONCURRENT_PREPARE = 2  # 同時轉碼 + 場景偵測的執行緒數


# Logging setup


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)
logging.getLogger("pyscenedetect").setLevel(logging.WARNING)


# Types


@dataclasses.dataclass
class CLIPContext:
    model: torch.nn.Module
    preprocess: object
    device: str


@dataclasses.dataclass
class VideoContext:
    video_path: Path
    h264_path: str | None
    scenes: list[tuple[float, float]] | None
    tmp_dir: str | None
    error: str | None


class VideoFragment(NamedTuple):
    timestamps: list[float]
    frames: list


# Utilities


def parse_args():
    p = argparse.ArgumentParser(description="批量影片關鍵幀擷取")
    p.add_argument("--video-dir", default="/app/video", help="影片資料夾路徑")
    p.add_argument("--output-dir", default="/app/output", help="輸出資料夾路徑")
    p.add_argument("--limit", type=int, default=0, help="只處理前 N 部影片 (0=全部)")
    p.add_argument("--cut-threshold", type=float, default=CUT_THRESHOLD)
    p.add_argument("--semantic-threshold", type=float, default=SEMANTIC_THRESHOLD)
    p.add_argument("--sample-fps", type=float, default=SAMPLE_FPS)
    p.add_argument("--max-interval", type=float, default=MAX_INTERVAL)
    p.add_argument("--max-captures", type=int, default=MAX_CAPTURES)
    p.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    return p.parse_args()


def format_name(name: str, max_length=50) -> str:
    if len(name) <= max_length:
        return name
    else:
        return name[: max_length - 3] + "..."


# Video utilities


def list_videos(video_dir: str) -> list[Path]:
    """列出影片目錄中的所有 mp4，按檔名字母排序。"""
    vdir = Path(video_dir)
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = [f for f in vdir.iterdir() if f.is_file() and f.suffix.lower() in exts]
    return natsorted(videos, key=lambda p: p.name)


def transcode_to_h264(video_path: str, tmp_dir: str) -> str:
    """用系統 ffmpeg 將影片轉碼為 H.264，回傳暫存檔路徑。"""
    tmp_path = os.path.join(tmp_dir, "tmp_video.mp4")

    cmd_parts1 = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video_path]
    cmd_parts2 = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-an", tmp_path]
    result = subprocess.run(cmd_parts1 + cmd_parts2, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 轉碼失敗: {result.stderr}")

    return tmp_path


# Scene utilities


def detect_scenes(video_path: str, threshold: float) -> list[tuple[float, float]]:
    """用 PySceneDetect 偵測硬剪輯場景邊界，回傳 [(start_sec, end_sec), ...]。"""
    video = open_video(video_path)

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        # 沒偵測到場景切換 → 整部影片視為一個場景
        duration = video.duration.get_seconds()
        return [(0.0, duration)]

    return [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]


def sample_scene_fragment(reader: cv2.VideoCapture, vid_fps: float, start: float, end: float, sample_fps: float) -> VideoFragment | None:
    """根據場景邊界按指定頻率取樣幀，回傳 VideoFragment"""
    sample_interval = 1.0 / sample_fps

    frames = []
    timestamps = []
    current_time = start

    while current_time < end:
        frame_no = int(current_time * vid_fps)
        reader.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

        success, frame = reader.read()
        if not success:
            break

        frames.append(frame)
        timestamps.append(current_time)
        current_time += sample_interval

    if not frames:
        return None

    return VideoFragment(timestamps=timestamps, frames=frames)


# CLIP utilities


def load_clip() -> CLIPContext:
    """載入 CLIP ViT-B/32 模型，自動選擇 device。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return CLIPContext(model=model, preprocess=preprocess, device=device)


def compute_clip_embeddings(clip_ctx: CLIPContext, frames_bgr, batch_size=32):
    """將多個 BGR frames 批量轉成 CLIP 嵌入向量。"""
    all_embeddings = []
    preprocess, model, device = clip_ctx.preprocess, clip_ctx.model, clip_ctx.device

    for i in range(0, len(frames_bgr), batch_size):
        batch_frames = frames_bgr[i : i + batch_size]
        batch_tensors = torch.stack([preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in batch_frames]).to(device)

        with torch.no_grad():
            embeddings = model.encode_image(batch_tensors)

        all_embeddings.append(embeddings)

    return torch.cat(all_embeddings, dim=0)


def cosine_distance(a, b):
    """計算兩個嵌入向量的餘弦距離 (0=完全相同, 2=完全相反)。"""
    return 1.0 - torch.nn.functional.cosine_similarity(a, b).item()


# Main processing functions


def prepare_video(video_path: Path, cut_threshold: float) -> VideoContext:
    """將影片轉碼並進行場景偵測，回傳處理影片所需的上下文。"""
    tmp_dir = tempfile.mkdtemp()

    try:
        h264_path = transcode_to_h264(str(video_path), tmp_dir)
        scenes = detect_scenes(h264_path, cut_threshold)

        return VideoContext(video_path=video_path, h264_path=h264_path, scenes=scenes, tmp_dir=tmp_dir, error=None)

    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)

        return VideoContext(video_path=video_path, h264_path=None, scenes=None, tmp_dir=None, error=str(e))


def process_fragment(clip_ctx: CLIPContext, fragment: VideoFragment, total_captures: int, args) -> list[tuple[float, object]]:
    """對場景取樣片段做 CLIP 語意分析，回傳要捕捉的 (timestamp, frame) 幀列表。"""
    threshold = args.semantic_threshold

    embeddings = compute_clip_embeddings(clip_ctx, fragment.frames)

    # 遍歷 embeddings 做捕捉判斷
    captures = []
    last_embedding = embeddings[0]
    last_capture_time = fragment.timestamps[0]
    captures.append((fragment.timestamps[0], fragment.frames[0]))  # 首幀必捕

    for i in range(1, len(fragment.timestamps)):
        if total_captures + len(captures) >= args.max_captures:
            break

        dist = cosine_distance(last_embedding.unsqueeze(0), embeddings[i].unsqueeze(0))
        time_since_last = fragment.timestamps[i] - last_capture_time

        if dist > threshold or time_since_last >= args.max_interval:
            captures.append((fragment.timestamps[i], fragment.frames[i]))
            last_embedding = embeddings[i]
            last_capture_time = fragment.timestamps[i]

    return captures


def process_video(clip_ctx: CLIPContext, vid_ctx: VideoContext, vid_idx: int, output_dir: Path, args) -> int:
    """對單部影片進行 CLIP 分析並儲存捕捉的幀，回傳捕捉的幀數量。"""
    display_name = format_name(vid_ctx.video_path.name)

    if vid_ctx.error:
        log.warning("[%03d] ✗ %s | %s", vid_idx, display_name, vid_ctx.error)
        return 0

    h264_path, scenes = vid_ctx.h264_path, vid_ctx.scenes

    reader = cv2.VideoCapture(h264_path)
    if not reader.isOpened():
        log.warning("[%03d] ✗ 無法開啟影片，跳過 | %s", vid_idx, display_name)
        return 0

    vid_fps = reader.get(cv2.CAP_PROP_FPS)
    if vid_fps <= 0:
        vid_fps = 30.0

    all_captures = []

    for start_sec, end_sec in scenes:
        total_captures = len(all_captures)
        if total_captures >= args.max_captures:
            break

        vid_fragment = sample_scene_fragment(reader, vid_fps, start_sec, end_sec, args.sample_fps)
        if not vid_fragment:
            continue

        captures = process_fragment(clip_ctx, vid_fragment, total_captures, args)
        all_captures.extend(captures)

    reader.release()

    for img_idx, (ts, frame) in enumerate(all_captures, start=1):
        filename = f"{vid_idx:03d}_{img_idx:03d}.jpg"
        out_path = output_dir / filename
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

    log.info("[%03d] %d scenes → %d frames | %s", vid_idx, len(scenes), len(all_captures), display_name)
    return len(all_captures)


# Main function


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(args.video_dir)
    if not videos:
        log.error("在 %s 中找不到任何影片", args.video_dir)
        sys.exit(1)

    if args.limit > 0:
        videos = videos[: args.limit]

    log.info("共 %d 部影片待處理", len(videos))

    log.info("載入 CLIP ViT-B/32 中...")
    clip_ctx = load_clip()
    log.info("CLIP ViT-B/32 (device=%s) 載入完成", clip_ctx.device)

    total_captures = 0

    with ThreadPoolExecutor(max_workers=CONCURRENT_PREPARE) as executor:
        futures = {}

        # Worker thread
        for idx, vid_path in enumerate(videos, start=1):
            future = executor.submit(prepare_video, vid_path, args.cut_threshold)
            futures[idx] = future

        # Main thread
        for idx in tqdm(range(1, len(videos) + 1), desc="處理影片", bar_format="{l_bar}{bar:20}{r_bar}\n"):
            vid_ctx = futures[idx].result()  # 等待此影片轉碼完成

            count = process_video(clip_ctx, vid_ctx, idx, output_dir, args)
            total_captures += count

            if vid_ctx.tmp_dir:
                shutil.rmtree(vid_ctx.tmp_dir, ignore_errors=True)

    log.info("完成！共擷取 %d 張圖片", total_captures)


if __name__ == "__main__":
    main()
