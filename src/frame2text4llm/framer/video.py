from .base import Video, VideoFrame
from typing import List, Optional, Union, Any
import os
import math
import time
import tempfile
import subprocess
import numpy as np
from loguru import logger
import cv2


class VideoReader:
    def __init__(self, video_path: str, engine: str = 'opencv'):
        """
        Initialize the video reader.
        
        Args:
            video_path: Path to the video file
            engine: Video processing engine ('opencv' or 'ffmpeg')
        """
        self.video_path = video_path
        self.engine = engine.lower()
        self._validate_path()
        self.video = Video(path=video_path)
        self._load_video_info()
    
    def _validate_path(self) -> None:
        """Validate that the video file exists."""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
    
    def _load_video_info(self) -> None:
        """Load video information using the selected engine."""
        if self.engine == 'opencv':
            self._load_video_info_opencv()
        elif self.engine == 'ffmpeg':
            self._load_video_info_ffmpeg()
        else:
            raise ValueError(f"Unsupported engine: {self.engine}. Use 'opencv' or 'ffmpeg'.")
    
    def _load_video_info_opencv(self) -> None:
        """Load video information using OpenCV."""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise IOError(f"Could not open video: {self.video_path}")
        
        self.video.fps = cap.get(cv2.CAP_PROP_FPS)
        self.video.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video.duration_seconds = self.video.frame_count / self.video.fps if self.video.fps else 0
        
        cap.release()
    
    def _load_video_info_ffmpeg(self) -> None:
        """Load video information using FFmpeg."""
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
            "-of", "default=noprint_wrappers=1", self.video_path
        ]
        
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
            lines = output.strip().split('\n')
            info = {}
            
            for line in lines:
                if '=' in line:
                    key, value = line.split('=')
                    info[key] = value
            
            # Parse frame rate (usually in the form "24000/1001")
            fps_parts = info.get('r_frame_rate', '').split('/')
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 0
            
            self.video.fps = fps
            self.video.width = int(info.get('width', 0))
            self.video.height = int(info.get('height', 0))
            self.video.frame_count = int(info.get('nb_frames', 0))
            
            # Try to get duration from FFprobe, otherwise calculate from frame count and FPS
            if 'duration' in info:
                self.video.duration_seconds = float(info['duration'])
            else:
                self.video.duration_seconds = self.video.frame_count / fps if fps else 0
            
        except (subprocess.CalledProcessError, ValueError) as e:
            logger.info(f"Error getting video info with FFmpeg: {e}")
            # Fallback to OpenCV if FFmpeg fails
            self._load_video_info_opencv()
    
    def print_info(self) -> None:
        """Print video information."""
        logger.info(f"Video information for {os.path.basename(self.video_path)}:")
        logger.info(f"  - Dimensions: {self.video.width}x{self.video.height}")
        logger.info(f"  - FPS: {self.video.fps:.2f}")
        logger.info(f"  - Total frames 📸 : {self.video.frame_count}")
        logger.info(f"  - Duration: {self.video.duration_seconds:.2f} seconds")
        logger.info(f"  - Time per frame: {self.video.duration_seconds / self.video.frame_count:.4f} seconds")
        logger.info(f"  - Engine: {self.engine}")
    
    def extract_frame(self, position: int) -> Optional[VideoFrame]:
        """
        Extract a specific frame from the video.
        
        Args:
            position: Frame number to extract
            
        Returns:
            VideoFrame object or None if extraction fails
        """
        if self.engine == 'opencv':
            return self._extract_frame_opencv(position)
        elif self.engine == 'ffmpeg':
            return self._extract_frame_ffmpeg(position)
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")
    
    def _extract_frame_opencv(self, position: int) -> Optional[VideoFrame]:
        """Extract a specific frame using OpenCV."""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            logger.info(f"Error: Could not open video {self.video_path}")
            return None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logger.info(f"Error: Could not extract frame {position}")
            return None
        
        # Calculate accurate time based on frame position and total video duration
        if self.video.frame_count > 0:
            time_seconds = position * (self.video.duration_seconds / self.video.frame_count)
        else:
            time_seconds = position / self.video.fps if self.video.fps else 0
            
        return VideoFrame(frame, time_seconds, position)
    
    def _extract_frame_ffmpeg(self, position: int) -> Optional[VideoFrame]:
        """Extract a specific frame using FFmpeg."""
        # Calculate accurate time based on frame position and total video duration
        if self.video.frame_count > 0:
            time_seconds = position * (self.video.duration_seconds / self.video.frame_count)
        else:
            time_seconds = position / self.video.fps if self.video.fps else 0
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            cmd = [
                "ffmpeg", "-v", "error", "-ss", str(time_seconds),
                "-i", self.video_path, "-vframes", "1",
                "-q:v", "2", temp_path
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            frame = cv2.imread(temp_path)
            
            if frame is None:
                logger.info(f"Error: Could not extract frame at position {position} (time {time_seconds}s)")
                return None
            
            return VideoFrame(frame, time_seconds, position)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def extract_frames(self,  target_fps: int = None, max_frames: int = -1, filter_duplicates: bool = False, diff_thresh: int = 10) -> List[VideoFrame]:
        """
        Extract frames from the video.
        
        Args:
            target_fps: Target sampling rate (frames per second)
            max_frames: Maximum number of frames to extract (-1 = no limit)
            filter_duplicates: Whether to filter duplicate frames
            diff_thresh: Threshold for duplicate detection (higher = more strict)
            
        Returns:
            List of VideoFrame objects
        """
        if self.engine == 'opencv':
            frames = self._extract_frames_opencv(target_fps, max_frames)
        elif self.engine == 'ffmpeg':
            frames = self._extract_frames_ffmpeg(target_fps, max_frames)
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")
        
        #if filter duplicates
        if filter_duplicates:
            #locally to avoid circular import
            from ..ocr.utils.image_utils import _filter_duplicates
            frames = _filter_duplicates(frames, diff_thresh)
        
        self.video.frames = frames
        return frames
    

    
    def _extract_frames_opencv(self,  target_fps: int, max_frames: int) -> List[VideoFrame]:
        """Extract frames at regular intervals using OpenCV."""
        max_frames = self.video.frame_count if max_frames is -1 else max_frames
        logger.info(f"Extracting frames with OpenCV: {os.path.basename(self.video_path)}")
        start_time = time.time()
        
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            logger.info(f"Error: Could not open video {self.video_path}")
            return []
        
        video_fps = self.video.fps
        frame_interval = int(video_fps / target_fps) if target_fps else 1
        
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frame_count / video_fps if video_fps else 0
        num_frames = int(min(target_fps * duration, total_frame_count))
        logger.info(f"💡 Extracting {num_frames} frames from {total_frame_count} total frames at {target_fps} FPS")
    
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Calculate accurate time based on frame position and total video duration
                if self.video.frame_count > 0:
                    current_time = frame_count * (self.video.duration_seconds / self.video.frame_count)
                else:
                    current_time = frame_count / video_fps if video_fps else 0
                    
                frames.append(VideoFrame(frame, current_time, frame_count))
                
                if max_frames and len(frames) >= max_frames:
                    break
                    
                if len(frames) % 10 == 0:
                    logger.info(f"Extracted {len(frames)} frames...")
            
            frame_count += 1
        
        cap.release()
        elapsed = time.time() - start_time
        
        logger.info(f"✅ Extraction complete: {len(frames)} frames in {elapsed:.2f} seconds")
        return frames
    
    def _extract_frames_ffmpeg(self,  target_fps: int, max_frames: int) -> List[VideoFrame]:
        """Extract frames at regular intervals using FFmpeg."""
        logger.info(f"Extracting frames with FFmpeg: {os.path.basename(self.video_path)}")
        start_time = time.time()
        
        frames = []
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Build FFmpeg command
            output_pattern = os.path.join(tmp_dir, "frame_%04d.png")
            
            cmd = [
                "ffmpeg", "-i", self.video_path, 
                "-vf", f"fps={target_fps}", 
                "-q:v", "2"
            ]
            
            if max_frames:
                cmd.extend(["-vframes", str(max_frames)])
            
            cmd.append(output_pattern)
            
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                logger.info(f"FFmpeg error: {e.stderr.decode('utf-8')}")
                return []
            
            # Collect extracted frames
            frame_files = sorted([f for f in os.listdir(tmp_dir) if f.startswith("frame_")])
            
            total_frames = len(frame_files)
            for i, frame_file in enumerate(frame_files):
                frame_path = os.path.join(tmp_dir, frame_file)
                image = cv2.imread(frame_path)
                
                if image is not None:
                    # Calculate accurate time based on number of extracted frames and total duration
                    if total_frames > 1:
                        current_time = i * (self.video.duration_seconds / (total_frames - 1))
                    else:
                        current_time = 0
                        
                    frames.append(VideoFrame(image, current_time, i))
        
        elapsed = time.time() - start_time
        logger.info(f"✅  Extraction complete: {len(frames)} frames in {elapsed:.2f} seconds")
        return frames