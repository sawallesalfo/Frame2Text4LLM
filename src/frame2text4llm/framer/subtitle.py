import os
import numpy as np

from loguru import logger
from beartype.typing import List, Tuple
import cv2
import matplotlib.pyplot as plt
from .video import VideoReader


class SubtitleRegionDetector:
    """
    A class for detecting the subtitle region in a video.
    """
    def __init__(self, video_reader: VideoReader):
        """
        Initialize the subtitle region detector.
        
        Args:
            video_reader: VideoReader object for the video
        """
        self.video_reader = video_reader
    
    def detect_region(self) -> Tuple[int, int, int, int]:
        """
        Detect the subtitle region in the video.
        
        Returns:
            Tuple (y1, y2, x1, x2) representing the subtitle region
        """
        logger.info(f"Detecting subtitle region for {os.path.basename(self.video_reader.video_path)}")
        
        total_frames = self.video_reader.video.frame_count
        
        sample_positions = [
            int(total_frames * 0.25),  
            int(total_frames * 0.5),  
            int(total_frames * 0.75),  
        ]
        
        logger.info(f"Analyzing frames at positions: {sample_positions}")
        
        frames = []
        for position in sample_positions:
            frame = self.video_reader.extract_frame(position)
            if frame:
                frames.append(frame.image)
        
        if not frames:
            logger.info("No frames could be extracted for analysis")
            return (0, 0, 0, 0)
        
        # Simple heuristic: subtitle region is often in the bottom third
        height, width = frames[0].shape[:2]
        
        y1 = int(height * 0.7)  # Start at 70% from the top
        y2 = height            # End at the bottom
        # Full width: a 5 % margin cut the first and last letters of wide subtitles
        x1 = 0
        x2 = width
        
        region = (y1, y2, x1, x2)
        logger.info(f"Detected subtitle region: y={y1}:{y2}, x={x1}:{x2}")
        
        return region
    
    def visualize_region(
        self,
        region: Tuple[int, int, int, int],
        frames: List[np.ndarray],
        frame_index: int = 0
    ) -> None:
        """
        Visualize the detected subtitle region on a specific frame from a list.

        Args:
            region: Tuple (y1, y2, x1, x2) representing the subtitle region.
            frames: List of frames as numpy arrays (BGR format).
            frame_index: Index of the frame in the list to visualize.
        """
        if not frames or frame_index >= len(frames):
            logger.info(f"Invalid frame index {frame_index}, or empty frame list")
            return

        y1, y2, x1, x2 = region

        original_frame = frames[frame_index]
        frame_marked = original_frame.copy()

        cv2.rectangle(frame_marked, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Convert BGR to RGB
        original_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        marked_rgb = cv2.cvtColor(frame_marked, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].imshow(original_rgb)
        axes[0].set_title("Original Frame")
        axes[0].axis('off')

        axes[1].imshow(marked_rgb)
        axes[1].set_title(f"Subtitle Region: y={y1}:{y2}, x={x1}:{x2}")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()