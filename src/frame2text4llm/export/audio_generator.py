import os
import subprocess
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import soundfile as sf
from loguru import logger

class AudioGenerator:
    """
    Extracts audio from a video and segments it according to timestamps.
    Segments are saved as .wav files in a temporary directory.
    """
    def __init__(self, video_path: str, tmp_audio_dir: str = "tmp_audio"):
        self.video_path = video_path
        self.tmp_audio_dir = Path(tmp_audio_dir)
        self.tmp_audio_dir.mkdir(parents=True, exist_ok=True)

    def extract_segments(self, segments: List[Dict[str, Any]],
                        sample_rate: int = 16000) -> List[str]:
        """
        For each segment (with start_time, end_time), extract audio and save as wav.
        Returns list of audio file paths.
        """
        audio_paths = []
        for idx, seg in enumerate(segments):
            start = self._parse_time(seg['start_time'])
            end = self._parse_time(seg.get('end_time'))
            duration = max(0.1, end - start) if end > start else 3.0
            out_path = self.tmp_audio_dir / f"segment_{idx:04d}.wav"
            cmd = [
                "ffmpeg", "-y", "-i", str(self.video_path),
                "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
                "-ar", str(sample_rate), "-ac", "1", "-vn", str(out_path),
                "-loglevel", "error"
            ]
            result = subprocess.run(cmd)
            if result.returncode == 0 and out_path.exists():
                audio_paths.append(str(out_path))
            else:
                logger.error(f"Failed to extract audio segment {idx} ({start}-{end})")
                audio_paths.append("")
        return audio_paths

    def _parse_time(self, t: str) -> float:
        if not t:
            return 0.0
        t = t.replace(",", ".")
        parts = t.split(":")
        if len(parts) == 2:
            minutes, rest = parts
            seconds, ms = rest.split(".") if "." in rest else (rest, "0")
            return int(minutes) * 60 + int(seconds) + int(ms) / 1000
        elif len(parts) == 3:
            hours, minutes, rest = parts
            seconds, ms = rest.split(".") if "." in rest else (rest, "0")
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(ms) / 1000
        return 0.0
