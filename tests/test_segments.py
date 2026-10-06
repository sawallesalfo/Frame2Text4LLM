"""From OCR'd frames to audio cuts: segment times. Run with pytest, or python tests/test_segments.py."""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from frame2text4llm.export.audio_generator import AudioGenerator
from frame2text4llm.framer.base import VideoFrame
from frame2text4llm.ocr.utils.text_utils import _merge_segments


def frames(texts):
    """One OCR result per second, as OCRBatchProcessor returns them."""
    return [
        {"time_formatted": VideoFrame(np.zeros(1), float(t), t).time_formatted, "text": x, "success": True}
        for t, x in enumerate(texts)
    ]


def test_a_segment_ends_when_its_subtitle_leaves_the_screen():
    segments = _merge_segments(frames(["", "Bonjour chef !", "Bonjour chef !", "", "Et la famille ?", ""]))
    assert [(s["text"], s["start_time"], s["end_time"]) for s in segments] == [
        ("Bonjour chef !", "00:01:000", "00:03:000"),
        ("Et la famille ?", "00:04:000", "00:05:000"),
    ]


def test_one_subtitle_read_differently_stays_one_segment():
    """A line skipped, accents lost, one frame read as empty: still one subtitle on screen 5 s."""
    segments = _merge_segments(frames([
        "",
        "Deuxièmement, le portail de ma cour",
        "Deuxièmement, le portail de ma cour a été confectionné par ma femme",
        "",
        "Deuxiemement le portail de ma cour a ete confectionne",
        "Si vous nous aidez, cela nous fera un grand bien.",
        "",
        "",
    ]))
    assert [(s["start_time"], s["end_time"], s["best_time"]) for s in segments] == [
        ("00:01:000", "00:05:000", "00:02:000"),
        ("00:05:000", "00:06:000", "00:05:000"),
    ]
    assert segments[0]["text"].endswith("par ma femme")


def test_times_are_read_back_in_seconds():
    parse = AudioGenerator.__new__(AudioGenerator)._parse_time
    assert parse(VideoFrame(np.zeros(1), 83.5, 0).time_formatted) == 83.5
    assert parse("00:00:05") == 5  # HH:MM:SS, as in notebooks/exporters.ipynb


def test_audio_is_cut_where_the_segment_is():
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "tone.wav"  # silence, then a tone from 3 s
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", "aevalsrc=if(gte(t\\,3)\\,sin(2*PI*440*t)\\,0):s=16000:d=6", str(audio)],
            check=True,
        )
        segments = _merge_segments(frames(["", "silence", "", "ton", "", ""]))
        paths = AudioGenerator(str(audio), tmp_audio_dir=str(Path(tmp) / "cuts")).extract_segments(segments)
        (silence, _), (tone, _) = (sf.read(p) for p in paths)
        assert len(silence) == len(tone) == 16000
        assert np.abs(silence).max() < 0.01 < np.abs(tone[:1600]).max()


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("ok")
