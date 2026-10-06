"""SRTExporter writes valid times. Run with pytest, or python tests/test_export.py."""
import tempfile
from pathlib import Path

from frame2text4llm.export import SRTExporter


def test_srt_times_past_an_hour_and_segment_ends():
    segments = [
        {"time_formatted": "97:12:500", "start_time": "97:12:500", "end_time": "97:15:000", "text": "Bonjour", "success": True},
        {"time_formatted": "98:00:000", "start_time": "98:00:000", "end_time": "98:02:000", "text": "Merci", "success": True},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.srt"
        SRTExporter().export(segments, str(path))
        srt = path.read_text(encoding="utf-8")
    assert "01:37:12,500 --> 01:37:15,000\nBonjour" in srt
    assert "01:38:00,000 --> 01:38:02,000\nMerci" in srt


if __name__ == "__main__":
    test_srt_times_past_an_hour_and_segment_ends()
    print("ok")
