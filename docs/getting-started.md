# Getting started

One video, from frames to audio cuts. Install with the EasyOCR extra, and have ffmpeg on the path:

```bash
pip install "frame2text4llm[easyocr]"
```

## 1. Read the frames

```python
from frame2text4llm.framer import VideoReader

reader = VideoReader("episode.mp4")
frames = reader.extract_frames(target_fps=2)
print(len(frames), frames[0].time_formatted)  # MM:SS:mmm
```

Two frames per second puts each subtitle's start and end within half a second. Do not pass
`filter_duplicates=True` here: it drops frames whose picture barely changed, and a new subtitle
on a still picture is exactly that.

## 2. Say where the subtitles are

```python
from frame2text4llm.ocr import OCRManager

h, w = frames[0].image.shape[:2]
manager = OCRManager(reader, region=(int(h * 0.7), h, 0, w))  # (y1, y2, x1, x2): the bottom 30 %
```

Without `region`, `SubtitleRegionDetector` looks for the band on a few frames and keeps it for the
whole video. Giving it is safer: look at one frame, and leave room above the text, since a second
line of subtitle sits higher.

## 3. Read and merge

```python
from frame2text4llm.ocr import OCRBatchProcessor

subtitles = OCRBatchProcessor(manager).process_batch(
    frames, tool="easyocr", lang="fr", n_cores=1, merge_segments=True
)
for s in subtitles[:3]:
    print(s["start_time"], s["end_time"], s["text"])
```

Each subtitle has `start_time`, the first frame that shows it; `end_time`, the first frame that no
longer does; `text`, the longest reading; and `best_time`, the frame that reading came from.

## 4. Cut the audio

```python
from frame2text4llm.export.audio_generator import AudioGenerator

paths = AudioGenerator("episode.mp4", tmp_audio_dir="cuts").extract_segments(subtitles)
```

One WAV per subtitle, 16 kHz mono. The subtitle appeared somewhere between the frame before
`start_time` and `start_time` itself: to be sure the cut holds the start of the line, open it half a
second earlier, as [the YouTube example](examples/youtube-dataset.md) does.

## 5. Export

```python
from frame2text4llm.export import JSONExporter, SRTExporter

SRTExporter().export(subtitles, "episode.srt")
JSONExporter().export(subtitles, "episode.json")
```

EasyOCR is fast and often wrong on details. For text you will train on, read each subtitle again with
a better engine: [Choosing an OCR engine](guide/engines.md).
