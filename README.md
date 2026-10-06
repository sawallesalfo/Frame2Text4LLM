# Frame2Text4LLM

[![PyPI](https://img.shields.io/pypi/v/frame2text4llm)](https://pypi.org/project/frame2text4llm/)
[![Tests](https://github.com/sawallesalfo/Frame2Text4LLM/actions/workflows/tests.yml/badge.svg)](https://github.com/sawallesalfo/Frame2Text4LLM/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/badge/docs-github%20pages-orange)](https://sawallesalfo.github.io/Frame2Text4LLM/)

**By Alban Nyantudre and Salif Sawadogo**, for [BurkimbIA](https://huggingface.co/burkimbia).

Read the subtitles burned into a video with OCR, find when each one starts and ends, and cut the
audio underneath. What comes out is pairs of speech and text, for speech recognition or speech
translation datasets in languages that have few of them.

![Frame2Text4LLM](https://raw.githubusercontent.com/sawallesalfo/Frame2Text4LLM/main/assets/diagram_frame2text.png)

## Install

```bash
pip install "frame2text4llm[easyocr]"
```

OCR engines are extras: `easyocr`, `paddle`, `vlm` (Qwen3-VL, Florence-2, InternVL2), `openai`, or
`all`. Audio cuts need [ffmpeg](https://ffmpeg.org/).

## Use

```python
from frame2text4llm.framer import VideoReader
from frame2text4llm.ocr import OCRBatchProcessor, OCRManager
from frame2text4llm.export.audio_generator import AudioGenerator

reader = VideoReader("episode.mp4")
frames = reader.extract_frames(target_fps=2)
h, w = frames[0].image.shape[:2]
manager = OCRManager(reader, region=(int(h * 0.7), h, 0, w))
subtitles = OCRBatchProcessor(manager).process_batch(
    frames, tool="easyocr", lang="fr", n_cores=1, merge_segments=True
)
audio = AudioGenerator("episode.mp4", tmp_audio_dir="cuts").extract_segments(subtitles)
```

## Which engine

Measured on 48 subtitle frames from four Burkinabè series with French subtitles:

| engine | character error | seconds per image (T4) |
|---|---|---|
| Qwen3-VL-2B (`tool="vlm"`) | 0.5 % | 1.18 |
| PaddleOCR | 3.5 % | 1.38 (CPU) |
| EasyOCR | 37.5 % | 0.05 |

We find subtitles with EasyOCR and read each one once with Qwen3-VL-2B.
[Choosing an OCR engine](https://sawallesalfo.github.io/Frame2Text4LLM/guide/engines/) has the details.

## Documentation

<https://sawallesalfo.github.io/Frame2Text4LLM/>

## Develop

```bash
uv sync
uv run pytest
uv run --only-group docs mkdocs serve
```

To release, raise `__version__` in `src/frame2text4llm/package_metadata.py`, add the version to
`HISTORY.md`, merge, and run the *Publish to PyPI* workflow.

## Authors

- **Alban Nyantudre** ([GitHub](https://github.com/anyantudre))
- **Salif Sawadogo** ([GitHub](https://github.com/sawallesalfo))

Contact: frame2text4llm@gmail.com. MIT licence.
