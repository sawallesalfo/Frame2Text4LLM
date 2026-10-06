# Changelog

## [0.1.0] - 2026-10-06

First release on PyPI.

### Added

- `pyproject.toml`, built with hatchling, and `uv.lock` for development. The OCR engines are
  extras: `easyocr`, `paddle`, `vlm`, `openai`, `all`.
- `VLMOCR` reads with any image-text-to-text model of transformers, Qwen3-VL-2B by default: on
  48 subtitle frames it made 0.53 % character error, against 37.49 % for `tool="easyocr"`.
- `merge_segments`: readings of one subtitle are merged when they share most of their words, one
  empty frame does not end a subtitle, and each segment has `best_time`, the frame read best.
- `OCRManager(region=...)`: the subtitle band, detected once per video when it is not given.
- `YoutubeConsumer(client=...)`: the pytubefix client, `MWEB` by default; files are named
  `youtube_<id>.mp4`.
- A documentation site: <https://sawallesalfo.github.io/Frame2Text4LLM/>.

### Fixed

- A segment ends when its subtitle leaves the screen; a subtitle seen on one frame no longer lasts
  0 s.
- `AudioGenerator` reads the frames' `MM:SS:mmm` times and seeks before reading: audio was cut in
  the wrong place.
- `SubtitleRegionDetector` takes the full width of the image; its margin cut the first letters.
- `OCRManager` loads a model once, not once per image.
- `SRTExporter` writes valid times past an hour, and each subtitle's own end.

## [0.0.1]

- First version, not published.
