# How it works

Each step is one class, and each can be used alone.

| step | class | what it does |
|---|---|---|
| download | `consumer.youtube_consumer.YoutubeConsumer` | a video or a playlist from YouTube, with pytubefix |
| frames | `framer.VideoReader` | frames at a chosen rate, each with its time |
| band | `framer.SubtitleRegionDetector` | where the subtitles are, if you do not say |
| OCR | `ocr.OCRManager`, `ocr.OCRBatchProcessor` | one engine on one frame, or on all of them |
| subtitles | `process_batch(merge_segments=True)` | frames that show one subtitle become one segment |
| audio | `export.audio_generator.AudioGenerator` | one WAV per segment, with ffmpeg |
| export | `export.SRTExporter`, `export.JSONExporter`, `export.HuggingfaceCooker` | files, or a Hugging Face dataset |

## Frames

`VideoReader.extract_frames(target_fps=2)` keeps one frame in `round(video_fps / 2)`: at 25 frames
per second, one every 0.48 s. Each `VideoFrame` has `image` (BGR, as OpenCV reads it), `time` in
seconds and `time_formatted` (`MM:SS:mmm`).

`filter_duplicates=True` drops a frame whose picture hash is close to the previous kept frame's.
It saves OCR time on static shots, but the hash looks at the whole picture, and a subtitle that
changes on a still shot is a small change: the new subtitle is dropped with the frame. Leave it off
when you cut audio.

## Merging frames into subtitles

An OCR engine does not read the same line the same way twice: one frame loses a word, the next an
accent, the third reads only one of two lines. If every new reading opened a segment, a subtitle
that stays five seconds on screen would come out as five one-second pieces, each with its audio cut
short.

`merge_segments=True` treats two readings as one subtitle when either

- they look alike: `difflib` similarity of at least `sim_thresh` (0.8), or
- at least half the words of the shorter one, three letters or more, accents ignored, are in the
  other (`min_shared`, 0.5),

and one frame read as empty inside a subtitle does not end it (`max_gap`, 1). A segment ends on the
first frame that shows something else, or nothing. It keeps the longest reading as `text`, and the
frame of that reading as `best_time`.

On one episode, this turned 224 one-second pieces into 126 subtitles of four seconds. Two short
subtitles in a row that share half their words ("Oui papa", then "Non papa") become one: the
price of the rule.

## Audio

`AudioGenerator(video, tmp_audio_dir).extract_segments(segments)` runs ffmpeg once per segment and
writes `segment_0000.wav` and on, 16 kHz mono, from `start_time` to `end_time`. Times are read as
`MM:SS:mmm`, `HH:MM:SS` or `HH:MM:SS.mmm`.
