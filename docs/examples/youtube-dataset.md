# A YouTube video to a speech dataset

The recipe BurkimbIA runs on Burkinabè series: EasyOCR finds the subtitles, Qwen3-VL-2B reads each
one once, and every subtitle gets its audio. It needs a GPU for the second pass; a free Colab T4 is
enough for a few hours of video.

```bash
pip install "frame2text4llm[easyocr,vlm]"
```

```python
from frame2text4llm.consumer.youtube_consumer import YoutubeConsumer
from frame2text4llm.export.audio_generator import AudioGenerator
from frame2text4llm.framer import VideoReader
from frame2text4llm.ocr import OCRBatchProcessor, OCRManager

# 1. Download. Files are named youtube_<id>.mp4, so a second run finds them.
path = YoutubeConsumer(output_dir="videos").consume_video("https://www.youtube.com/watch?v=PzKZpqFzWro")

# 2. Find the subtitles: EasyOCR on two frames per second, readings of one subtitle merged.
reader = VideoReader(path)
frames = reader.extract_frames(target_fps=2)
h, w = frames[0].image.shape[:2]
manager = OCRManager(reader, region=(int(h * 0.7), h, 0, w))
subtitles = OCRBatchProcessor(manager).process_batch(
    frames, tool="easyocr", lang="fr", n_cores=1, merge_segments=True
)

# 3. Read each subtitle once with Qwen3-VL-2B, on the frame EasyOCR read best.
image = {f.time_formatted: f.image for f in frames}
prompt = ("Recopie exactement le sous-titre incrusté dans cette image, sur une seule ligne, "
          "sans rien ajouter. S'il n'y a pas de sous-titre, réponds seulement : -")
vlm = manager.get_tool_instance("vlm", model_name="Qwen/Qwen3-VL-2B-Instruct")
vlm.prompt = prompt
for s in subtitles:
    s["text"] = manager.process(image[s["best_time"]], tool="vlm")

# 4. Cut the audio, from half a second before the first frame that shows the subtitle.
def earlier(t, by=0.5):
    m, sec, ms = map(int, t.split(":"))
    x = max(0, round((m * 60 + sec + ms / 1000 - by) * 1000))
    return f"{x // 60000:02d}:{x // 1000 % 60:02d}:{x % 1000:03d}"

cuts = [{"start_time": earlier(s["start_time"]), "end_time": s["end_time"]} for s in subtitles]
for s, wav in zip(subtitles, AudioGenerator(path, tmp_audio_dir="cuts").extract_segments(cuts)):
    s["audio"] = wav
```

The prompt is the one we measured, in French because the subtitles are; write it in the language
of yours. `subtitles` is now a list of dicts with `text`, `audio`, `start_time` and `end_time`:
`datasets.Dataset.from_list(subtitles).cast_column("audio", datasets.Audio())` makes it a Hugging
Face dataset.

## Long videos

`extract_frames` keeps every frame whole in memory: a 97-minute film at two frames per second is
about 6 GB at 480x360. Above an hour, read the frames yourself and keep only the band:

```python
import cv2
from frame2text4llm.framer.base import VideoFrame

def bands(path, top=0.7, fps=2):
    cap = cv2.VideoCapture(path)
    rate = cap.get(cv2.CAP_PROP_FPS)
    step, frames, n = max(1, round(rate / fps)), [], 0
    while cap.grab():
        if n % step == 0:
            f = cap.retrieve()[1]
            frames.append(VideoFrame(f[int(f.shape[0] * top):], n / rate, n))
        n += 1
    return frames
```

Then give `OCRManager` the whole band as its region: `region=(0, band_height, 0, band_width)`.
