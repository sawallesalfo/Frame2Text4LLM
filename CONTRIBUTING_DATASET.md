# Contributor Guide - Moore Collaborative Dataset

> *"If you want to go fast, go alone. If you want a 100,000-hour Moore dataset, call the crew."*

![Contributor User Journey](user%20journey.jpg)

---

## 1. Project Vision

Let's be honest: current LLMs don't speak Moore. And when they try, it sounds like a toubabou saying "laafi bala" with a Parisian accent. We're here to fix that.

We are building a **massive collaborative dataset** for **Moore** - all dialects, all accents, from Ouaga to Koudougou to Koupela. The goal is to collect enough data to train and fine-tune models for:

- **ASR (Automatic Speech Recognition)**: So machines can finally understand Moore
- **TTS (Text-to-Speech)**: So machines can speak Moore (without a weird accent)
- **Fine-tuning reasoning models**: Moore <-> French pairs for knowledge transfer
- **Machine translation**: Moore <-> French

**The idea is simple (the work, not so much):** The internet is full of videos spoken in Moore with burned-in French subtitles. We will:
1. Extract the French subtitles using OCR (our `frame2text4llm` package)
2. Transcribe the Moore audio using ASR (fine-tuned Whisper model)
3. Align both to get parallel pairs: audio + Moore transcription + French translation

**This work is iterative.** As you process videos, you will inevitably hit cases where the package doesn't work well: OCR gone wrong, weird video format, poorly detected subtitles. That's **normal and that's the point**. Every bug found = an issue opened = a package improvement. Every package improvement = a better dataset. It's a virtuous cycle: **the more we collect, the better the package gets, the better we collect.**

```
    ┌──────────────┐
    │   Collect    │
    │   videos     │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐         ┌──────────────────┐
    │  Generate    │────────>│  Find bugs /     │
    │  dataset     │         │  limitations     │
    └──────────────┘         └────────┬─────────┘
           ▲                          │
           │                          ▼
    ┌──────┴───────┐         ┌──────────────────┐
    │  Better      │<────────│  Improve the     │
    │  dataset     │         │  package         │
    └──────────────┘         └──────────────────┘
```

---

## 2. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FULL PIPELINE                                │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │ 1. VIDEO │───>│ 2. EXTRACT   │───>│ 3. OCR       │              │
│  │ Download │    │    Frames    │    │  French      │              │
│  │ (yt-dlp) │    │(frame2text)  │    │  Subtitles   │              │
│  └──────────┘    └──────────────┘    └──────┬───────┘              │
│       │                                      │                      │
│       │          ┌──────────────┐    ┌───────▼───────┐             │
│       └─────────>│ 4. EXTRACT   │───>│ 6. ALIGNMENT  │             │
│                  │    Audio     │    │  & MERGE      │             │
│                  └──────┬───────┘    └───────┬───────┘             │
│                         │                    │                      │
│                  ┌──────▼───────┐    ┌───────▼───────┐             │
│                  │ 5. ASR       │    │ 7. DATASET    │             │
│                  │ Whisper      │    │    Final      │             │
│                  │ Transcription│    │               │             │
│                  └──────────────┘    └───────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Prerequisites

### 3.1 Environment

- **Python** >= 3.8
- **FFmpeg** installed and available in PATH
- **Tesseract OCR** installed (for subtitle extraction)
- **GPU recommended** (for Whisper ASR) - if you've run out of GPU quota, don't panic: we have a **shared Google Colab account** for the project. Ask the maintainers for access
- **A Hugging Face account** with your own username (you'll publish your dataset there)
- **OpenAI API**: A shared account is available for OCR via the OpenAI engine. Use it **sparingly** (it costs money, and we're not OpenAI)

### 3.2 Installation

```bash
# 1. Clone the repo
git clone https://github.com/sawadogosalif/Frame2Text4LLM.git
cd Frame2Text4LLM

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3. Install the package in dev mode
pip install -e .

# 4. Install extra dependencies for the dataset pipeline
pip install yt-dlp transformers torch torchaudio datasets soundfile librosa

# 5. Install Whisper for ASR
pip install openai-whisper
```

### 3.3 Tokens and Keys

```bash
# Create a .env file at the project root
# DO NOT COMMIT THIS FILE (or the group chat will never let you forget it)

# Hugging Face token for the ASR model (will be provided by maintainers)
HF_TOKEN_ASR=hf_token_for_asr_model

# YOUR personal Hugging Face token (to publish your dataset on your account)
HF_TOKEN=hf_your_personal_token
```

The ASR model used is: **`sawadogosalif/WHISPER-LARGE`** on Hugging Face. The access token will be shared with you directly.

---

## 4. Contribution Workflow - Step by Step

> Each contributor works on their own videos and publishes their dataset on **their own Hugging Face account**. At the end, we merge everything into one big dataset. Like a good zoom-koom: everyone brings their share.

### Step 1: Pick and register your videos

Before starting, **register your videos** in `registry/video_registry.csv` to avoid duplicates. If two people process the same Moore sermon video, that's wasted time (and we don't have enough of it).

```csv
video_id,url,platform,has_french_subtitles,contributor,status,date_added
moore_001,https://youtube.com/watch?v=XXXXX,youtube,yes,first_last,in_progress,2025-01-15
moore_002,https://facebook.com/watch/?v=YYYYY,facebook,yes,first_last,pending,2025-01-15
```

**Rules:**
- Check that the URL doesn't already exist in the registry before adding it
- Use the format `moore_{number}` for the `video_id`
- Set `status=in_progress` when you start processing a video
- Set `status=done` when processing is complete
- The video **must** be spoken in Moore with visible French subtitles (burned-in)

### Step 2: Download the video

```python
import subprocess

video_url = "https://youtube.com/watch?v=XXXXX"
output_path = "data/raw_videos/moore_001.mp4"

# Download with yt-dlp
subprocess.run([
    "yt-dlp",
    "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
    "-o", output_path,
    video_url
])
```

### Step 3: Extract French subtitles (OCR via frame2text4llm)

This is where our package comes in. It extracts video frames, detects the subtitle area, and runs OCR to get the French text.

```python
from frame2text4llm.framer.video import VideoReader
from frame2text4llm.ocr.manager import OCRManager
from frame2text4llm.ocr.batch import OCRBatchProcessor
from frame2text4llm.ocr.utils.group import group_and_clean_text

# 1. Load the video
video_reader = VideoReader("data/raw_videos/moore_001.mp4", engine="opencv")
video_reader.print_info()

# 2. Extract frames (1 frame per second recommended)
frames = video_reader.extract_frames(target_fps=1)

# 3. Run OCR on frames to extract French subtitles
ocr_manager = OCRManager(video_reader)
processor = OCRBatchProcessor()

results = processor.process_batch(
    frames=frames,
    tool="tesseract",       # or "paddleocr" for French
    lang="fra",             # French
    n_cores=4,
    show_progress=True
)

# 4. Clean and group text by time window
grouped_subtitles = group_and_clean_text(results, window_sec=10)

# grouped_subtitles = [
#     {"start_time": "00:00:00", "end_time": "00:00:10", "aggregated_text": "Bonjour a tous..."},
#     {"start_time": "00:00:10", "end_time": "00:00:20", "aggregated_text": "Aujourd'hui nous..."},
#     ...
# ]
```

**Not working well?** That's normal, it's iterative. Open an issue on the repo with the problematic case. We improve the package together.

### Step 4: Extract and transcribe audio (ASR Whisper)

```python
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import subprocess
import os

# 1. Extract audio from the video
video_path = "data/raw_videos/moore_001.mp4"
audio_path = "data/audio/moore_001.wav"

subprocess.run([
    "ffmpeg", "-i", video_path,
    "-vn",                    # no video
    "-acodec", "pcm_s16le",  # WAV format
    "-ar", "16000",           # 16kHz (required by Whisper)
    "-ac", "1",               # mono
    audio_path
])

# 2. Load the ASR model
HF_TOKEN_ASR = os.getenv("HF_TOKEN_ASR")

processor = WhisperProcessor.from_pretrained(
    "sawadogosalif/WHISPER-LARGE",
    token=HF_TOKEN_ASR
)
model = WhisperForConditionalGeneration.from_pretrained(
    "sawadogosalif/WHISPER-LARGE",
    token=HF_TOKEN_ASR
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 3. Transcribe segments aligned with OCR subtitles
def transcribe_segment(audio_path, start_sec, end_sec, processor, model, device):
    """Transcribe an audio segment in Moore."""
    audio, sr = librosa.load(audio_path, sr=16000, offset=start_sec, duration=end_sec - start_sec)
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription
```

### Step 5: Align and create the dataset

```python
import json

def parse_time_to_seconds(time_str):
    """Convert 'HH:MM:SS' to seconds."""
    parts = time_str.split(":")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return h * 3600 + m * 60 + s

# Build the aligned dataset
dataset_entries = []

for segment in grouped_subtitles:
    start_sec = parse_time_to_seconds(segment["start_time"])
    end_sec = parse_time_to_seconds(segment["end_time"])
    french_text = segment["aggregated_text"]

    # Transcribe the audio segment in Moore
    moore_transcription = transcribe_segment(
        audio_path, start_sec, end_sec, processor, model, device
    )

    entry = {
        "video_id": "moore_001",
        "start_time": segment["start_time"],
        "end_time": segment["end_time"],
        "duration_sec": end_sec - start_sec,
        "audio_file": f"audio_segments/moore_001_{start_sec}_{end_sec}.wav",
        "transcription_moore": moore_transcription,
        "translation_fr": french_text,
        "source_url": "https://youtube.com/watch?v=XXXXX",
        "contributor": "first_last"
    }
    dataset_entries.append(entry)

# Save as JSONL (one entry per line)
output_path = "data/datasets/moore_001.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for entry in dataset_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Dataset generated: {len(dataset_entries)} entries -> {output_path}")
```

### Step 6: Split and save audio segments

```python
def extract_audio_segments(audio_path, dataset_entries, output_dir):
    """Split audio into individual segments."""
    os.makedirs(output_dir, exist_ok=True)

    for entry in dataset_entries:
        start_sec = parse_time_to_seconds(entry["start_time"])
        end_sec = parse_time_to_seconds(entry["end_time"])
        segment_path = os.path.join(output_dir, os.path.basename(entry["audio_file"]))

        subprocess.run([
            "ffmpeg", "-i", audio_path,
            "-ss", str(start_sec),
            "-to", str(end_sec),
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            segment_path
        ], capture_output=True)

    print(f"Audio segments extracted: {len(dataset_entries)} files -> {output_dir}")

extract_audio_segments(
    audio_path="data/audio/moore_001.wav",
    dataset_entries=dataset_entries,
    output_dir="data/audio_segments/"
)
```

### Step 7: Publish your dataset on YOUR Hugging Face account

Each contributor publishes their dataset on **their own** Hugging Face account. It's your work, your name is on it. Be proud of it.

```python
from datasets import Dataset, Audio
import json
import os

HF_TOKEN = os.getenv("HF_TOKEN")  # your personal token

# 1. Load your JSONL
entries = []
with open("data/datasets/moore_001.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        entries.append(json.loads(line))

# 2. Create Hugging Face dataset
dataset = Dataset.from_list(entries)
dataset = dataset.cast_column("audio_file", Audio(sampling_rate=16000))

# 3. Publish on YOUR account
# Replace "your_hf_username" with your actual Hugging Face username
dataset.push_to_hub(
    "your_hf_username/moore-dataset-contribution",
    token=HF_TOKEN
)

print("Dataset published! You're making Moore AI happen.")
```

**Naming convention on Hugging Face:** `{your_username}/moore-dataset-{your_name}`

Examples:
- `salif_sawadogo/moore-dataset-salif`
- `alban_ny/moore-dataset-alban`

---

## 5. Dataset Format

### 5.1 File structure per contributor

```
data/
├── raw_videos/                    # Raw videos (DO NOT COMMIT - too heavy)
│   └── moore_001.mp4
├── audio/                         # Full extracted audio
│   └── moore_001.wav
├── audio_segments/                # Split audio segments
│   ├── moore_001_0_10.wav
│   ├── moore_001_10_20.wav
│   └── ...
└── datasets/                      # JSONL datasets per video
    ├── moore_001.jsonl
    ├── moore_002.jsonl
    └── ...
```

### 5.2 JSONL entry schema

| Field | Type | Required | Description |
|---|---|---|---|
| `video_id` | string | yes | Unique identifier (format: `moore_{num}`) |
| `start_time` | string | yes | Segment start (HH:MM:SS) |
| `end_time` | string | yes | Segment end (HH:MM:SS) |
| `duration_sec` | float | yes | Duration in seconds |
| `audio_file` | string | yes | Relative path to the audio segment file |
| `transcription_moore` | string | yes | ASR transcription in Moore |
| `translation_fr` | string | yes | French subtitle extracted by OCR |
| `source_url` | string | yes | Source video URL |
| `contributor` | string | yes | Contributor identifier |
| `accent_region` | string | no | Region/accent variant (nice-to-have) |
| `speaker_gender` | string | no | Speaker gender (nice-to-have) |
| `topic` | string | no | Video topic (nice-to-have) |
| `quality_score` | float | no | Confidence score 0-1 (nice-to-have) |

### 5.3 Example entry

```json
{
    "video_id": "moore_001",
    "start_time": "00:01:30",
    "end_time": "00:01:40",
    "duration_sec": 10.0,
    "audio_file": "audio_segments/moore_001_90_100.wav",
    "transcription_moore": "Wend na ko tond laafi",
    "translation_fr": "Que Dieu nous donne la paix",
    "source_url": "https://youtube.com/watch?v=XXXXX",
    "contributor": "salif_sawadogo"
}
```

---

## 6. Video Registry (No Duplicates)

The file `registry/video_registry.csv` is the **central coordination point**. No registry, no clean dataset. It's like land with no cadastre: total mess.

### Workflow:

1. **Before starting**: Pull the repo, check the registry
2. **Reserve your videos**: Add them with `status=in_progress`
3. **Open a PR** with your reservation to avoid conflicts
4. **Process your videos** locally
5. **Update status** to `done` when finished
6. **Publish your dataset** on your own Hugging Face account
7. **Update the registry** with your HF dataset link

### Check for duplicates:

```python
import pandas as pd

registry = pd.read_csv("registry/video_registry.csv")

# Check if a URL already exists
url = "https://youtube.com/watch?v=XXXXX"
if url in registry["url"].values:
    print("WARNING: This video is already taken! Find another one.")
else:
    print("OK: Video available, go for it.")

# See stats
print(f"Total videos: {len(registry)}")
print(f"In progress: {(registry['status'] == 'in_progress').sum()}")
print(f"Done: {(registry['status'] == 'done').sum()}")
```

---

## 7. Final Merge - The Big Dataset

Everyone produces and publishes their dataset individually on their Hugging Face account. At the end, we merge everything. Like a good to: everyone pounds their own millet, but we all eat together.

```python
from datasets import load_dataset, concatenate_datasets

# List of individual datasets published by contributors
contributor_datasets = [
    "salif_sawadogo/moore-dataset-salif",
    "alban_ny/moore-dataset-alban",
    "contributor3/moore-dataset-contributor3",
    # ... add all contributors
]

# 1. Load and merge all datasets
all_datasets = []
for repo_id in contributor_datasets:
    ds = load_dataset(repo_id, split="train")
    all_datasets.append(ds)
    print(f"  {repo_id}: {len(ds)} entries")

merged = concatenate_datasets(all_datasets)
print(f"\nMerged dataset: {len(merged)} total entries")
print(f"Contributors: {len(contributor_datasets)}")

# 2. Publish the final dataset
merged.push_to_hub(
    "sawadogosalif/moore-speech-dataset",
    token=HF_TOKEN
)
```

---

## 8. Contributor Checklist

### Before you start
- [ ] Python environment set up with all dependencies
- [ ] Hugging Face tokens (ASR + personal) configured in `.env`
- [ ] FFmpeg and Tesseract installed
- [ ] Repo cloned and branch up to date
- [ ] Hugging Face account created with your username

### For each video
- [ ] URL checked in the registry (no duplicate)
- [ ] Video added to registry with `status=in_progress`
- [ ] Video downloaded
- [ ] Frames extracted and OCR done (French subtitles)
- [ ] Audio extracted and transcribed (Moore ASR)
- [ ] Audio segments split
- [ ] JSONL dataset generated
- [ ] Manual check of a sample (~5-10 entries)
- [ ] Dataset published on your Hugging Face account
- [ ] Registry updated with `status=done`

### Something not working?
- [ ] Open an issue on the repo with the problematic case
- [ ] The package will be improved, and everyone benefits

> *"You don't leave a pothole on the road just because you drive a 4x4."*

---

## 9. Contact and Coordination

- **Repository**: https://github.com/sawadogosalif/Frame2Text4LLM
- **Email**: frame2text4llm@gmail.com
- **ASR Model**: https://huggingface.co/sawadogosalif/WHISPER-LARGE
- **Maintainers**: Alban NYANTUDRE, Salif SAWADOGO

---

*This document is alive, just like the project. Feel free to suggest improvements via PR.*
