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

```
DOWNLOAD video from URL using yt-dlp
SAVE to data/raw_videos/moore_XXX.mp4
```

### Step 3: Extract French subtitles (OCR via frame2text4llm)

This is where our package comes in. It extracts video frames, detects the subtitle area, and runs OCR to get the French text.

```
LOAD video using VideoReader
EXTRACT frames at 1 frame per second
RUN OCR on each frame (tool: tesseract or paddleocr, lang: fra)
GROUP and CLEAN text by time window (e.g. 10 seconds)

OUTPUT -> list of {start_time, end_time, french_text}
```

**Not working well?** That's normal, it's iterative. Open an issue on the repo with the problematic case. We improve the package together.

### Step 4: Extract and transcribe audio (ASR Whisper)

**4a. Extract audio from video:**

```
EXTRACT audio from video using ffmpeg
CONVERT to WAV, 16kHz, mono
SAVE to data/audio/moore_XXX.wav
```

**4b. Load the ASR model:**

```python
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os

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
```

**4c. Transcribe each segment:**

```
FOR each segment from Step 3:
    LOAD audio chunk from start_time to end_time
    RUN Whisper model on the chunk
    GET Moore transcription
```

### Step 5: Align and create the dataset

```
FOR each segment:
    PAIR the Moore transcription (from ASR) with the French subtitle (from OCR)
    ADD metadata: video_id, start_time, end_time, duration, source_url, contributor
    APPEND to dataset

SAVE dataset as JSONL -> data/datasets/moore_XXX.jsonl
```

### Step 6: Split and save audio segments

```
FOR each segment in the dataset:
    CUT audio from start_time to end_time using ffmpeg
    SAVE as WAV (16kHz, mono) -> data/audio_segments/moore_XXX_start_end.wav
```

### Step 7: Publish your dataset on YOUR Hugging Face account

Each contributor publishes their dataset on **their own** Hugging Face account. It's your work, your name is on it. Be proud of it.

```
LOAD your JSONL file
CREATE a Hugging Face Dataset (with Audio column)
PUSH to your account: {your_username}/moore-dataset-{your_name}
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

```
LOAD registry/video_registry.csv
CHECK if your video URL already exists
IF exists -> find another video
IF not -> add it with status=in_progress
```

---

## 7. Final Merge - The Big Dataset

Everyone produces and publishes their dataset individually on their Hugging Face account. At the end, we merge everything. Like a good to: everyone pounds their own millet, but we all eat together.

```
FOR each contributor dataset on Hugging Face:
    LOAD dataset from {username}/moore-dataset-{name}

CONCATENATE all datasets into one
PUBLISH final merged dataset -> sawadogosalif/moore-speech-dataset
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
