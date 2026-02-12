import io
import cv2
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
from scipy.fft import dct
from difflib import SequenceMatcher #find similarity between 2 strings
from vlm_service import VLMService


def extract_frames(video_path, sample_rate=1):
    """yield (frame_no, timestamp_sec, gray_frame)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(fps / sample_rate))
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no % step == 0:
            ts = frame_no / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield frame_no, ts, gray
        frame_no += 1
    cap.release()



def _phash(image, hash_size=8, highfreq_factor=4):
    """computes a perceptual hash (fingerprint) of an image"""
    img = cv2.resize(image, (hash_size * highfreq_factor,)*2)
    d = dct(dct(img.astype(float), axis=0), axis=1)
    low = d[:hash_size, :hash_size]
    return (low > np.median(low)).flatten()

def filter_duplicates(frames, diff_thresh=10):
    kept, prev = [], None
    for fn, ts, img in frames:
        h = _phash(img)
        if prev is None or np.count_nonzero(h != prev) > diff_thresh:
            kept.append((fn, ts, img))
            prev = h
    return kept



def _similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def merge_segments(ocr_res, sim_thresh=0.8):
    if not ocr_res:
        return []
    segs = []
    cur_txt = ocr_res[0][2]
    start, end = ocr_res[0][1], ocr_res[0][1]
    for _, ts, txt in ocr_res[1:]:
        if _similar(cur_txt, txt) >= sim_thresh: #if text == cur_text:
            end = ts
            # keep longer
            if len(txt) > len(cur_txt):
                cur_txt = txt
        else:
            segs.append((start, end, cur_txt))
            cur_txt, start, end = txt, ts, ts
    segs.append((start, end, cur_txt))
    return segs



def _crop_subtitle_band(frame, height_ratio=0.2):
    """Crop bottom `height_ratio` of the frame."""
    h = frame.shape[0]
    return frame[int(h * (1 - height_ratio)):, :]

def _format_ts(sec):
    ms = int((sec - int(sec)) * 1000)
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def video_to_subs_json(
        video_path, 
        sample_rate=2, 
        vlm_model="microsoft/Florence-2-base"
    ):
    vlm = VLMService(model_name=vlm_model)

    frames = list(extract_frames(video_path, sample_rate))
    frames = filter_duplicates(frames)

    ocrs = []
    for fn, ts, img in tqdm(frames, desc="VLM OCR"):
        crop = _crop_subtitle_band(img)
        # convert the gray crop to JPEG bytes
        pil = Image.fromarray(crop)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG")
        text = vlm.perform_ocr(buf.getvalue())
        #print(text)
        if text:
            if vlm_model == "microsoft/Florence-2-base":
                ocrs.append((fn, ts, text['<OCR>'].strip()))
            elif vlm_model == "OpenGVLab/InternVL2-1B":
                ocrs.append((fn, ts, text.strip()))

    segs = merge_segments(ocrs)

    return [
        {"start_time": _format_ts(s), "end_time": _format_ts(e), "text": t}
        for s, e, t in segs
    ]



subs = video_to_subs_json(
    "sample_short.mp4", 
    sample_rate=2, 
    vlm_model="OpenGVLab/InternVL2-1B"
)
with open(f"resultat_first_tests_internvl2.json", "w", encoding="utf-8") as f:
    json.dump(subs, f, ensure_ascii=False, indent=2)
