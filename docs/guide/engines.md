# Choosing an OCR engine

Frame2Text4LLM can read a subtitle with several engines. They differ a lot, and not always in the
way their reputation suggests. This page gives the numbers we measured, and what we use.

## What we measured

48 subtitle bands were drawn at random, 12 from each of four Burkinabè series and films with French
subtitles burned in: a TV series, a web series, a vertical sketch filmed on a phone, and a 2005 film
in 480x360. Each was read by eye: 32 show a subtitle, 16 show none. Each engine then read the same
48 images on a Google Colab T4.

The error is the character error rate (`jiwer.cer`) on the 32 subtitles, once apostrophes and
quotes are brought to one form and spaces to one. The 16 empty images are counted apart: an engine
that reads text where there is none puts false subtitles in your dataset.

| engine | how it was called | character error | empty images read as text | seconds per image |
|---|---|---|---|---|
| Qwen3-VL-2B | `tool="vlm"`, image doubled, float32 on GPU | **0.53 %** | 0 | 1.18 |
| PaddleOCR | `tool="paddleocr"`, CPU | 3.48 % | 3 | 1.38 |
| EasyOCR, paragraphs | `readtext(paragraph=True)` on the image doubled | 10.45 % | 0 | 0.11 |
| EasyOCR | `tool="easyocr"` | 37.49 % | 0 | 0.05 |

What the numbers hide:

- **Qwen3-VL-2B** differed from the eye on 3 subtitles out of 32: a full stop added, a letter
  dropped, and one line wider than the picture, cut at the source.
- **PaddleOCR** glues words together ("Ne t'avais-jepas demande") and read Chinese characters on
  three empty images. On CPU it needs `enable_mkldnn=False`, or it fails on its first image.
- **EasyOCR** loses most of its accuracy to how it is called: `tool="easyocr"` keeps the words in
  the order the detector found them and drops those below 0.5 confidence. Read as paragraphs on a
  doubled image, its error falls by a factor of 3.6.

These are four videos and 48 images. Measure on yours before trusting a table made on someone
else's: [Measuring an engine on your videos](#measuring-an-engine-on-your-videos) below.

## What we use: two passes

A good reader is slow and a fast reader is wrong, but the fast one is good enough to tell **where**
a subtitle starts and ends. So:

1. EasyOCR reads every frame, two per second, and `merge_segments=True` turns the frames into
   subtitles. Two readings that share most of their words are one subtitle, however differently
   the engine spelled them.
2. Qwen3-VL-2B reads each subtitle **once**, on the frame where EasyOCR read the most text
   (`best_time`).

On the four videos above, the first pass took 20 minutes for 2.3 hours of video and found 1,557
subtitles; the second pass costs about 1.2 seconds per subtitle on a T4.
[A YouTube video to a speech dataset](../examples/youtube-dataset.md) shows the code.

## Measuring an engine on your videos

Draw a few dozen frames at random, write down what each one says (`-` for no subtitle), and compare:

```python
import jiwer

truth = ["Si vous nous aidez, cela nous fera un grand bien.", "-"]
read = [manager.process(image, tool="easyocr", lang="fr") for image in images]

subtitles = [(t, r) for t, r in zip(truth, read) if t != "-"]
print("character error:", jiwer.cer([t for t, _ in subtitles], [r for _, r in subtitles]))
print("empty images read as text:", sum(bool(r) for t, r in zip(truth, read) if t == "-"))
```

Draw at random, not the frames that look easy: the hard ones are where engines differ.
