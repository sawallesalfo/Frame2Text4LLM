import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional
from loguru import logger


def _similar(a: str, b: str) -> float:
        """Calculate similarity between two strings."""
        return SequenceMatcher(None, a, b).ratio()


def _words(text: str) -> set:
    """Words of three letters or more, lowercased and without accents."""
    text = unicodedata.normalize("NFD", text.lower())
    return set(re.findall(r"[^\W\d_]{3,}", "".join(c for c in text if not unicodedata.combining(c))))


def _same_subtitle(a: str, b: str, sim_thresh: float, min_shared: float) -> bool:
    """Two readings of one subtitle: they look alike, or the shorter one's words are mostly in the
    other, as when the OCR skipped a line or a word on one frame."""
    wa, wb = _words(a), _words(b)
    shared = len(wa & wb) / min(len(wa), len(wb)) if wa and wb else 0
    # ponytail: two short subtitles in a row that share half their words ("Oui papa", "Non papa") become one
    return shared >= min_shared or _similar(a, b) >= sim_thresh


def _merge_segments(ocr_results: List[Dict[str, Any]], sim_thresh: float = 0.8,
                    min_shared: float = 0.5, max_gap: int = 1) -> List[Dict[str, Any]]:
    """Merge consecutive frames that show the same text into one segment.

    The OCR reads one subtitle differently from frame to frame: a line skipped, an accent, a word
    lost. Two readings are the same subtitle when they look alike (`sim_thresh`) or when at least
    `min_shared` of the shorter one's words are in the other; and up to `max_gap` frames read as
    empty inside a subtitle do not end it. The segment keeps the longest reading, and `best_time`
    is the frame it was read on, the one to read again with a better OCR.

    A segment ends at the first frame that shows other text, or none: that is when the subtitle
    left the screen, so a subtitle seen on a single frame still lasts one frame interval. The
    frames must be in time order and not deduplicated, or a segment runs on to the next kept frame.
    """
    if not ocr_results:
        return []

    logger.info(f"Merging {len(ocr_results)} frames with similarity threshold {sim_thresh}")

    merged, current, gap_start, gap = [], None, None, 0
    for result in ocr_results:
        text = result.get('text', '').strip() if result.get('success', False) else ''
        time_formatted = result['time_formatted']

        if current and text and _same_subtitle(current['text'], text, sim_thresh, min_shared):
            #mm sous-titre, extend le segment
            current['end_time'] = time_formatted
            gap_start, gap = None, 0
            # garder plus long
            if len(text) > len(current['text']):
                current['text'], current['best_time'] = text, time_formatted
            continue

        if current and not text and gap < max_gap:
            #image vide : peut-être une lecture ratée, on attend la suivante
            gap_start, gap = gap_start or time_formatted, gap + 1
            continue

        #texte different ou plus de texte : le segment actuel finit à la première image sans lui
        if current:
            current['end_time'] = gap_start or time_formatted
            merged.append(current)
        gap_start, gap = None, 0
        current = {
            'time_formatted': time_formatted,
            'text': text,
            'start_time': time_formatted,
            'end_time': time_formatted,
            'best_time': time_formatted,
            'success': True
        } if text else None

    #pas oublier dernier segment
    if current:
        current['end_time'] = gap_start or current['end_time']
        merged.append(current)

    logger.info(f"Merged into {len(merged)} segments")
    return merged
