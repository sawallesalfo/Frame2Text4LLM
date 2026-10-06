from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional
from loguru import logger


def _similar(a: str, b: str) -> float:
        """Calculate similarity between two strings."""
        return SequenceMatcher(None, a, b).ratio()
    
def _merge_segments(ocr_results: List[Dict[str, Any]], sim_thresh: float = 0.8) -> List[Dict[str, Any]]:
    """Merge consecutive frames that show the same text into one segment.

    A segment ends at the first frame that shows other text, or none: that is when the subtitle
    left the screen, so a subtitle seen on a single frame still lasts one frame interval. The
    frames must be in time order and not deduplicated, or a segment runs on to the next kept frame.
    """
    if not ocr_results:
        return []

    logger.info(f"Merging {len(ocr_results)} frames with similarity threshold {sim_thresh}")

    merged, current = [], None
    for result in ocr_results:
        text = result.get('text', '').strip() if result.get('success', False) else ''
        time_formatted = result['time_formatted']

        if current and text and _similar(current['text'], text) >= sim_thresh:
            #mm text, extend le segment
            current['end_time'] = time_formatted
            # garder plus long
            if len(text) > len(current['text']):
                current['text'] = text
            continue

        #texte different ou plus de texte : le segment actuel finit ici
        if current:
            current['end_time'] = time_formatted
            merged.append(current)
        current = {
            'time_formatted': time_formatted,
            'text': text,
            'start_time': time_formatted,
            'end_time': time_formatted,
            'success': True
        } if text else None

    #pas oublier dernier segment
    if current:
        merged.append(current)

    logger.info(f"Merged into {len(merged)} segments")
    return merged
