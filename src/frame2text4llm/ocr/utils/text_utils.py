from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional
from loguru import logger


def _similar(a: str, b: str) -> float:
        """Calculate similarity between two strings."""
        return SequenceMatcher(None, a, b).ratio()
    
def _merge_segments(ocr_results: List[Dict[str, Any]], sim_thresh: float = 0.8) -> List[Dict[str, Any]]:
    """Merge similar consecutive text segments."""
    if not ocr_results:
        return []
    
    #keep only successful results
    successful_results = [r for r in ocr_results if r.get('success', False) and r.get('text', '').strip()]
    
    if not successful_results:
        return []
    
    logger.info(f"Merging {len(successful_results)} segments with similarity threshold {sim_thresh}")
    
    merged = []
    current_text = successful_results[0]['text']
    start_time_formatted = successful_results[0]['time_formatted']
    end_time_formatted = successful_results[0]['time_formatted']
    
    for result in successful_results[1:]:
        text = result['text']
        time_formatted = result['time_formatted']
        
        if _similar(current_text, text) >= sim_thresh:
            #mm text, extend le segment
            end_time_formatted = time_formatted
            # garder plus long
            if len(text) > len(current_text):
                current_text = text
        else:
            #texte different, save le segment actuel et demarer un nouveau
            merged.append({
                'time_formatted': start_time_formatted,
                'text': current_text,
                'start_time': start_time_formatted,
                'end_time': end_time_formatted,
                'success': True
            })
            current_text = text
            start_time_formatted = time_formatted
            end_time_formatted = time_formatted
    
    #pas oublier dernier segment
    merged.append({
        'time_formatted': start_time_formatted,
        'text': current_text,
        'start_time': start_time_formatted,
        'end_time': end_time_formatted,
        'success': True
    })
    
    logger.info(f"Merged into {len(merged)} segments")
    return merged