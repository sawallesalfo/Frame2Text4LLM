
import cv2
import base64
import numpy as np
from typing import List, Any
from loguru import logger
from scipy.fft import dct


def encode_numpy_image(np_image, image_format=".jpg"):
    """Convert a NumPy array image to base64-encoded string."""
    success, buffer = cv2.imencode(image_format, np_image)
    if not success:
        raise ValueError("Image encoding failed")
    base64_str = base64.b64encode(buffer).decode("utf-8")
    return base64_str

def _phash(image: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
        """Compute a perceptual hash (fingerprint) of an image."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        img = cv2.resize(gray, (hash_size * highfreq_factor,)*2)
        d = dct(dct(img.astype(float), axis=0), axis=1)
        low = d[:hash_size, :hash_size]
        return (low > np.median(low)).flatten()
    
def _filter_duplicates(frames: List[Any], diff_thresh: int = 10) -> List[Any]:
    """Filter duplicate frames using perceptual hashing."""
    if not frames:
        return frames
        
    logger.info(f"Filtering duplicates from {len(frames)} frames (threshold={diff_thresh})")
    
    kept, prev = [], None
    for frame in frames:
        h = _phash(frame.image)
        if prev is None or np.count_nonzero(h != prev) > diff_thresh:
            kept.append(frame)
            prev = h
    
    logger.info(f"Kept {len(kept)} frames after duplicate filtering")
    return kept