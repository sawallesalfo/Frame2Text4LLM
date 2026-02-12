import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PaddleOCR:
    """
    PaddleOCR implementation.
    """
    
    def __init__(self, cache_dir=None):
        """Initialize PaddleOCR and create cache directory if needed."""
        self._check_paddleocr()
        if cache_dir is None:
            # Default cache directory in user's temp folder
            cache_dir = os.path.join(os.path.expanduser("~"), ".paddleocr_cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _check_paddleocr(self):
        """Check if PaddleOCR is installed."""
        try:
            from paddleocr import PaddleOCR as OCREngine
            logger.info("PaddleOCR is available")
        except ImportError:
            logger.warning("PaddleOCR not installed. Install with: pip install paddleocr")
    
    def process_image(self, image: np.ndarray, lang: str = "en") -> str:
        """
        Extract text from image using PaddleOCR.
        
        Args:
            image: NumPy array containing the image
            lang: Language code (for PaddleOCR, use 'en', 'ch', etc.)
            
        Returns:
            Extracted text from the image
        """
        try:
            from paddleocr import PaddleOCR
            
            # Map common language codes to PaddleOCR language codes
            paddle_lang = lang
            if lang == "eng":
                paddle_lang = "en"
            elif lang == "fra":
                paddle_lang = "fr"
            elif lang == "deu":
                paddle_lang = "german"
            elif lang == "jpn":
                paddle_lang = "japan"
            elif lang == "kor":
                paddle_lang = "korean"
            
            # Use cache directory for models
            ocr = PaddleOCR(
                use_textline_orientation=True,
                lang=paddle_lang,
            )
            result = ocr.predict(image)[0]

            return  " ".join(result["rec_texts"])
        except Exception as e:
            raise RuntimeError(f"PaddleOCR failed: {str(e)}")