import numpy as np
from loguru import logger

class EasyOCR:
    """
    EasyOCR implementation for text extraction from images.
    """
    
    def __init__(self):
        self.languages = ['en', 'fr']  #en et fr pour l'instant
        self.reader = None
    
    def _initialize_reader(self):
        """Initialize EasyOCR reader if not already done."""
        if self.reader is None:
            try:
                import easyocr
                self.reader = easyocr.Reader(self.languages)
                logger.info("EasyOCR reader initialized")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {str(e)}")
                raise RuntimeError(f"Failed to initialize EasyOCR: {str(e)}")
    
    def process_image(self, image: np.ndarray, lang: str = "en") -> str:
        """
        Extract text from image using EasyOCR.
        
        Args:
            image: NumPy array containing the image
            lang: Language code
            
        Returns:
            Extracted text from the image
        """
        try:
            self._initialize_reader()
            
            #map lang codes for compatiibillity
            lang_map = {"eng": "en", "fra": "fr"}
            easyocr_lang = lang_map.get(lang, lang)
            
            if easyocr_lang not in self.languages:
                logger.warning(f"Language '{easyocr_lang}' not supported. Using 'en'.")
                easyocr_lang = "en"
            
            results = self.reader.readtext(image)
            text_results = [text for (bbox, text, confidence) in results if confidence > 0.5]
            
            return " ".join(text_results)
            
        except Exception as e:
            logger.error(f"EasyOCR failed: {str(e)}")
            raise RuntimeError(f"EasyOCR failed: {str(e)}") 