import os
from typing import Optional
from loguru import logger
from mistralai import Mistral

class MistralOCR:
    """
    OCR implementation using Mistral's OCR API.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Mistral OCR engine.

        Args:
            api_key: Optional API key. If not provided, tries to use MISTRAL_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key not provided and MISTRAL_API_KEY env var not set.")

    def process_image(self, base64_img: str) -> str:
        """
        Extract text from image using Mistral OCR API.

        Args:
            base64_img: Base64-encoded image string.
            lang: Optional language hint (not directly used).

        Returns:
            Extracted text from the image.
        """
        try:

            client = Mistral(api_key=self.api_key)

            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_img}" 
                }
            )


            transcription = ocr_response.pages[0].markdown
            return transcription

        except Exception as e:
            raise RuntimeError(f"Mistral OCR failed: {str(e)}")
