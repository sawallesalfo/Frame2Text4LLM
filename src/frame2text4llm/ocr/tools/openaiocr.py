import os
from typing import Optional
from loguru import logger
import re

class OpenAIOCR:
    """
    OCR implementation using OpenAI's Vision API.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI OCR engine.

        Args:
            api_key: Optional API key. If not provided, tries to use OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY env var not set.")

    def process_image(self, base64_img: str) -> str:
        """
        Extract text from image using OpenAI's Vision model.

        Args:
            base64_img: Base64-encoded image string.

        Returns:
            Extracted text from the image.
        """
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)

            prompt = """Extract all text from this image. Output only the raw text with no additional commentary. Grade 1  when text exist and -1 else.
                Don't add image signature. OCR output shoulb be beetween xml tags <ocr> and </ocr>. 
                Grade shoul be beetween <grade>...</grade>"
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                            }
                        ]
                    }
                ],
                max_tokens=300
            )

            if response.choices and len(response.choices) > 0:
                output =  response.choices[0].message.content.strip()
                match = re.search(r"<ocr>(.*?)</ocr>", output)
                if match:
                    return match.group(1)
            return ""

        except Exception as e:
            raise RuntimeError(f"OpenAI OCR failed: {str(e)}")
