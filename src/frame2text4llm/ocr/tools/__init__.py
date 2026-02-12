
from .paddleocr import PaddleOCR
from .openaiocr import OpenAIOCR
from .mistralocr import MistralOCR
from .easyocr import EasyOCR
from .vlmocr import VLMOCR

OCR_TOOLS = {
    "paddleocr": PaddleOCR,
    "openai": OpenAIOCR,
    "mistral": MistralOCR,
    "easyocr": EasyOCR,
    "vlm": VLMOCR,
}
