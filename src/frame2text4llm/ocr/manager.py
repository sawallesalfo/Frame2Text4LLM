"""
OCR Manager implementation for Frame2Tex4LLM.
"""

import os
import numpy as np
from typing import Dict, Any, Optional, List
from frame2text4llm.framer.subtitle import SubtitleRegionDetector
from frame2text4llm.framer.video import VideoReader

from .tools import OCR_TOOLS
from .utils.image_utils import encode_numpy_image


class OCRManager:
    """
    Manages different OCR tools for text extraction from images.
    Supports multiple OCR engines: Tesseract, PaddleOCR, OpenAI, and Mistral.
    """

    def __init__(self, video_reader: VideoReader):
        """Initialize OCR Manager with available tools."""
        self.video_reader = video_reader
        self.available_tools = list(OCR_TOOLS.keys())
        self.tool_instances: Dict[str, Any] = {}

    def get_tool_instance(self, tool_name: str, api_key: Optional[str] = None, model_name: Optional[str] = None) -> Any:
        """
        Get or create an instance of the requested OCR tool.

        Args:
            tool_name: Name of the OCR tool.
            api_key: API key for cloud-based OCR services.
            model_name: Model name for VLM tools (e.g., "Florence-2-base", "InternVL2-1B").

        Returns:
            Instance of the OCR tool.
        """
        if tool_name not in self.available_tools:
            raise ValueError(f"Unsupported OCR tool: {tool_name}. Available tools: {self.available_tools}")

        # Create a new instance if it doesn't exist or if an API key is explicitly provided
        if tool_name not in self.tool_instances or api_key or model_name:
            tool_class = OCR_TOOLS[tool_name]

            if tool_name in ["openai", "mistral"]:
                if not api_key:
                    raise ValueError(f"API key is required for {tool_name} OCR")
                self.tool_instances[tool_name] = tool_class(api_key=api_key)
            elif tool_name == "vlm":
                # Map short names to full model names
                model_mapping = {
                    "Florence-2-base": "microsoft/Florence-2-base",
                    "Florence-2-base-ft": "microsoft/Florence-2-base-ft", 
                    "InternVL2-1B": "OpenGVLab/InternVL2-1B"
                }
                full_model_name = model_mapping.get(model_name, model_name) if model_name else "microsoft/Florence-2-base-ft"
                self.tool_instances[tool_name] = tool_class(model_name=full_model_name)
            else:
                self.tool_instances[tool_name] = tool_class()

        return self.tool_instances[tool_name]

    def process(
        self,
        image: np.ndarray,
        tool: str = "tesseract",
        lang: str = "eng",
        api_key: Optional[str] = None,
        detect_subtitle_reggion: bool = True,
        model_name: Optional[str] = None,
    ) -> str:
        """
        Process an image using the selected OCR tool.

        Args:
            image: NumPy array containing the image.
            tool: OCR tool to use ('tesseract', 'paddleocr', 'openai', 'mistral', 'vlm').
            lang: Language code (for tools that support it).
            api_key: API key for cloud-based OCR services.
            detect_subtitle_reggion:  Whether to detect subtitle regions in the image and crop it before ocr.
            model_name: Model name for VLM tools (e.g., "Florence-2-base", "InternVL2-1B").

        Returns:
            Extracted text from the image.
        """
        tool_instance = self.get_tool_instance(tool, api_key, model_name)
        if detect_subtitle_reggion:
            detector = SubtitleRegionDetector(self.video_reader)
            y1, y2, x1, x2  = detector.detect_region()
            image = image[y1:y2, x1:x2]

        if tool in ["openai", "mistral"]:
            base64_img = encode_numpy_image(image)
            return tool_instance.process_image(base64_img)
        else:
            return tool_instance.process_image(image, lang)

    def batch_process(
        self,
        images: List[np.ndarray],
        tool: str = "tesseract",
        lang: str = "",
        api_key: Optional[str] = None,
    ) -> List[str]:
        """
        Process multiple images using the selected OCR tool.

        Args:
            images: List of NumPy arrays containing images.
            tool: OCR tool to use.
            lang: Language code.
            api_key: API key for cloud-based OCR services.

        Returns:
            List of extracted texts.
        """
        return [self.process(img, tool, lang, api_key) for img in images]
