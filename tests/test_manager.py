"""OCRManager keeps one engine per model. Run with pytest, or python tests/test_manager.py."""
import frame2text4llm.ocr.manager as manager_module
from frame2text4llm.framer.video import VideoReader
from frame2text4llm.ocr.manager import OCRManager


class FakeVLM:
    loaded = []

    def __init__(self, model_name):
        self.model_name = model_name
        FakeVLM.loaded.append(model_name)


def test_a_model_is_loaded_once_not_once_per_image():
    manager_module.OCR_TOOLS["vlm"] = FakeVLM
    manager = OCRManager(VideoReader.__new__(VideoReader), region=(0, 10, 0, 10))
    for _ in range(3):
        manager.get_tool_instance("vlm", model_name="Qwen/Qwen3-VL-2B-Instruct")
    manager.get_tool_instance("vlm")
    manager.get_tool_instance("vlm", model_name="Florence-2-base")
    assert FakeVLM.loaded == ["Qwen/Qwen3-VL-2B-Instruct", "microsoft/Florence-2-base"]


if __name__ == "__main__":
    test_a_model_is_loaded_once_not_once_per_image()
    print("ok")
