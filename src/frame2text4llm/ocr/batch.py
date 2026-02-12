from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from loguru import logger 

from .manager import OCRManager
from .utils.text_utils import _merge_segments


class OCRBatchProcessor:
    """
    A class to handle batch processing of OCR tasks with parallelization and progress tracking.
    """
    
    def __init__(
        self, 
        ocr_manager, 
        default_tool: str = "tesseract",
        default_lang: str = "en",
        default_cores: int = 4
    ):
        """
        Initialize the OCR batch processor.
        
        Args:
            ocr_manager: The OCR manager instance that handles actual text extraction
            default_tool: Default OCR engine to use
            default_lang: Default language code
            default_cores: Default number of concurrent threads
        """
        self.ocr_manager = ocr_manager
        self.default_tool = default_tool
        self.default_lang = default_lang
        self.default_cores = default_cores
    
    def _process_single_frame(
        self, 
        frame: Any, 
        tool: str, 
        lang: str, 
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        custom_processor: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process a single frame with OCR.
        
        Args:
            frame: Frame object with image data
            tool: OCR tool to use
            lang: Language code
            api_key: API key for services requiring authentication
            custom_processor: Optional custom processing function
            
        Returns:
            Dictionary with OCR results and metadata
        """
        frame_id = getattr(frame, "id", None)
        timestamp = getattr(frame, "time_formatted", None)
        
        try:
            if custom_processor:
                text = custom_processor(frame.image)
            else:
                text = self.ocr_manager.process(
                    image=frame.image, 
                    tool=tool, 
                    lang=lang, 
                    api_key=api_key,
                    model_name=model_name
                )
                
            result = {
                "id": frame_id,
                "time_formatted": timestamp,
                "text": text,
                "success": True
            }
            logger.debug(f"Successfully processed frame {frame_id} at {timestamp}")
            return result
            
        except Exception as e:
            logger.warning(f"Error processing frame {frame_id} at {timestamp}: {str(e)}")
            return {
                "id": frame_id,
                "time_formatted": timestamp,
                "text": "",
                "success": False,
                "error": str(e)
            }
    
    def process_batch(
        self,
        frames: List[Any],
        tool: Optional[str] = None,
        lang: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        n_cores: Optional[int] = None,
        preserve_order: bool = True,
        show_progress: bool = True,
        custom_processor: Optional[Callable] = None,
        description: str = "🔠 OCR Processing",
        merge_segments: bool = False,
        sim_thresh: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Run OCR on a batch of video frames using parallel processing.
        
        Args:
            frames: List of Frame objects (must contain `.image`)
            tool: OCR tool to use (defaults to instance default)
            lang: Language code (defaults to instance default)
            api_key: API key for tools like OpenAI or Mistral
            model_name: Model name for VLM tools (e.g., "Florence-2-base", "InternVL2-1B")
            n_cores: Number of threads to use (defaults to instance default)
            preserve_order: Whether to preserve original frame order in results
            show_progress: Whether to display a progress bar
            custom_processor: Optional custom function to process images
            description: Description for the progress bar
            merge_segments: Whether to merge similar consecutive segments
            sim_thresh: Similarity threshold for merging (0.0-1.0)
            
        Returns:
            List of dicts with frame metadata and extracted text
        """
        # Use default values if not specified
        tool = tool or self.default_tool
        lang = lang or self.default_lang
        n_cores = n_cores or self.default_cores
        
        logger.info(f"Starting batch OCR processing of {len(frames)} frames using {n_cores} threads")
        
        # Map frame indices for order preservation if needed
        frame_map = {i: frame for i, frame in enumerate(frames)} if preserve_order else {}
        
        # Prepare for parallel execution
        futures = []
        results = []
        
        with ThreadPoolExecutor(max_workers=n_cores) as executor:
            # Submit all tasks
            for idx, frame in enumerate(frames):
                future = executor.submit(
                    self._process_single_frame,
                    frame=frame,
                    tool=tool,
                    lang=lang,
                    api_key=api_key,
                    model_name=model_name,
                    custom_processor=custom_processor
                )
                if preserve_order:
                    futures.append((idx, future))
                else:
                    futures.append(future)
            
            completed_iter = as_completed([f[1] if preserve_order else f for f in futures])
            
            if show_progress:
                completed_iter = tqdm(completed_iter, total=len(frames), desc=description)
                
            for completed in completed_iter:
                if preserve_order:
                    # We'll collect results and sort them later
                    for idx, future in futures:
                        if future == completed:
                            results.append((idx, completed.result()))
                            break
                else:
                    # Just append results as they complete
                    results.append(completed.result())
        
        if preserve_order:
            results.sort(key=lambda x: x[0])
            results = [r[1] for r in results]
        
        logger.info(f"Completed OCR processing. Success rate: {sum(1 for r in results if r.get('success', False))}/{len(results)}")
        
        #if merge segments
        if merge_segments:
            results = _merge_segments(results, sim_thresh)
        
        return results
    
        

def run_ocr_batch_threaded(
    frames: List[Any],
    ocr_manager,
    tool: str = "tesseract",
    lang: str = "fr",
    api_key: Optional[str] = None,
    n_cores: int = 1
) -> List[Dict[str, Any]]:
    """
    Legacy function for backward compatibility.
    Run OCR on a batch of video frames using threads.
    
    Args:
        frames: List of Frame objects (must contain `.image` and `.time_formatted`)
        ocr_manager: An instance of OCRManager
        tool: OCR tool to use ("tesseract", "paddleocr", "openai", etc.)
        lang: Language code
        api_key: API key for tools like OpenAI or Mistral
        n_cores: Number of threads to use
    Returns:
        List of dicts with frame metadata and extracted text
    """
    processor = OCRBatchProcessor(ocr_manager, default_tool=tool, default_lang=lang)
    return processor.process_batch(
        frames=frames,
        tool=tool,
        lang=lang,
        api_key=api_key,
        n_cores=n_cores
    )