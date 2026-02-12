from typing import List, Dict, Any, Optional
from pathlib import Path
import os
import json
import shutil
from loguru import logger
from datasets import Dataset, Audio, Features, Value, load_dataset
from huggingface_hub import HfApi, upload_file
from .audio_generator import AudioGenerator

class HuggingfaceCooker:
    """A class to prepare and upload datasets to Huggingface Hub."""
    
    def __init__(
        self,
        username: str,
        token: Optional[str] = None,
        repository_id: Optional[str] = None
    ):
        """
        Initialize the Huggingface Cooker.
        
        Args:
            username: Huggingface username
            token: Huggingface API token (optional if not pushing to hub)
            repository_id: Repository ID to push to (optional)
        """
        self.username = username
        self.token = token
        self.repository_id = repository_id or f"{username}/audio-ocr-dataset"
        self.api = HfApi()
    
    def cook_dataset(
        self,
        ocr_results: List[Dict[str, Any]],
        video_path: str,
        output_dir: str,
        push_to_hub: bool = False
    ) -> Dataset:
        """
        Cook a Huggingface dataset from OCR results and video file.
        
        Args:
            ocr_results: List of OCR results with timestamps and text
            video_path: Path to the video file
            output_dir: Directory to save the prepared dataset
            push_to_hub: Whether to push the dataset to Huggingface Hub
            
        Returns:
            The created Huggingface Dataset
        """
        try:
            # Prepare output directory
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate audio segments
            audio_gen = AudioGenerator(video_path, tmp_audio_dir=output_dir/"tmp_audio")
            # segments must have start_time and end_time
            audio_paths = audio_gen.extract_segments(ocr_results)
            
            # Format the data for the dataset
            dataset_items = []
            
            for entry, audio_path in zip(ocr_results, audio_paths):
                if not entry.get('success', False):
                    continue
                
                text = entry.get('text', '').strip()
                if not text or not audio_path:
                    continue
                
                start_time = entry.get('start_time', entry.get('time_formatted', ''))
                
                dataset_items.append({
                    'audio': str(audio_path),
                    'text': text,
                    'start_time': start_time
                })
            
            # Create the dataset
            features = Features({
                'audio': Audio(),
                'text': Value('string'),
                'start_time': Value('string')
            })
            
            dataset = Dataset.from_list(dataset_items, features=features)
            
            # Save dataset to disk
            dataset.save_to_disk(str(output_dir))
            
            logger.info(f"✅ Successfully created dataset with {len(dataset)} entries")
            
            if push_to_hub and self.token:
                self._push_to_hub(str(output_dir))
            
            return dataset
            
        except Exception as e:
            logger.error(f"❌ Error cooking dataset: {str(e)}")
            raise
    
    def _push_to_hub(self, dataset_path: str) -> None:
        """Push the dataset to Huggingface Hub."""
        try:
            if not self.token:
                raise ValueError("Token required to push to hub")
            
            # Create the repository if it doesn't exist
            self.api.create_repo(
                repo_id=self.repository_id,
                token=self.token,
                repo_type="dataset",
                exist_ok=True
            )
            
            # Upload the dataset
            self.api.upload_folder(
                folder_path=dataset_path,
                repo_id=self.repository_id,
                repo_type="dataset",
                token=self.token
            )
            
            logger.info(f"✅ Successfully pushed dataset to {self.repository_id}")
            
        except Exception as e:
            logger.error(f"❌ Error pushing to hub: {str(e)}")
            raise
    
    def _validate_audio_segments(self, segments: List[Dict[str, Any]]) -> bool:
        """Validate audio segments data structure."""
        required_fields = {'start_time', 'end_time', 'text'}
        
        for segment in segments:
            if not all(field in segment for field in required_fields):
                return False
        
        return True
