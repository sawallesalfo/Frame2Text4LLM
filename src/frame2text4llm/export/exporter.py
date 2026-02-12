from abc import ABC, abstractmethod
from typing import List, Dict, Any
from loguru import logger
import json
import os

class BaseExporter(ABC):
    """Base class for exporters."""
    
    @abstractmethod
    def export(self, data: List[Dict[str, Any]], output_path: str) -> None:
        """Export data to specified format."""
        pass

class JSONExporter(BaseExporter):
    """Export OCR results to JSON format."""
    
    def export(self, data: List[Dict[str, Any]], output_path: str) -> None:
        """
        Export OCR results to JSON format.
        
        Args:
            data: List of dictionaries containing OCR results
            output_path: Path to save the JSON file
        """
        try:
            # Format the data as required
            formatted_data = []
            for entry in data:
                if entry.get('success', False):
                    formatted_data.append({
                        'start_time': entry.get('time_formatted', ''),
                        'end_time': '',  # Will be populated from next entry if exists
                        'text': entry.get('text', '').strip()
                    })
            
            # Calculate end times
            for i in range(len(formatted_data) - 1):
                formatted_data[i]['end_time'] = formatted_data[i + 1]['start_time']
            
            # Set last entry's end time if exists
            if formatted_data:
                if not formatted_data[-1]['end_time']:
                    formatted_data[-1]['end_time'] = formatted_data[-1]['start_time']  # Or calculate from video duration
            
            # Export to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"✅ Successfully exported {len(formatted_data)} entries to {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Error exporting to JSON: {str(e)}")
            raise

class SRTExporter(BaseExporter):
    """Export OCR results to SRT format."""
    
    def _format_time(self, time_str: str) -> str:
        """Convert MM:SS:mmm to SRT time format (HH:MM:SS,mmm)."""
        if not time_str:
            return "00:00:00,000"
        
        # Parse MM:SS:mmm
        parts = time_str.split(':')
        if len(parts) != 3:
            return "00:00:00,000"
        
        minutes, seconds, milliseconds = parts
        # Format as HH:MM:SS,mmm
        return f"00:{minutes}:{seconds},{milliseconds}"
    
    def export(self, data: List[Dict[str, Any]], output_path: str) -> None:
        """
        Export OCR results to SRT format.
        
        Args:
            data: List of dictionaries containing OCR results
            output_path: Path to save the SRT file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                subtitle_number = 1
                
                for i, entry in enumerate(data):
                    if not entry.get('success', False):
                        continue
                        
                    text = entry.get('text', '').strip()
                    if not text:
                        continue
                    
                    start_time = self._format_time(entry.get('time_formatted', ''))
                    
                    # Get end time from next entry or use start time
                    end_time = self._format_time(data[i + 1].get('time_formatted', '')) if i < len(data) - 1 else start_time
                    
                    # Write SRT entry
                    f.write(f"{subtitle_number}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
                    
                    subtitle_number += 1
                    
            logger.info(f"✅ Successfully exported {subtitle_number - 1} subtitles to {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Error exporting to SRT: {str(e)}")
            raise
