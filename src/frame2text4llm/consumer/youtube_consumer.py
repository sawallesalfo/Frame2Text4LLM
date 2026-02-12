import os
import subprocess
from loguru import logger
from pytubefix import Playlist, YouTube

from frame2text4llm.utils.sanitize_filename import sanitize_filename


class YoutubeConsumer:
    """
    Class to consume playlists and videos from YouTube.
    """
    
    def __init__(self, output_dir="datasets/youtube/raw"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def consume_playlist(
            self, 
            playlist_url, 
        ):
        output_dir = self.output_dir
        
        try:
            playlist_name = playlist_url.split('list=')[-1]
            folder_name = f"playlist_{playlist_name}"
            playlist_folder = os.path.join(output_dir, folder_name)
            
            if not os.path.exists(playlist_folder):
                os.makedirs(playlist_folder)
            
            playlist = Playlist(playlist_url)
            logger.info(f"Consumption of playlist: {playlist.title} started")
            
            for video in playlist.videos:
                try:
                    logger.info(f"Processing video: {video.title}")
                    
                    sanitized_title = sanitize_filename(video.title)
                    file_path = os.path.join(playlist_folder, f"{sanitized_title}.mp4")
                    
                    if os.path.exists(file_path):
                        logger.info(f"File already exists: {file_path}. Download skipped.")
                        continue
                    
                    #best video stream
                    video_stream = video.streams.get_highest_resolution()
                    
                    if video_stream is None:
                        #fallback to first available video stream
                        video_stream = video.streams.filter(progressive=True, file_extension='mp4').first()
                    
                    if video_stream is None:
                        logger.warning(f"No suitable video stream found for: {video.title}")
                        continue
                    
                    video_stream.download(output_path=playlist_folder, filename=f"{sanitized_title}.mp4")
                    logger.info(f"Downloaded: {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error while processing the video '{video.title}': {e}")
            
            logger.info("Consumption of playlist completed!")
            return playlist_folder
            
        except Exception as e:
            logger.error(f"Error while consuming the playlist: {e}")
            return None
    
    def consume_video(
            self, 
            video_url,
        ):
        output_dir = self.output_dir
        
        try:
            logger.info(f"Consumption of video: {video_url} started")
            
            yt = YouTube(video_url)
            
            original_title = yt.title
            sanitized_title = sanitize_filename(original_title)
            
            downloaded_file = os.path.join(output_dir, sanitized_title + ".mp4")
            
            if os.path.exists(downloaded_file):
                logger.info(f"File already exists: {downloaded_file}. Download skipped.")
                return downloaded_file
            
            #best video stream
            video_stream = yt.streams.get_highest_resolution()
            
            if video_stream is None:
                #fallback to first available video stream
                video_stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
            
            if video_stream is None:
                logger.warning(f"No suitable video stream found for: {video_url}")
                return None
            
            downloaded_file = video_stream.download(output_path=output_dir, filename=sanitized_title + ".mp4")
            
            logger.info(f"Video file downloaded: {downloaded_file}")
            return downloaded_file
            
        except Exception as e:
            logger.error(f"Error while consuming video: {video_url}: {e}")
            return None
    
    def consume_from_urls_list(self, urls_list):
        downloaded_paths = []
        for url in urls_list:
            if "playlist" in url:
                path = self.consume_playlist(url)
                if path:
                    downloaded_paths.append(path)
            else:
                path = self.consume_video(url)
                if path:
                    downloaded_paths.append(path)
        
        return downloaded_paths
    
    def consume_from_file(self, urls_file_path):
        try:
            with open(urls_file_path, 'r') as file:
                urls = file.read().splitlines()
            urls = [url.strip() for url in urls if url.strip()]
            
            return self.consume_from_urls_list(urls)
            
        except Exception as e:
            logger.error(f"Error while reading file {urls_file_path}: {e}")
            return []