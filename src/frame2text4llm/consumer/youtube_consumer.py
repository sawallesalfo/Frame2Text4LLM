import os

from loguru import logger
from yt_dlp import YoutubeDL


class YoutubeConsumer:
    """
    Download videos with yt-dlp: YouTube, Facebook and every site yt-dlp reads.

    Each file is named after its platform and id (youtube_9xIwaKcn2HI.mp4), never after its
    title: two uploads with the same title stay apart, and the name is the key a catalogue
    deduplicates on. All files go to one folder, so a video met in two playlists, or again in
    a later run, is downloaded once. Next to each video:
    - the uploader's own subtitle files, when there are any, so OCR only has to run on burned-in
      subtitles (automatic captions are skipped: on a language YouTube does not know, they are
      another language's speech recognition);
    - its .info.json (title, channel, channel id, URL), so every subtitle window can say which
      video and which channel it comes from.
    """

    def __init__(self, output_dir="datasets/youtube/raw", max_height=720):
        self.output_dir = output_dir
        self.max_height = max_height
        os.makedirs(self.output_dir, exist_ok=True)

    def _options(self):
        h = self.max_height
        return {
            "format": f"bv*[height<={h}]+ba/b[height<={h}]/b",
            "merge_output_format": "mp4",
            "outtmpl": os.path.join(self.output_dir, "%(extractor)s_%(id)s.%(ext)s"),
            "writesubtitles": True,
            "writeautomaticsub": False,
            "subtitleslangs": ["all", "-live_chat"],
            "writeinfojson": True,
            "ignoreerrors": True,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
        }

    @staticmethod
    def _path(ydl, info):
        downloads = info.get("requested_downloads") or [{}]
        return downloads[0].get("filepath") or ydl.prepare_filename(info)

    def consume(self, url):
        """Download a video, a playlist or a channel; return the video's path, or the folder for many."""
        logger.info(f"Consumption of {url} started")
        with YoutubeDL(self._options()) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                logger.error(f"Error while consuming {url}")
                return None
            if "entries" in info:
                done = sum(1 for e in info["entries"] or [] if e)
                logger.info(f"Consumption of {info.get('title')} completed: {done} videos")
                return self.output_dir
            path = self._path(ydl, info)
        logger.info(f"Video file ready: {path}")
        return path

    consume_video = consume_playlist = consume

    def consume_from_urls_list(self, urls_list):
        downloaded_paths = []
        for url in urls_list:
            path = self.consume(url)
            if path:
                downloaded_paths.append(path)
        return downloaded_paths

    def consume_from_file(self, urls_file_path):
        try:
            with open(urls_file_path, "r") as file:
                urls = [url.strip() for url in file.read().splitlines() if url.strip()]
            return self.consume_from_urls_list(urls)
        except Exception as e:
            logger.error(f"Error while reading file {urls_file_path}: {e}")
            return []
