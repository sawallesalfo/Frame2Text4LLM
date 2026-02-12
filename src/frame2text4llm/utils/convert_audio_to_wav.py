### basic usage: python convert_to_wav.py -i "./youtube-mp3-downloads/kibaye-wakato" -o "./data/raw_data"
# no need for it for now


import argparse
from pydub import AudioSegment
import os
#from utils import sanitize_filename


def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert audio files to WAV format from mp3.")
    parser.add_argument("-i", "--input_dir", type=str, default="./youtube-downloads", help="directory containing the downloaded audio files")
    parser.add_argument("-o", "--output_dir", type=str, default="./data/wavs", help="directory to save the converted WAV files")
    return parser.parse_args()

def convert_to_wav(input_file, output_dir):
    """Convert an audio file from mp3 to WAV format."""
    try:
        print(f"converting {input_file} to WAV...")
        base, ext = os.path.splitext(input_file)
        wav_file = f"{base}.wav"
        
        #move wav file to output_dir
        wav_file_final_path = os.path.join(output_dir, os.path.basename(wav_file))

        if os.path.exists(wav_file_final_path):
            print(f"WAV file already exists: {wav_file_final_path}. Skipping conversion.")
            return wav_file_final_path

        #convert to wav using pydub
        audio = AudioSegment.from_file(input_file)
        audio.export(wav_file, format="wav")
        
        os.rename(wav_file, wav_file_final_path)
        
        print(f"converted and saved WAV file: {wav_file_final_path}")
        return wav_file_final_path
    
    except Exception as e:
        print(f"failed to convert {input_file}: {e}")
        return None


if __name__ == "__main__":
    args = parse_arguments()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    downloaded_files = [f for f in os.listdir(args.input_dir) if f.endswith(".mp3")]
    for file in downloaded_files:
        file_path = os.path.join(args.input_dir, file)
        convert_to_wav(file_path, args.output_dir)

    print("all audio files have been converted to WAV.")