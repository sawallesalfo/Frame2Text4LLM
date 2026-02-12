import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="Count the number of audio files in the specified output directory.")
    parser.add_argument("output_dir", type=str, help="the directory to count audio files in")
    return parser.parse_args()

def count_audio_files(output_dir):
    audio_extensions = {'.mp3', '.wav'} 
    count = 0
    
    #iterate through files in the specified directory
    for file in os.listdir(output_dir):
        if os.path.isfile(os.path.join(output_dir, file)):
            _, ext = os.path.splitext(file)
            if ext.lower() in audio_extensions:
                count += 1
    
    return count


if __name__ == "__main__":
    args = parse_arguments()
    num_files = count_audio_files(args.output_dir)
    print(f"Number of audio files in '{args.output_dir}': {num_files}")