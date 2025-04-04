import os
import argparse
from main import process_video
from concurrent.futures import ThreadPoolExecutor

def process_videos(dataset_dir, output_dir, config_file, model_path):
    print(f"Starting video processing with the following parameters:")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Configuration file: {config_file}")
    print(f"Model path: {model_path}")

    tasks = []

    with ThreadPoolExecutor() as executor:
        for action in os.listdir(dataset_dir):
            action_path = os.path.join(dataset_dir, action)
            print(f"Checking action directory: {action_path}")
            if not os.path.isdir(action_path):
                print(f"Skipping non-directory: {action_path}")
                continue

            for video in os.listdir(action_path):
                print(f"    Checking video file: {video}")
                if video.endswith((".mpg", ".avi")):
                    video_path = os.path.join(action_path, video)
                    print(f"    Processing video file: {video_path}")

                    # Maintain original structure in output
                    output_subdir = os.path.join(output_dir, action, os.path.splitext(video)[0])
                    os.makedirs(output_subdir, exist_ok=True)
                    print(f"    Output directory created: {output_subdir}")

                    # Schedule process_video function to be called concurrently
                    tasks.append(executor.submit(
                        process_video,
                        input_video=video_path,
                        output_dir=output_subdir,
                        model_path=model_path,
                        cfg_path=config_file,
                        percentile=80,  # Default value, can be parameterized if needed
                        tauf=10,  # Default value, can be parameterized if needed
                        skip_factor=0  # Default value, can be parameterized if needed
                    ))
                else:
                    print(f"    Skipping non-video file: {video}")

        # Wait for all tasks to complete
        for task in tasks:
            try:
                task.result()
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process UCF11 videos to extract keyframes.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory for keyframes")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")

    args = parser.parse_args()
    process_videos(args.dataset_dir, args.output_dir, args.config_file, args.model_path)
