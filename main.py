import os
import cv2
import numpy as np
import argparse
from single_graph_generate import generate_video_temporal_graph
from inference import predict_single, get_cfg
import shutil
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

def extract_keyframes_adaptive(video_path, predictions, output_dir, percentile=90):
    """Extract frames using adaptive thresholding"""
    os.makedirs(output_dir, exist_ok=True)

    if len(predictions) == 0 or len(predictions[0]) < 2:
        raise ValueError(f"Predictions format is unexpected: {predictions}")

    scores = predictions[0][1]

    print(f"Predictions length: {len(predictions)}, Scores length: {len(scores)}")

    # Calculate adaptive threshold
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    adaptive_threshold = mean_score + std_score

    # Also calculate percentile threshold as backup
    percentile_threshold = np.percentile(scores, percentile)

    # Use the lower of the two thresholds to ensure we get enough frames
    final_threshold = min(adaptive_threshold, percentile_threshold)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    key_frames_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count >= len(scores):
            raise IndexError(f"Frame count {frame_count} out of range for scores length {len(scores)}")

        if scores[frame_count] > final_threshold:
            output_path = os.path.join(output_dir, f'keyframe_{frame_count:04d}_score_{scores[frame_count]:.3f}.jpg')
            cv2.imwrite(output_path, frame)
            key_frames_count += 1

        frame_count += 1

    cap.release()
    return key_frames_count, frame_count

def process_video(input_video, output_dir, model_path, cfg_path, percentile=80, tauf=10, skip_factor=0):
    # Check if keyframes already exist
    if os.path.exists(output_dir) and any(fname.endswith('.jpg') for fname in os.listdir(output_dir)):
        print(f"Keyframes already exist in {output_dir}. Skipping processing for {input_video}.")
        return

    # Enable GPU memory growth to avoid allocating all memory at once
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            device = '/GPU:0'
        except RuntimeError as e:
            print(e)
            device = '/CPU:0'
    else:
        device = '/CPU:0'

    # Create temporary directory for graph
    temp_graph_dir = 'temp_graph'
    os.makedirs(temp_graph_dir, exist_ok=True)

    try:
        # Step 1: Generate graph from video
        print("Generating graph from video...")
        generate_video_temporal_graph(input_video, 1, argparse.Namespace(tauf=tauf, skip_factor=skip_factor), temp_graph_dir)

        # Step 2: Run inference
        print("Running inference...")
        cfg = get_cfg(argparse.Namespace(cfg=cfg_path))
        
        # Ensure model and tensors are placed on GPU if available
        with tf.device(device):
            predictions = predict_single(cfg, temp_graph_dir, model_path)

        # Step 3: Extract keyframes
        print("Extracting keyframes...")
        num_keyframes, total_frames = extract_keyframes_adaptive(input_video, predictions, output_dir, percentile)
        print(f"Extracted {num_keyframes} keyframes (of {total_frames} total frames) to {output_dir}")

    finally:
        # Cleanup temporary directory
        if os.path.exists(temp_graph_dir):
            shutil.rmtree(temp_graph_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, required=True, help='Path to input video (e.g., .mpg, .avi)')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory for keyframes')
    parser.add_argument('--model_path', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--cfg', type=str, required=True, help='Path to model config file')
    parser.add_argument('--percentile', type=float, default=80, help='Percentile threshold for keyframe selection')
    parser.add_argument('--tauf', type=int, default=10, help='Maximum frame difference between neighboring nodes')
    parser.add_argument('--skip_factor', type=int, default=0, help='Make additional connections between non-adjacent nodes')
    args = parser.parse_args()

    process_video(args.input_video, args.output_dir, args.model_path, args.cfg, args.percentile, args.tauf, args.skip_factor)

if __name__ == "__main__":
    main()