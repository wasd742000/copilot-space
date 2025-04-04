import os
import glob
import torch
import argparse
import numpy as np
from functools import partial
from multiprocessing import Pool
from torch_geometric.data import Data
import cv2
from torchvision import transforms
from torchvision.models import resnet50
from torch.nn import Module


class FeatureExtractor(Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Load pretrained ResNet50 and remove final FC layer
        self.model = resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        # Add a linear layer to reduce dimension to 1024
        self.fc = torch.nn.Linear(2048, 1024)
        self.model.eval()

    def forward(self, x):
        features = self.model(x).squeeze()
        return self.fc(features)


def extract_video_features(video_path):
    """Extract features from video frames using ResNet50"""
    # Initialize feature extractor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = FeatureExtractor().to(device)

    # Video preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Read video
    cap = cv2.VideoCapture(video_path)
    features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract features
        with torch.no_grad():
            frame_tensor = transform(frame).unsqueeze(0).to(device)
            feature = feature_extractor(frame_tensor)
            features.append(feature.cpu().numpy())

    cap.release()
    return np.array(features)


def get_edge_info(num_frame, args):
    skip = args.skip_factor

    node_source = []
    node_target = []
    edge_attr = []
    for i in range(num_frame):
        for j in range(num_frame):
            frame_diff = i - j

            if abs(frame_diff) <= args.tauf:
                node_source.append(i)
                node_target.append(j)
                edge_attr.append(np.sign(frame_diff))

            elif skip:
                if (frame_diff % skip == 0) and (
                        abs(frame_diff) <= skip * args.tauf):
                    node_source.append(i)
                    node_target.append(j)
                    edge_attr.append(np.sign(frame_diff))

    return node_source, node_target, edge_attr


def generate_video_temporal_graph(video_path, video_id, args, output_path):
    """Generate temporal graph from video file"""
    # Extract features
    features = extract_video_features(video_path)
    num_samples = features.shape[0]

    # Get edge information
    node_source, node_target, edge_attr = get_edge_info(num_samples, args)

    # Create graph data
    graphs = Data(x=torch.tensor(features, dtype=torch.float32),
                  g=video_id,
                  edge_index=torch.tensor([node_source, node_target],
                                          dtype=torch.long),
                  edge_attr=torch.tensor(edge_attr, dtype=torch.float32))

    # Save graph
    graph_path = os.path.join(output_path, f'{video_id}.pt')
    torch.save(graphs, graph_path)
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input video file or directory')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output directory for graph files')
    parser.add_argument('--tauf', type=int, required=True,
                        help='Maximum frame difference between neighboring nodes')
    parser.add_argument('--skip_factor', type=int, default=10,
                        help='Make additional connections between non-adjacent nodes')

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # Handle both single video and directory of videos
    # if os.path.isfile(args.input_path):
    #     video_paths = [args.input_path]
    # else:
    #     video_paths = glob.glob(os.path.join(args.input_path, '*.mp4'))

    # In the main section, update the video file pattern:
    if os.path.isfile(args.input_path):
        video_paths = [args.input_path]
    else:
        # Add MPG pattern
        video_paths = glob.glob(os.path.join(args.input_path, '*.mpg')) + \
                      glob.glob(os.path.join(args.input_path, '*.MPG'))

    for idx, video_path in enumerate(video_paths):
        print(f'Processing video {idx + 1}/{len(video_paths)}: {video_path}')
        generate_video_temporal_graph(video_path, idx + 1, args,
                                      args.output_path)

    print('Graph generation completed!')
