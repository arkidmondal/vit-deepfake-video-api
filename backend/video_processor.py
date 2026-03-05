import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Same settings used during training
NUM_FRAMES = 16
IMAGE_SIZE = 224

# Validation / inference transform (must match training)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def extract_frames(video_path, num_frames=NUM_FRAMES):
    """
    Extract uniformly sampled frames from video
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("Could not read video.")

    frame_indices = np.linspace(
        0, total_frames - 1, num_frames, dtype=int
    )

    frames = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        img = transform(img)

        frames.append(img)

    cap.release()

    # Pad if fewer frames
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return torch.stack(frames)  # [T,3,H,W]


def process_video(video_path):
    """
    Convert video to model input tensor
    """
    frames_tensor = extract_frames(video_path)

    # Add batch dimension
    frames_tensor = frames_tensor.unsqueeze(0)

    return frames_tensor  # shape: [1,16,3,224,224]