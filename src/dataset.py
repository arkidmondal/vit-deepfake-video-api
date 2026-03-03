import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class VideoDataset(Dataset):
    """
    Video Dataset supporting:
    - MP4 video files
    - Frame folder videos (WildDeepfake)

    Returns:
        video_tensor: [T, 3, H, W]
        label: float tensor
    """

    def __init__(self, csv_file, transform=None, num_frames=16):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data)

    # =========================================================
    # Load frames from MP4 video
    # =========================================================
    def _load_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return None

        frame_indices = np.linspace(
            0, total_frames - 1, self.num_frames, dtype=int
        )

        frames = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)

            if self.transform:
                img = self.transform(img)

            frames.append(img)

        cap.release()

        return frames if len(frames) > 0 else None

    # =========================================================
    # Load frames from folder
    # =========================================================
    def _load_from_folder(self, folder_path):
        frame_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        if len(frame_files) == 0:
            return None

        total_frames = len(frame_files)

        frame_indices = np.linspace(
            0, total_frames - 1, self.num_frames, dtype=int
        )

        frames = []

        for idx in frame_indices:
            frame_path = os.path.join(folder_path, frame_files[idx])
            img = Image.open(frame_path).convert("RGB")

            if self.transform:
                img = self.transform(img)

            frames.append(img)

        return frames if len(frames) > 0 else None

    # =========================================================
    # Get Item
    # =========================================================
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = str(row["path"])
        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        # Auto detect input type
        if os.path.isdir(path):
            frames = self._load_from_folder(path)
        else:
            frames = self._load_from_video(path)

        # Corruption fallback
        if frames is None or len(frames) == 0:
            # Safe dummy tensor
            dummy = torch.zeros(self.num_frames, 3, 224, 224)
            return dummy, label

        # Pad if needed
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        video_tensor = torch.stack(frames)  # [T, 3, H, W]

        return video_tensor, label