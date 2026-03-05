import torch
import timm
from pathlib import Path

# Model download URL (HuggingFace)
MODEL_URL = "https://huggingface.co/arkid-mondal/vit-deepfake-video-detector/resolve/main/best_model.pth"

# Local path where model will be stored
MODEL_PATH = Path("backend/best_model.pth")


class ViTVideo(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "vit_base_patch8_224",
            pretrained=True,
            num_classes=0
        )

        embed_dim = self.backbone.num_features

        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        features = self.backbone(x)

        features = features.view(B, T, -1)
        features = features.mean(dim=1)

        return self.head(features)


def download_model():
    import requests

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        print("Downloading model weights from HuggingFace...")

        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("Model downloaded.")


def load_model(device="cpu"):
    download_model()

    model = ViTVideo()

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model