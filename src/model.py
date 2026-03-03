import torch
import torch.nn as nn
import timm


class ViTVideo(nn.Module):
    """
    ViT Patch8 Video Classifier
    Supports:
    - Full freeze
    - Unfreeze last N transformer blocks
    """

    def __init__(self, freeze_backbone=True, unfreeze_last_n=0):
        super(ViTVideo, self).__init__()

        self.backbone = timm.create_model(
            "vit_base_patch8_224",
            pretrained=True,
            num_classes=0
        )

        self.embed_dim = self.backbone.num_features  # 768

        # Freeze entire backbone first
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Unfreeze last N blocks
        if unfreeze_last_n > 0:
            for block in self.backbone.blocks[-unfreeze_last_n:]:
                for param in block.parameters():
                    param.requires_grad = True

            # Always unfreeze final norm when fine-tuning
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.contiguous().view(B * T, C, H, W)  # Safe memory handling
        features = self.backbone(x)

        features = features.view(B, T, -1)
        features = features.mean(dim=1)

        return self.head(features)