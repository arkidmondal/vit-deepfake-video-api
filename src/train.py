import os
import time
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import VideoDataset
from model import ViTVideo
import config


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Freeze Backbone:", config.FREEZE_BACKBONE)
    print("Unfreeze Last N Blocks:", config.UNFREEZE_LAST_N)
    print("Head LR:", config.HEAD_LR)
    print("Backbone LR:", config.BACKBONE_LR)
    print("Epochs:", config.EPOCHS)

    set_seed(42)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    print("\nSaving checkpoints to:", config.CHECKPOINT_DIR)

    # =========================
    # Moderate (Stable) Augmentation
    # =========================
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.03),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    train_ds = VideoDataset(config.TRAIN_CSV, train_transform, config.NUM_FRAMES)
    val_ds   = VideoDataset(config.VAL_CSV, val_transform, config.NUM_FRAMES)

    train_dl = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    # =========================
    # Model
    # =========================
    model = ViTVideo(
        freeze_backbone=config.FREEZE_BACKBONE,
        unfreeze_last_n=config.UNFREEZE_LAST_N
    ).to(device)

    # Resume from previous best model
    if config.RESUME_FROM is not None:
        print("Loading previous checkpoint:", config.RESUME_FROM)
        model.load_state_dict(
            torch.load(config.RESUME_FROM, map_location=device)
        )

    criterion = nn.BCEWithLogitsLoss()

    # =========================
    # Dual LR Optimizer
    # =========================
    head_params = list(model.head.parameters())

    backbone_params = [
        p for name, p in model.named_parameters()
        if "backbone" in name and p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": config.HEAD_LR},
            {"params": backbone_params, "lr": config.BACKBONE_LR},
        ],
        weight_decay=config.WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    counter = 0

    print("\nTraining started at:",
          datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    training_start_time = time.time()

    # =========================
    # Training Loop
    # =========================
    for epoch in range(config.EPOCHS):

        print(f"\nEpoch [{epoch+1}/{config.EPOCHS}]")
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0

        for x, y in train_dl:
            x = x.to(device)
            y = y.unsqueeze(1).float().to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dl)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device)
                y = y.unsqueeze(1).float().to(device)

                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()

        val_loss /= len(val_dl)

        epoch_duration = time.time() - epoch_start_time

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f}")
        print(f"Epoch Duration: {int(epoch_duration//60)}m {int(epoch_duration%60)}s")

        torch.save(
            model.state_dict(),
            os.path.join(
                config.CHECKPOINT_DIR,
                f"checkpoint_epoch_{epoch+1}.pth"
            )
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            )

            print("Best model updated.")
        else:
            counter += 1
            print(f"No improvement. Patience: {counter}/{config.PATIENCE}")

            if counter >= config.PATIENCE:
                print("Early stopping triggered.")
                break

    total_time = time.time() - training_start_time

    print("\nTraining finished at:",
          datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Total Training Time: {int(total_time//60)}m {int(total_time%60)}s")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()