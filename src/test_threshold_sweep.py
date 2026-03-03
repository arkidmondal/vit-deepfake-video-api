import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from dataset import VideoDataset
from model import ViTVideo
import config


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =========================
    # Transform (Same as Validation)
    # =========================
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_ds = VideoDataset(
        "splits/test.csv",
        transform=transform,
        num_frames=config.NUM_FRAMES
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    print("Total test samples:", len(test_ds))

    # =========================
    # Load Model (MATCH TRAINING CONFIG)
    # =========================
    model = ViTVideo(
        freeze_backbone=config.FREEZE_BACKBONE,
        unfreeze_last_n=config.UNFREEZE_LAST_N
    ).to(device)

    checkpoint_path = f"{config.CHECKPOINT_DIR}/best_model.pth"

    print("\nEvaluating checkpoint directory:")
    print(config.CHECKPOINT_DIR)
    print("Loading weight file:", checkpoint_path)

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )

    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    all_probs = []
    all_labels = []
    total_loss = 0.0

    # =========================
    # Evaluation Loop
    # =========================
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device)
            y = y.unsqueeze(1).float().to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy().flatten())

    test_loss = total_loss / len(test_dl)

    # =========================
    # Overall Model Quality
    # =========================
    roc_auc = roc_auc_score(all_labels, all_probs)

    print("\n===== Overall Model Quality =====")
    print(f"Test Loss : {test_loss:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    # =========================
    # Threshold Sweep
    # =========================
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    print("\n===== Threshold Sweep =====")

    best_f1 = 0.0
    best_threshold = 0.5

    for threshold in thresholds:

        preds_binary = [1 if p > threshold else 0 for p in all_probs]

        accuracy = accuracy_score(all_labels, preds_binary)
        precision = precision_score(all_labels, preds_binary, zero_division=0)
        recall = recall_score(all_labels, preds_binary, zero_division=0)
        f1 = f1_score(all_labels, preds_binary, zero_division=0)

        print(f"\nThreshold: {threshold:.2f}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print("-" * 35)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print("\n===== Best Threshold Based on F1 =====")
    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Best F1 Score : {best_f1:.4f}")


if __name__ == "__main__":
    main()