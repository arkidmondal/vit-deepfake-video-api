MODEL_NAME = "vit_base_patch8_224"

FREEZE_BACKBONE = True
UNFREEZE_LAST_N = 1   # Best performing configuration

TRAIN_CSV = "splits/train.csv"
VAL_CSV   = "splits/val.csv"

NUM_FRAMES = 16
IMAGE_SIZE = 224

BATCH_SIZE = 4
NUM_WORKERS = 8

# Final fine-tune hyperparameters
HEAD_LR = 7e-5          # Slightly reduced for smoother updates
BACKBONE_LR = 1e-5      # Keep same
WEIGHT_DECAY = 0.07     # Slightly stronger regularization

EPOCHS = 10             # Reduced to avoid unnecessary long runs
PATIENCE = 3

# Resume from best Unfreeze-1 baseline (non-augmented)
RESUME_FROM = "checkpoints/vitp8_imgnet_unf1_6000_hlr1e4_blr1e5/best_model.pth"

# Clean final experiment folder
CHECKPOINT_DIR = "checkpoints/vitp8_imgnet_unf1_6000_hlr1e4_blr1e5"