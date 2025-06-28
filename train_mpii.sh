#!/bin/bash

# Sample script to run training for the MPII dataset.

# -- Configuration --
DATASET="mpii"
DATA_DIR="datasets/MPIIFaceGaze" # IMPORTANT: Path to the MPII dataset root directory
CHECKPOINT_DIR="./checkpoints_mpii_train"
EPOCHS=50
BATCH_SIZE=32
LR=1e-4
NUM_WORKERS=8 # Adjust based on your machine's capabilities
TRAINING_MODE="both" # Options: 'landmarks', 'gaze', 'both'

# -- Run Training --
echo "======================================================"
echo "Starting training on $DATASET dataset..."
echo "Configuration:"
echo "  - Data Directory: $DATA_DIR"
echo "  - Checkpoint Directory: $CHECKPOINT_DIR"
echo "  - Epochs: $EPOCHS"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Learning Rate: $LR"
echo "  - Dataloader Workers: $NUM_WORKERS"
echo "  - Training Mode: $TRAINING_MODE"
echo "======================================================"

python train.py \
    --dataset $DATASET \
    --data_dir $DATA_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --num_workers $NUM_WORKERS \
    --training_mode $TRAINING_MODE \
    --amp \
    --affine_aug \
    --flip_aug \
    --use_cache \
    --warmup_epochs 5 \
    --freeze_backbone_warmup \

echo "Training finished." 