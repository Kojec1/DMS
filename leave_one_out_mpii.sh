#!/usr/bin/env bash

# leave_one_out_training.sh
# ---------------------------------
# Run leave-one-out cross-validation training for the MPIIFaceGaze dataset.
# For each participant in the list, trains a model using the remaining participants
# for training and the held-out participant for validation. Each run is saved to
# its own sub-directory under a common checkpoint directory. After all runs finish
# the aggregated analysis script misc/one_out_analysis.py is executed.

set -euo pipefail

# List of participant IDs present in the dataset. Modify if necessary.
PARTICIPANTS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14)

# Base directory to store all leave-one-out runs. Can be overridden via the
# --checkpoint_root flag when invoking this script.
CHECKPOINT_ROOT="./checkpoints_leave_one_out"

# Helper: prints usage and exits
usage() {
  echo "Usage: $0 --data_dir PATH [--checkpoint_root PATH] [OTHER_SHARED_ARGS]" >&2
  echo "  --data_dir         Path to the root MPIIFaceGaze directory (mandatory)" >&2
  echo "  --checkpoint_root  Root directory where each run's checkpoints will be saved (optional, default: $CHECKPOINT_ROOT)" >&2
  echo "  Any additional arguments are passed verbatim to train.py for every run." >&2
  exit 1
}

# Parse arguments
if [[ $# -lt 2 ]]; then
  usage
fi

DATA_DIR=""
COMMON_ARGS=(
    --dataset mpii
    --batch_size 312
    --epochs 50
    --lr 1e-5
    --weight_decay 1e-3
    --log_interval 50
    --amp
    --affine_aug
    --flip_aug
    --use_clahe
    --use_cache
    --warmup_epochs 5
    --freeze_backbone_warmup
    --training_mode gaze
    --num_angle_bins 32
    --angle_bin_width 3.0
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --checkpoint_root)
      CHECKPOINT_ROOT="$2"
      shift 2
      ;;
    *)
      COMMON_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$DATA_DIR" ]]; then
  echo "Error: --data_dir is mandatory." >&2
  usage
fi

# Create the base checkpoint directory if it does not exist.
mkdir -p "$CHECKPOINT_ROOT"

# Main loop
for VAL_ID in "${PARTICIPANTS[@]}"; do
  # Compose comma-separated train ID string (all IDs except VAL_ID)
  TRAIN_IDS=()
  for PID in "${PARTICIPANTS[@]}"; do
    if [[ "$PID" != "$VAL_ID" ]]; then
      TRAIN_IDS+=("$PID")
    fi
  done
  IFS="," read -r -a _ <<< "${TRAIN_IDS[*]}"   # Ensures proper separation
  TRAIN_IDS_STR=$(IFS=,; echo "${TRAIN_IDS[*]}")

  # Create dedicated directory for this run
  RUN_DIR="$CHECKPOINT_ROOT/run_val_${VAL_ID}"
  mkdir -p "$RUN_DIR"

  echo "\n=============================================================="
  echo "Starting leave-one-out run: validation participant = $VAL_ID"
  echo "Training IDs          = $TRAIN_IDS_STR"
  echo "Checkpoints directory = $RUN_DIR"
  echo "==============================================================\n"

  python train.py \
    --dataset mpii \
    --data_dir "$DATA_DIR" \
    --train_participant_ids "$TRAIN_IDS_STR" \
    --val_participant_ids "$VAL_ID" \
    --checkpoint_dir "$RUN_DIR" \
    "${COMMON_ARGS[@]}"

done

# Run the aggregated analysis once all individual trainings are complete.
python misc/one_out_analysis.py --dir "$CHECKPOINT_ROOT" --output "$CHECKPOINT_ROOT/analysis.png"

echo "\nLeave-one-out training completed. Results aggregated in $CHECKPOINT_ROOT"