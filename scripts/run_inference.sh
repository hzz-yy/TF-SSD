#!/bin/bash
set -e

CONFIG=${1:-configs/default.yaml}
DATASET=${2:-CoSal2015}
SAM_CKPT=${3:-./checkpoints/sam_vit_h_4b8939.pth}
DINO_CKPT=${4:-./checkpoints/dino_vitbase8_pretrain.pth}
OUTPUT_DIR=${5:-./predictions}

echo "=============================="
echo " TF-SSD Inference"
echo "=============================="
echo " Config     : $CONFIG"
echo " Dataset    : $DATASET"
echo " SAM ckpt   : $SAM_CKPT"
echo " DINO ckpt  : $DINO_CKPT"
echo " Output dir : $OUTPUT_DIR"
echo "=============================="

python inference.py \
    --config "$CONFIG" \
    --dataset_name "$DATASET" \
    --sam_checkpoint "$SAM_CKPT" \
    --dino_checkpoint "$DINO_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --log_file "logs/${DATASET}_inference.log"

echo "Inference done."
