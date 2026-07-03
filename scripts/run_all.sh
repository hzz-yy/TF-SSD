#!/bin/bash
set -e

SAM_CKPT=${1:-./checkpoints/sam_vit_h_4b8939.pth}
DINO_CKPT=${2:-./checkpoints/dino_vitbase8_pretrain.pth}

mkdir -p logs results

DATASETS=("CoSal2015" "CoSOD3k" "CoCA")
CONFIGS=("configs/cosal2015.yaml" "configs/cosod3k.yaml" "configs/coca.yaml")

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    CONFIG="${CONFIGS[$i]}"

    echo ""
    echo "##############################"
    echo "  Running: $DATASET"
    echo "##############################"

    bash scripts/run_inference.sh "$CONFIG" "$DATASET" "$SAM_CKPT" "$DINO_CKPT" "./predictions"
    bash scripts/run_eval.sh "$CONFIG" "$DATASET" "./predictions" "./datasets/${DATASET}/gt" "./results/${DATASET}_results.yaml"
done

echo ""
echo "=============================="
echo " All datasets completed."
echo "=============================="
