#!/bin/bash
set -e

CONFIG=${1:-configs/default.yaml}
DATASET=${2:-CoSal2015}
PRED_DIR=${3:-./predictions}
GT_DIR=${4:-./datasets/${DATASET}/gt}
SAVE_RESULTS=${5:-./results/${DATASET}_results.yaml}

echo "=============================="
echo " TF-SSD Evaluation"
echo "=============================="
echo " Dataset    : $DATASET"
echo " Pred dir   : $PRED_DIR"
echo " GT dir     : $GT_DIR"
echo "=============================="

mkdir -p results

python evaluate.py \
    --config "$CONFIG" \
    --dataset_name "$DATASET" \
    --pred_dir "$PRED_DIR" \
    --gt_dir "$GT_DIR" \
    --save_results "$SAVE_RESULTS" \
    --log_file "logs/${DATASET}_eval.log"
