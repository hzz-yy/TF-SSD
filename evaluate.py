import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from utils.metrics import compute_metrics
from utils.logger import get_logger


SUPPORTED_DATASETS = ["CoSal2015", "CoSOD3k", "CoCA"]


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate_dataset(pred_dir: Path, gt_dir: Path, logger) -> dict[str, float]:
    group_dirs = sorted([d for d in gt_dir.iterdir() if d.is_dir()])
    if not group_dirs:
        logger.error(f"No groups found in: {gt_dir}")
        return {}

    all_metrics = defaultdict(list)
    missing = 0

    for group_dir in tqdm(group_dirs, desc="Evaluating"):
        gt_files = sorted(list(group_dir.glob("*.png")) + list(group_dir.glob("*.jpg")))
        for gt_path in gt_files:
            pred_path = pred_dir / group_dir.name / (gt_path.stem + ".png")
            if not pred_path.exists():
                missing += 1
                continue

            gt   = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
            if gt is None or pred is None:
                continue

            if gt.shape != pred.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)

            for k, v in compute_metrics(pred, gt).items():
                all_metrics[k].append(v)

    if missing > 0:
        logger.warning(f"{missing} prediction(s) not found — skipped.")

    if not all_metrics:
        logger.error("No valid predictions found.")
        return {}

    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


def print_results(results: dict[str, float], dataset_name: str, logger) -> None:
    logger.info(f"\n{'─' * 40}")
    logger.info(f"  Dataset : {dataset_name}")
    logger.info(f"{'─' * 40}")
    logger.info(f"  {'Metric':<10}  {'Value':>8}")
    logger.info(f"  {'─' * 22}")
    for metric, value in sorted(results.items()):
        logger.info(f"  {metric:<10}  {value:>8.4f}")
    logger.info(f"{'─' * 40}\n")


def main():
    parser = argparse.ArgumentParser(description="TF-SSD Evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--pred_dir", default=None,
                        help="Override eval.pred_dir in config.")
    parser.add_argument("--gt_dir", default=None,
                        help="Path to ground-truth directory (dataset/gt/).")
    parser.add_argument("--dataset_name", default=None,
                        choices=SUPPORTED_DATASETS)
    parser.add_argument("--save_results", default=None,
                        help="Path to save results as a YAML file.")
    parser.add_argument("--log_file", default=None)
    args = parser.parse_args()

    logger = get_logger("tfssd-eval", log_file=args.log_file)

    cfg = {}
    if Path(args.config).exists():
        cfg = load_config(args.config)

    dataset_name = args.dataset_name or cfg.get("data", {}).get("dataset_name", "CoSal2015")
    pred_root    = Path(args.pred_dir or cfg.get("eval", {}).get("pred_dir", "./predictions"))
    pred_dir     = pred_root / dataset_name

    if args.gt_dir:
        gt_dir = Path(args.gt_dir)
    else:
        gt_dir = Path(cfg.get("data", {}).get("dataset_root", "./datasets")) / dataset_name / "gt"

    if not pred_dir.exists():
        sys.exit(f"Prediction directory not found: {pred_dir}\nRun inference.py first.")
    if not gt_dir.exists():
        sys.exit(f"Ground-truth directory not found: {gt_dir}")

    logger.info(f"Pred dir : {pred_dir}")
    logger.info(f"GT dir   : {gt_dir}")

    results = evaluate_dataset(pred_dir, gt_dir, logger)

    if results:
        print_results(results, dataset_name, logger)

    if args.save_results and results:
        import yaml
        out = {"dataset": dataset_name, "metrics": results}
        with open(args.save_results, "w") as f:
            yaml.dump(out, f, default_flow_style=False)
        logger.info(f"Results saved to: {args.save_results}")


if __name__ == "__main__":
    main()
