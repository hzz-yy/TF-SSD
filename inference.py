import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from models.pipeline import TFSSDPipeline
from utils.logger import get_logger
from utils.visualization import save_group_visualization


def load_config(config_path: str, overrides: dict) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for k, v in overrides.items():
        keys = k.split(".")
        node = cfg
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = v
    return cfg


def main():
    parser = argparse.ArgumentParser(description="TF-SSD Inference")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--dataset_root", default=None,
                        help="Override data.dataset_root in config.")
    parser.add_argument("--dataset_name", default=None,
                        choices=["CoSal2015", "CoSOD3k", "CoCA"],
                        help="Override data.dataset_name in config.")
    parser.add_argument("--sam_checkpoint", default=None,
                        help="Override sam.checkpoint in config.")
    parser.add_argument("--dino_checkpoint", default=None,
                        help="Override dino.checkpoint in config.")
    parser.add_argument("--output_dir", default=None,
                        help="Override data.output_dir in config.")
    parser.add_argument("--device", default=None,
                        help="Override device in config (e.g. cuda:0, cpu).")
    parser.add_argument("--visualize", action="store_true",
                        help="Save group-level visualization figures.")
    parser.add_argument("--log_file", default=None,
                        help="Path to save log file.")
    args = parser.parse_args()

    overrides = {}
    if args.dataset_root:  overrides["data.dataset_root"]  = args.dataset_root
    if args.dataset_name:  overrides["data.dataset_name"]  = args.dataset_name
    if args.sam_checkpoint: overrides["sam.checkpoint"]    = args.sam_checkpoint
    if args.dino_checkpoint: overrides["dino.checkpoint"]  = args.dino_checkpoint
    if args.output_dir:    overrides["data.output_dir"]    = args.output_dir
    if args.device:        overrides["device"]             = args.device

    if not Path(args.config).exists():
        sys.exit(f"Config not found: {args.config}")

    cfg = load_config(args.config, overrides)
    logger = get_logger("tfssd", log_file=args.log_file)

    dataset_name = cfg["data"]["dataset_name"]
    image_root = Path(cfg["data"]["dataset_root"]) / dataset_name / "image"
    output_dir = Path(cfg["data"]["output_dir"]) / dataset_name

    if not image_root.exists():
        sys.exit(f"Dataset not found: {image_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"TF-SSD Inference — dataset: {dataset_name}")
    logger.info(f"  Image root : {image_root}")
    logger.info(f"  Output dir : {output_dir}")
    logger.info(f"  Device     : {cfg.get('device', 'cuda')}")

    logger.info("Building pipeline...")
    pipeline = TFSSDPipeline(cfg)

    groups = sorted([d for d in image_root.iterdir() if d.is_dir()])
    logger.info(f"Found {len(groups)} groups.")

    total_images = 0
    t0 = time.time()

    for group_dir in tqdm(groups, desc="Groups"):
        image_files = sorted(
            list(group_dir.glob("*.jpg")) + list(group_dir.glob("*.png"))
        )
        if not image_files:
            continue

        predictions = pipeline.run_group(image_files)

        group_out = output_dir / group_dir.name
        group_out.mkdir(parents=True, exist_ok=True)

        for name, mask in predictions.items():
            cv2.imwrite(str(group_out / f"{name}.png"), mask)

        total_images += len(predictions)

        if args.visualize and predictions:
            vis_path = output_dir / "_vis" / f"{group_dir.name}.png"
            save_group_visualization(
                [str(p) for p in image_files],
                predictions,
                str(vis_path),
            )

    elapsed = time.time() - t0
    logger.info(f"Done. {total_images} images processed in {elapsed:.1f}s "
                f"({total_images / max(elapsed, 1e-6):.1f} img/s).")
    logger.info(f"Predictions saved to: {output_dir}")


if __name__ == "__main__":
    main()
