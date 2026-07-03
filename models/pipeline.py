from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from .qmg import QualityMaskGenerator
from .isf import IntraImageSaliencyFilter
from .ips import InterImagePrototypeSelector
from .sam_wrapper import build_sam_generator
from .dino_wrapper import build_dino_model


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class TFSSDPipeline:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device(
            cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        )
        self._transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        self._build_models()

    def _build_models(self):
        sam_cfg = self.cfg["sam"]
        dino_cfg = self.cfg["dino"]
        qmg_cfg = self.cfg["qmg"]
        isf_cfg = self.cfg["isf"]

        sam_gen = build_sam_generator(
            checkpoint=sam_cfg["checkpoint"],
            model_type=sam_cfg.get("model_type", "vit_h"),
            device=self.device,
            points_per_side=sam_cfg.get("points_per_side", 32),
            pred_iou_thresh=sam_cfg.get("pred_iou_thresh", 0.80),
            stability_score_thresh=sam_cfg.get("stability_score_thresh", 0.80),
        )
        dino = build_dino_model(
            checkpoint=dino_cfg["checkpoint"],
            arch=dino_cfg.get("arch", "vit_base"),
            patch_size=dino_cfg.get("patch_size", 8),
            device=self.device,
        )

        self.qmg = QualityMaskGenerator(
            sam_model=sam_gen,
            tau_area=qmg_cfg.get("tau_area", 0.008),
            tau_con=qmg_cfg.get("tau_con", 0.90),
            r_min=qmg_cfg.get("r_min", 0.01),
            r_max=qmg_cfg.get("r_max", 0.50),
            alpha=qmg_cfg.get("alpha", 0.70),
            beta=qmg_cfg.get("beta", 0.30),
            sigma=qmg_cfg.get("sigma", 0.70),
            gamma=qmg_cfg.get("gamma", 1.50),
            top_tr=qmg_cfg.get("top_tr", 10),
            edge_touch_thresh=qmg_cfg.get("edge_touch_thresh", 5),
            edge_coverage_thresh=qmg_cfg.get("edge_coverage_thresh", 0.80),
            min_touching_edges=qmg_cfg.get("min_touching_edges", 3),
            stability_thresh=qmg_cfg.get("stability_thresh", 0.80),
        )
        self.isf = IntraImageSaliencyFilter(
            dino_model=dino,
            patch_size=dino_cfg.get("patch_size", 8),
            top_t=isf_cfg.get("top_t", 3),
            device=self.device,
            fallback_thresh=isf_cfg.get("fallback_thresh", 0.10),
            fallback_percentile=isf_cfg.get("fallback_percentile", 70.0),
        )
        self.ips = InterImagePrototypeSelector(dino_model=dino, device=self.device)

    def run_group(self, image_files: list[Path]) -> dict[str, np.ndarray]:
        salient_masks: dict[str, list[dict]] = {}
        image_paths:   dict[str, str]        = {}
        original_sizes: dict[str, tuple]     = {}

        for img_path in image_files:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            H, W = rgb.shape[:2]

            refined = self.qmg.generate(rgb)
            if not refined:
                continue

            img_tensor = self._transform(Image.fromarray(rgb))
            salient = self.isf.filter(refined, img_tensor)
            if not salient:
                continue

            name = img_path.stem
            salient_masks[name]   = salient
            image_paths[name]     = str(img_path)
            original_sizes[name]  = (H, W)

        if not salient_masks:
            return {}

        results = self.ips.select(salient_masks, image_paths)

        predictions: dict[str, np.ndarray] = {}
        for name, res in results.items():
            mask = res["mask"].astype(np.uint8) * 255
            H, W = original_sizes[name]
            predictions[name] = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)

        return predictions

    @classmethod
    def from_config_file(cls, config_path: str) -> "TFSSDPipeline":
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)
