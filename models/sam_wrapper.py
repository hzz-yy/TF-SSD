import torch
from pathlib import Path


SAM_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

_SAM_CONFIG = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.80,
    "stability_score_thresh": 0.80,
    "crop_n_layers": 1,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 100,
    "box_nms_thresh": 0.70,
    "crop_overlap_ratio": 0.20,
    "crop_nms_thresh": 0.70,
}


def build_sam_generator(checkpoint, model_type="vit_h", device=None, **kwargs):
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        raise ImportError(
            "segment_anything is required.\n"
            "Install: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )

    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {ckpt}\n"
            f"Download from: {SAM_URLS.get(model_type, '')}"
        )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=str(ckpt))
    sam.to(device).eval()

    config = {**_SAM_CONFIG, **kwargs}
    return SamAutomaticMaskGenerator(model=sam, **config)
