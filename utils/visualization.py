from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


_PALETTE = [
    (255,  59,  48), ( 52, 199,  89), (  0, 122, 255),
    (255, 204,   0), (175,  82, 222), (255, 149,   0),
    ( 90, 200, 250), (255,  45,  85), ( 88, 187,  88),
    ( 50, 173, 230),
]


def overlay_masks(image: np.ndarray, masks: list[dict], alpha: float = 0.45) -> np.ndarray:
    out = image.copy().astype(np.float32)
    for i, mask_dict in enumerate(masks):
        seg = mask_dict["segmentation"]
        color = np.array(_PALETTE[i % len(_PALETTE)], dtype=np.float32)
        out[seg] = out[seg] * (1 - alpha) + color * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_prediction(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray | None = None,
) -> np.ndarray:
    h, w = image.shape[:2]
    cols = 3 if gt_mask is not None else 2
    canvas = np.zeros((h, w * cols, 3), dtype=np.uint8)

    canvas[:, :w] = image

    overlay = image.copy().astype(np.float32)
    bin_pred = (pred_mask > 127) if pred_mask.max() > 1 else pred_mask.astype(bool)
    overlay[bin_pred] = overlay[bin_pred] * 0.5 + np.array([255, 80, 80]) * 0.5
    canvas[:, w:2*w] = np.clip(overlay, 0, 255).astype(np.uint8)

    if gt_mask is not None:
        gt_color = np.zeros_like(image)
        bin_gt = (gt_mask > 127) if gt_mask.max() > 1 else gt_mask.astype(bool)
        gt_color[bin_gt] = [80, 200, 80]
        canvas[:, 2*w:] = gt_color

    return canvas


def save_group_visualization(
    image_paths: list[str],
    pred_masks: dict[str, np.ndarray],
    output_path: str,
    gt_masks: dict[str, np.ndarray] | None = None,
) -> None:
    n = len(image_paths)
    cols = 3 if gt_masks is not None else 2
    fig, axes = plt.subplots(n, cols, figsize=(cols * 4, n * 3))
    if n == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["Image", "Prediction", "Ground Truth"] if gt_masks else ["Image", "Prediction"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=11, fontweight="bold")

    for i, img_path in enumerate(image_paths):
        name = Path(img_path).stem
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        axes[i, 0].imshow(image)
        axes[i, 0].set_ylabel(name, fontsize=8)
        axes[i, 0].axis("off")

        if name in pred_masks:
            axes[i, 1].imshow(pred_masks[name], cmap="gray")
        axes[i, 1].axis("off")

        if gt_masks is not None:
            if name in gt_masks:
                axes[i, 2].imshow(gt_masks[name], cmap="gray")
            axes[i, 2].axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
