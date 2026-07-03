import numpy as np
import torch


class QualityMaskGenerator:

    def __init__(
        self,
        sam_model,
        tau_area: float = 0.008,
        tau_con: float = 0.90,
        r_min: float = 0.01,
        r_max: float = 0.50,
        alpha: float = 0.70,
        beta: float = 0.30,
        sigma: float = 0.70,
        gamma: float = 1.50,
        top_tr: int = 10,
        edge_touch_thresh: int = 5,
        edge_coverage_thresh: float = 0.80,
        min_touching_edges: int = 3,
        stability_thresh: float = 0.80,
    ):
        self.sam = sam_model
        self.tau_area = tau_area
        self.tau_con = tau_con
        self.r_min = r_min
        self.r_max = r_max
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.top_tr = top_tr
        self.edge_touch_thresh = edge_touch_thresh
        self.edge_coverage_thresh = edge_coverage_thresh
        self.min_touching_edges = min_touching_edges
        self.stability_thresh = stability_thresh

    def generate(self, image: np.ndarray) -> list[dict]:
        with torch.no_grad():
            raw_masks = self.sam.generate(image)

        H, W = image.shape[:2]
        image_pixels = H * W

        coarse = self._stage1_initial_filter(raw_masks, image_pixels)
        coarse.sort(key=lambda m: m["area"], reverse=True)
        purified = self._stage2_overlap_filter(coarse)
        purified = self._suppress_background(purified, H, W)

        for m in purified:
            m["balanced_score"] = self._balanced_score(m, image_pixels)
        purified.sort(key=lambda m: m["balanced_score"], reverse=True)

        return purified[: self.top_tr]

    def _stage1_initial_filter(self, masks, image_pixels):
        threshold = image_pixels * self.tau_area
        return [
            m for m in masks
            if m["area"] >= threshold and m.get("stability_score", 1.0) >= self.stability_thresh
        ]

    def _stage2_overlap_filter(self, masks):
        purified = []
        for candidate in masks:
            redundant = any(
                self._overlap_ratio(candidate["segmentation"], kept["segmentation"]) >= self.tau_con
                for kept in purified
            )
            if not redundant:
                purified.append(candidate)
        return purified

    def _suppress_background(self, masks, H, W):
        return [m for m in masks if not self._is_background(m["segmentation"], H, W)]

    def _is_background(self, seg, H, W):
        coords = np.where(seg)
        if len(coords[0]) == 0:
            return True
        y1, y2 = coords[0].min(), coords[0].max() + 1
        x1, x2 = coords[1].min(), coords[1].max() + 1
        et = self.edge_touch_thresh
        touches = [
            y1 <= et,
            y2 >= H - et,
            x1 <= et,
            x2 >= W - et,
        ]
        if sum(touches) < self.min_touching_edges:
            return False
        coverages = []
        if touches[0]:
            row = seg[y1, x1:x2]
            coverages.append(row.sum() / max(len(row), 1))
        if touches[1]:
            row = seg[y2 - 1, x1:x2]
            coverages.append(row.sum() / max(len(row), 1))
        if touches[2]:
            col = seg[y1:y2, x1]
            coverages.append(col.sum() / max(len(col), 1))
        if touches[3]:
            col = seg[y1:y2, x2 - 1]
            coverages.append(col.sum() / max(len(col), 1))
        return any(c >= self.edge_coverage_thresh for c in coverages)

    @staticmethod
    def _overlap_ratio(seg_a, seg_b):
        area_b = float(seg_b.sum())
        if area_b == 0.0:
            return 0.0
        return float(np.logical_and(seg_a, seg_b).sum()) / area_b

    def _balanced_score(self, mask, image_pixels):
        ratio = mask["area"] / image_pixels
        return self.alpha * mask["predicted_iou"] + self.beta * self._area_score(ratio)

    def _area_score(self, ratio):
        if self.r_min <= ratio <= self.r_max:
            return 1.0
        if ratio < self.r_min:
            return ratio / self.r_min
        return max(self.sigma, 1.0 - (ratio - self.r_max) * self.gamma)
