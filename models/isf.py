import cv2
import numpy as np
import torch
import torch.nn.functional as F


class IntraImageSaliencyFilter:

    def __init__(
        self,
        dino_model,
        patch_size: int = 8,
        top_t: int = 3,
        device=None,
        fallback_thresh: float = 0.10,
        fallback_percentile: float = 70.0,
    ):
        self.model = dino_model
        self.patch_size = patch_size
        self.top_t = top_t
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fallback_thresh = fallback_thresh
        self.fallback_percentile = fallback_percentile

    def filter(self, refined_masks, image_tensor):
        attn_map = self._extract_attention_map(image_tensor)

        for mask in refined_masks:
            mask["saliency_score"] = self._saliency_score(mask["segmentation"], attn_map)

        refined_masks.sort(key=lambda m: m["saliency_score"], reverse=True)
        selected = refined_masks[: self.top_t]

        if not selected or selected[0]["saliency_score"] < self.fallback_thresh:
            fallback = self._fallback_from_attention(attn_map)
            if fallback is not None:
                return [fallback]

        return selected

    def _extract_attention_map(self, image_tensor):
        ps = self.patch_size
        _, H, W = image_tensor.shape
        H_crop = H - H % ps
        W_crop = W - W % ps

        img = image_tensor[:, :H_crop, :W_crop].unsqueeze(0).to(self.device)

        with torch.no_grad():
            attn = self.model.get_last_selfattention(img)

        nh = attn.shape[1]
        pH = H_crop // ps
        pW = W_crop // ps

        attn_cls = attn[0, :, 0, 1:].reshape(nh, pH, pW).mean(0)

        attn_up = F.interpolate(
            attn_cls.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        a_min, a_max = attn_up.min(), attn_up.max()
        if a_max > a_min:
            attn_up = (attn_up - a_min) / (a_max - a_min)

        return attn_up.cpu().numpy().astype(np.float32)

    def _saliency_score(self, segmentation, attention_map):
        mask = cv2.resize(
            segmentation.astype(np.uint8),
            (224, 224),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        area = mask.sum()
        if area == 0.0:
            return 0.0
        return float((mask * attention_map).sum() / area)

    def _fallback_from_attention(self, attention_map):
        thresh = float(np.percentile(attention_map, self.fallback_percentile))
        binary = (attention_map >= thresh).astype(np.uint8)
        if binary.sum() == 0:
            return None
        return {
            "segmentation": binary.astype(bool),
            "area": int(binary.sum()),
            "predicted_iou": 0.0,
            "stability_score": 0.0,
            "balanced_score": 0.0,
            "saliency_score": float(attention_map[binary.astype(bool)].mean()),
            "is_fallback": True,
        }
