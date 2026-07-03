from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T


class InterImagePrototypeSelector:

    def __init__(self, dino_model, device=None):
        self.model = dino_model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def select(self, salient_masks, image_paths):
        all_protos = []
        all_meta = []

        for img_name, masks in salient_masks.items():
            for mask_dict in masks:
                proto = self._extract_prototype(image_paths[img_name], mask_dict["segmentation"])
                all_protos.append(proto)
                all_meta.append({
                    "image_name": img_name,
                    "mask_dict": mask_dict,
                    "proto_idx": len(all_protos) - 1,
                })

        C = self._similarity_matrix(all_protos)
        image_names = list(salient_masks.keys())
        results = {}

        for img_name in image_names:
            own_idx = [m["proto_idx"] for m in all_meta if m["image_name"] == img_name]
            best_score, best_pid = self._co_salient_score(img_name, own_idx, C, all_meta, image_names)
            for meta in all_meta:
                if meta["proto_idx"] == best_pid:
                    results[img_name] = {
                        "mask": meta["mask_dict"]["segmentation"],
                        "score": best_score,
                        "prototype": all_protos[best_pid],
                    }
                    break

        return results

    def _extract_prototype(self, image_path, mask):
        img_np = np.array(Image.open(image_path).convert("RGB"))
        mask_3d = np.stack([mask] * 3, axis=-1) if mask.ndim == 2 else mask
        masked = (img_np * mask_3d).astype(np.uint8)

        tensor = self._transform(Image.fromarray(masked)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)
            cls = out[0] if isinstance(out, (tuple, list)) else out
            cls = F.normalize(cls, p=2, dim=-1)

        return cls.squeeze(0)

    @staticmethod
    def _similarity_matrix(prototypes):
        if not prototypes:
            return torch.empty(0, 0)
        P = torch.stack(prototypes)
        with torch.no_grad():
            return torch.mm(P, P.t())

    def _co_salient_score(self, current_name, own_indices, C, all_meta, image_names):
        dev = C.device
        own_t = torch.tensor(own_indices, device=dev)
        scores = torch.zeros(len(own_indices), device=dev)

        for other_name in image_names:
            if other_name == current_name:
                continue
            other_t = torch.tensor(
                [m["proto_idx"] for m in all_meta if m["image_name"] == other_name],
                device=dev,
            )
            if other_t.numel() == 0:
                continue
            scores += C[own_t][:, other_t].max(dim=1).values

        best_local = int(scores.argmax())
        return float(scores[best_local]), own_indices[best_local]
