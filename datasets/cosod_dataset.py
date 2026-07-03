from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


SUPPORTED_DATASETS = ["CoSal2015", "CoSOD3k", "CoCA"]


class CoSODDataset(Dataset):

    def __init__(self, root, dataset_name, split="test", image_transform=None):
        assert dataset_name in SUPPORTED_DATASETS, (
            f"Unknown dataset '{dataset_name}'. Supported: {SUPPORTED_DATASETS}"
        )

        self.root = Path(root) / dataset_name
        self.dataset_name = dataset_name
        self.split = split
        self.image_transform = image_transform

        self.image_root = self.root / "image"
        self.gt_root = self.root / "gt"

        if not self.image_root.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_root}\n"
                f"Please place the dataset under datasets/{dataset_name}/image/"
            )

        self.groups = sorted([d for d in self.image_root.iterdir() if d.is_dir()])
        self.samples = self._build_sample_list()

    def _build_sample_list(self):
        samples = []
        for group_dir in self.groups:
            img_files = sorted(
                list(group_dir.glob("*.jpg")) + list(group_dir.glob("*.png"))
            )
            for img_path in img_files:
                gt_path = self.gt_root / group_dir.name / (img_path.stem + ".png")
                samples.append({
                    "image_path": img_path,
                    "gt_path": gt_path,
                    "group": group_dir.name,
                    "name": img_path.stem,
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        original_size = image.size

        if self.image_transform is not None:
            image = self.image_transform(image)

        gt = None
        if sample["gt_path"].exists():
            gt = np.array(Image.open(sample["gt_path"]).convert("L"))
            gt = (gt > 127).astype(np.uint8)

        return {
            "image": image,
            "gt": gt,
            "image_path": str(sample["image_path"]),
            "group": sample["group"],
            "name": sample["name"],
            "original_size": original_size,
        }

    def get_groups(self):
        group_map = {}
        for sample in self.samples:
            g = sample["group"]
            if g not in group_map:
                group_map[g] = []
            group_map[g].append(sample)
        return group_map

    @property
    def num_groups(self):
        return len(self.groups)
