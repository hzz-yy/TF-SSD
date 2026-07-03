import torch
from pathlib import Path


DINO_URLS = {
    "vit_base/8":  "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
    "vit_base/16": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
    "vit_small/8": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
}


def build_dino_model(checkpoint, arch="vit_base", patch_size=8, device=None):
    try:
        import vision_transformer as vits
    except ImportError:
        raise ImportError(
            "vision_transformer.py not found.\n"
            "Download from: https://github.com/facebookresearch/dino/blob/main/vision_transformer.py"
        )

    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"DINO checkpoint not found: {ckpt}\n"
            "Download from: https://github.com/facebookresearch/dino#pretrained-models"
        )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)

    state_dict = torch.load(str(ckpt), map_location="cpu")
    for key in ("teacher", "model", "state_dict"):
        if key in state_dict:
            state_dict = state_dict[key]
            break

    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    return model
