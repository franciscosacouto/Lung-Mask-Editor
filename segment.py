"""Optional SAM-family click/box segmentation, one model per layer:

- "lung"   -> MobileSAM (models/mobile_sam.pt): natively point-prompted, so one click
              segments a large, clearly-bounded structure like the lung.
- "nodule" -> LiteMedSAM (models/lite_medsam.pth): MedSAM distilled for CPU (CVPR24
              "MedSAM on Laptop"), trained on BOX prompts only — a click is turned into
              a small box around it.

LiteMedSAM is NOT a drop-in SAM checkpoint — its TinyViT and 256px pipeline differ from
stock SAM/MobileSAM — so we build *its* architecture from the MedSAM repo's own
tiny_vit_sam.py and reproduce its preprocessing, rather than loading the weights into
someone else's model. MobileSAM is standard SAM (vit_t), used through its own repo's
SamPredictor.

Setup
-----
MobileSAM (lung, point prompt):
    git clone --depth 1 https://github.com/ChaoningZhang/MobileSAM.git
    checkpoint -> tools/models/mobile_sam.pt   (or --mobilesam-checkpoint)

LiteMedSAM (nodule, box prompt):
    git clone --depth 1 -b LiteMedSAM https://github.com/bowang-lab/MedSAM.git
    checkpoint -> tools/models/lite_medsam.pth   (or --sam-checkpoint)

Deps (both): pip install torch torchvision timm
Everything is lazy — nothing heavy imports until the first prompt for that layer.
"""
import os
import sys
from pathlib import Path

import numpy as np

IMG_SIZE = 256  # LiteMedSAM's fixed input size
CFG = {"medsam_checkpoint": None, "medsam_repo": None, "mobilesam_checkpoint": None, "mobilesam_repo": None}
_MEDSAM = {"model": None, "embed": None, "last_key": None, "meta": None}
_MOBILESAM = {"predictor": None, "last_key": None}

LAYER_BACKEND = {"lung": "mobilesam", "nodule": "litemedsam"}


# ------------------------------- configuration ------------------------------
def configure(checkpoint=None, repo=None, mobilesam_checkpoint=None, mobilesam_repo=None, **_ignored):
    """Set the checkpoint/repo paths (from the CLI). Resets any loaded models."""
    if checkpoint:
        CFG["medsam_checkpoint"] = Path(checkpoint)
    if repo:
        CFG["medsam_repo"] = Path(repo)
    if mobilesam_checkpoint:
        CFG["mobilesam_checkpoint"] = Path(mobilesam_checkpoint)
    if mobilesam_repo:
        CFG["mobilesam_repo"] = Path(mobilesam_repo)
    _MEDSAM.update(model=None, embed=None, last_key=None, meta=None)
    _MOBILESAM.update(predictor=None, last_key=None)


def backend(layer):
    return LAYER_BACKEND.get(layer, "litemedsam")


def prompt_mode(layer):
    return "point" if backend(layer) == "mobilesam" else "box"


def available(layer):
    return _mobilesam_available() if backend(layer) == "mobilesam" else _medsam_available()


def segment(image01, layer, point=None, box=None, key=None):
    """image01: HxW float in [0,1]. `key` identifies the image so its encoder only runs
    once per image; later prompts on the same image (for that layer's model) reuse the
    cached embedding."""
    if backend(layer) == "mobilesam":
        return _mobilesam_segment(image01, point=point, box=box, key=key)
    return _medsam_segment(image01, point=point, box=box, key=key)


# ---------------------------------------------------------------------------
# LiteMedSAM — box prompt, nodule layer
# ---------------------------------------------------------------------------
def _medsam_checkpoint_path():
    if CFG["medsam_checkpoint"]:
        return Path(CFG["medsam_checkpoint"])
    env = os.environ.get("SAM_CHECKPOINT")
    if env:
        return Path(env)
    models = Path(__file__).with_name("models")
    for cand in (models / "lite_medsam.pth", models / "lite_medsam.pt"):
        if cand.exists():
            return cand
    return models / "lite_medsam.pth"


def _medsam_repo_path():
    """Folder holding the MedSAM LiteMedSAM code (must contain tiny_vit_sam.py)."""
    for cand in (CFG["medsam_repo"], os.environ.get("LITEMEDSAM_REPO")):
        if cand:
            return Path(cand)
    here = Path(__file__).parent
    for name in ("MedSAM", "MedSAM-LiteMedSAM", "LiteMedSAM"):
        p = here / name
        if (p / "tiny_vit_sam.py").exists():
            return p
    return here / "MedSAM"


def _medsam_available():
    """True if the code and checkpoint are present (cheap — no torch import)."""
    return (_medsam_repo_path() / "tiny_vit_sam.py").exists() and _medsam_checkpoint_path().exists()


def _build_medsam():
    if _MEDSAM["model"] is not None:
        return _MEDSAM["model"]
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    repo, ckpt = _medsam_repo_path(), _medsam_checkpoint_path()
    if not (repo / "tiny_vit_sam.py").exists():
        raise FileNotFoundError(f"LiteMedSAM code not found at {repo} (need tiny_vit_sam.py) — "
                                f"clone the MedSAM LiteMedSAM branch or pass --litemedsam-repo")
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found at {ckpt}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from tiny_vit_sam import TinyViT
    try:  # the SAM decoder pieces — either package provides them
        from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
    except ImportError:
        from mobile_sam.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer

    class MedSAM_Lite(nn.Module):
        """LiteMedSAM's wrapper (image encoder + prompt encoder + mask decoder)."""

        def __init__(self, image_encoder, mask_decoder, prompt_encoder):
            super().__init__()
            self.image_encoder = image_encoder
            self.mask_decoder = mask_decoder
            self.prompt_encoder = prompt_encoder

        def postprocess_masks(self, masks, new_size, original_size):
            masks = F.interpolate(masks, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
            masks = masks[..., :new_size[0], :new_size[1]]  # drop the padding
            return F.interpolate(masks, size=original_size, mode="bilinear", align_corners=False)

    encoder = TinyViT(
        img_size=IMG_SIZE, in_chans=3, embed_dims=[64, 128, 160, 320], depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10], window_sizes=[7, 7, 14, 7], mlp_ratio=4.0,
        drop_rate=0.0, drop_path_rate=0.0, use_checkpoint=False,
        mbconv_expand_ratio=4.0, local_conv_size=3, layer_lr_decay=0.8,
    )
    prompt_encoder = PromptEncoder(
        embed_dim=256, image_embedding_size=(64, 64),
        input_image_size=(IMG_SIZE, IMG_SIZE), mask_in_chans=16,
    )
    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
        transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256,
    )
    model = MedSAM_Lite(image_encoder=encoder, mask_decoder=mask_decoder, prompt_encoder=prompt_encoder)

    sd = torch.load(str(ckpt), map_location="cpu")
    if isinstance(sd, dict):
        for k in ("model", "state_dict"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    try:
        model.load_state_dict(sd)
    except RuntimeError as e:
        raise RuntimeError(f"'{ckpt.name}' does not match the LiteMedSAM architecture — is it "
                           f"the lite_medsam.pth checkpoint? ({str(e).splitlines()[0]})") from e
    model.eval()
    _MEDSAM["model"] = model
    return model


def _medsam_preprocess(image01):
    """LiteMedSAM's pipeline: longest side -> 256, min-max normalise, pad to 256x256."""
    from PIL import Image
    H, W = image01.shape
    scale = IMG_SIZE / max(H, W)
    nh, nw = int(round(H * scale)), int(round(W * scale))
    im = Image.fromarray((np.clip(image01, 0, 1) * 255).astype(np.uint8)).resize((nw, nh), Image.BICUBIC)
    arr = np.asarray(im).astype(np.float32) / 255.0
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / max(1e-8, hi - lo)
    padded = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.float32)
    padded[:nh, :nw] = np.stack([arr, arr, arr], axis=-1)
    return padded, (nh, nw), (H, W), scale


def _medsam_segment(image01, point=None, box=None, key=None):
    """Give a box (x0,y0,x1,y1) — a point becomes a small box (LiteMedSAM is box-only)."""
    import torch
    model = _build_medsam()
    if key is None or key != _MEDSAM["last_key"] or _MEDSAM["embed"] is None:
        padded, new_size, orig_size, scale = _medsam_preprocess(image01)
        t = torch.tensor(padded).float().permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            _MEDSAM["embed"] = model.image_encoder(t)
        _MEDSAM["meta"] = (new_size, orig_size, scale)
        _MEDSAM["last_key"] = key
    new_size, orig_size, scale = _MEDSAM["meta"]

    if box is None:
        if point is None:
            raise ValueError("need a point or a box prompt")
        x, y = float(point[0]), float(point[1])
        r = 16.0  # click -> a small box around it
        box = (x - r, y - r, x + r, y + r)
    H, W = orig_size
    b = np.array(box, dtype=np.float32)
    b = np.array([max(0, b[0]), max(0, b[1]), min(W - 1, b[2]), min(H - 1, b[3])], dtype=np.float32)
    box_256 = b * scale  # same scale on both axes (aspect preserved)

    with torch.no_grad():
        box_t = torch.as_tensor(box_256[None, None, ...], dtype=torch.float)
        sparse, dense = model.prompt_encoder(points=None, boxes=box_t, masks=None)
        # multimask_output=True + picking the best-scored candidate (standard SAM/MedSAM
        # practice for ambiguous prompts) — forcing a single mask on a small/tight box
        # otherwise tends to degenerate into "just fill the box".
        low_res, iou_pred = model.mask_decoder(
            image_embeddings=_MEDSAM["embed"],
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
        )
        best = int(torch.argmax(iou_pred, dim=1).item())
        low_res = low_res[:, best:best + 1]
        pred = model.postprocess_masks(low_res, new_size, orig_size)
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()
    return (pred > 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# MobileSAM — point prompt (box also supported), lung layer
# ---------------------------------------------------------------------------
def _mobilesam_checkpoint_path():
    if CFG["mobilesam_checkpoint"]:
        return Path(CFG["mobilesam_checkpoint"])
    env = os.environ.get("MOBILESAM_CHECKPOINT")
    if env:
        return Path(env)
    models = Path(__file__).with_name("models")
    for cand in (models / "mobile_sam.pt", models / "mobile_sam.pth"):
        if cand.exists():
            return cand
    return models / "mobile_sam.pt"


def _mobilesam_repo_path():
    """Folder holding the MobileSAM code (must contain mobile_sam/__init__.py)."""
    for cand in (CFG["mobilesam_repo"], os.environ.get("MOBILESAM_REPO")):
        if cand:
            return Path(cand)
    here = Path(__file__).parent
    p = here / "MobileSAM"
    return p


def _mobilesam_available():
    """True if the code and checkpoint are present (cheap — no torch import)."""
    return (_mobilesam_repo_path() / "mobile_sam" / "__init__.py").exists() and _mobilesam_checkpoint_path().exists()


def _build_mobilesam():
    if _MOBILESAM["predictor"] is not None:
        return _MOBILESAM["predictor"]

    repo, ckpt = _mobilesam_repo_path(), _mobilesam_checkpoint_path()
    if not (repo / "mobile_sam" / "__init__.py").exists():
        raise FileNotFoundError(f"MobileSAM code not found at {repo} (need mobile_sam/) — "
                                f"clone github.com/ChaoningZhang/MobileSAM or pass --mobilesam-repo")
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found at {ckpt}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from mobile_sam import sam_model_registry, SamPredictor

    model = sam_model_registry["vit_t"](checkpoint=str(ckpt))
    model.eval()
    predictor = SamPredictor(model)
    _MOBILESAM["predictor"] = predictor
    return predictor


def _mobilesam_segment(image01, point=None, box=None, key=None):
    """One click (point) segments the structure under it; a dragged box also works."""
    predictor = _build_mobilesam()
    if key is None or key != _MOBILESAM["last_key"]:
        gray = (np.clip(image01, 0, 1) * 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)
        predictor.set_image(rgb)
        _MOBILESAM["last_key"] = key

    point_coords = point_labels = box_arr = None
    if box is not None:
        H, W = image01.shape
        b = np.array(box, dtype=np.float32)
        box_arr = np.array([max(0, b[0]), max(0, b[1]), min(W - 1, b[2]), min(H - 1, b[3])])
    elif point is not None:
        point_coords = np.array([[float(point[0]), float(point[1])]], dtype=np.float32)
        point_labels = np.array([1])
    else:
        raise ValueError("need a point or a box prompt")

    masks, _scores, _logits = predictor.predict(
        point_coords=point_coords, point_labels=point_labels, box=box_arr, multimask_output=False)
    return masks[0].astype(np.uint8)
