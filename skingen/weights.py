"""Load the packed weights folder written by export_weights.py.

    model, vq, vocab = load_weights("weights")
"""
import json
import os

from safetensors.torch import load_file

from . import config as C
from .maskgit import MaskGIT
from .vqvae import VQVAE


def load_weights(src):
    """-> (maskgit, vqvae, vocab), both on C.DEVICE in eval mode."""
    if not os.path.isdir(src):
        raise FileNotFoundError(f"{src} not found - create it with: python export_weights.py")
    with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    # The token alphabet is baked into module constants (SEQ_LEN, MASK_ID) - it can't
    # be switched at runtime, only checked.
    wrong = {k: v for k, v in cfg["vqvae"].items() if getattr(C, k) != v}
    if wrong:
        raise ValueError(f"{src} was trained with {wrong}, skingen/config.py differs.")
    for k, v in cfg["maskgit"].items():   # architecture of the export, not of the config
        setattr(C, k, v)

    vocab = cfg["vocab"]
    model = MaskGIT(len(vocab)).to(C.DEVICE)
    model.load_state_dict(load_file(os.path.join(src, "maskgit.safetensors"), device=C.DEVICE))
    vq = VQVAE().to(C.DEVICE)
    vq.load_state_dict(load_file(os.path.join(src, "vqvae.safetensors"), device=C.DEVICE))
    model.eval().requires_grad_(False)
    vq.eval().requires_grad_(False)
    return model, vq, vocab
