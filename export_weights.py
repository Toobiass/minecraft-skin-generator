"""Pack trained checkpoints into a small weights folder.   python export_weights.py

The training checkpoints (runs/*/maskgit.pth, runs/*/vqvae.pth) also hold the optimizer
state and are about three times larger than needed. This writes only what generating
needs:

    weights/maskgit.safetensors
    weights/vqvae.safetensors
    weights/config.json        tag vocabulary + architecture

generate.py --weights weights/ and demo/app.py load from this folder.
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file

from skingen import config as C

p = argparse.ArgumentParser()
p.add_argument("--maskgit", default=os.path.join(C.RUNS_DIR, C.MG_RUN, "maskgit.pth"))
p.add_argument("--vqvae", default=os.path.join(C.RUNS_DIR, C.VQ_RUN, "vqvae.pth"))
p.add_argument("--out", default=os.path.join(C.ROOT, "weights"))
args = p.parse_args()


def export(ckpt, path):
    """Checkpoint -> safetensors with only the weights. -> number of parameters."""
    sd = torch.load(ckpt, map_location="cpu")["model"]
    save_file({k: v.contiguous() for k, v in sd.items()}, path)
    return sum(v.numel() for v in sd.values())


if __name__ == "__main__":
    shutil.rmtree(args.out, ignore_errors=True)
    os.makedirs(args.out)

    ck = torch.load(args.maskgit, map_location="cpu")
    vocab = ck.get("vocab") or ck["categories"]
    n_mg = export(args.maskgit, os.path.join(args.out, "maskgit.safetensors"))
    n_vq = export(args.vqvae, os.path.join(args.out, "vqvae.safetensors"))
    cfg = {
        "vocab": vocab,
        "maskgit": {"DIM": C.DIM, "DEPTH": C.DEPTH, "HEADS": C.HEADS, "MLP": C.MLP, **ck.get("arch", {})},
        "vqvae": {k: getattr(C, k) for k in
                  ("IMAGE_SIZE", "CHANNELS", "CODE_DIM", "NUM_CODES", "DOWNSAMPLE", "BASE_CH")},
        "epoch": ck.get("epoch", -1) + 1,
    }
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"MaskGIT {n_mg / 1e6:.1f} M + VQ-VAE {n_vq / 1e6:.1f} M parameters, "
          f"{len(vocab)} tags -> {args.out}")
