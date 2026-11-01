"""Small helpers both training scripts need."""
import csv
import os
import time

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image


def save_atomic(obj, path):
    """Write to .tmp first, then rename - an interruption mid-write never leaves a
    half checkpoint, and a parallel reader never sees a half file."""
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def log_metrics(path, epoch, **values):
    """One row per epoch in metrics.csv (always: epoch, timestamp, loss)."""
    row = {"epoch": epoch, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **values}
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def save_grid(imgs, path, nrow):
    """(N,4,H,W) in [-1,1] -> PNG, alpha over a checkerboard (transparent != black)."""
    imgs = ((imgs.float() + 1) / 2).clamp(0, 1)
    h, w = imgs.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    board = (((yy // 8) + (xx // 8)) % 2).float().to(imgs.device) * 0.2 + 0.4
    rgb = imgs[:, 3:4] * imgs[:, :3] + (1 - imgs[:, 3:4]) * board
    save_image(rgb, path, nrow=nrow)


def to_skin(img):
    """(4,64,64) in [-1,1] -> PIL RGBA, a real loadable skin. Alpha is binary in the game."""
    a = ((img.permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    a[..., 3] = np.where(a[..., 3] >= 128, 255, 0)
    return Image.fromarray(a, "RGBA")
