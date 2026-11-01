"""Stage 1: train the VQ-VAE.   python train_vqvae.py

Needs only PNGs, no tags. Result: runs/vqvae/vqvae.pth

CAREFUL: a new VQ-VAE = a new alphabet = every MaskGIT trained on the old one is
worthless. Once the run is finished (VQ_EPOCHS reached), a restart aborts.
Judge by eye on samples/recon*.png (top original, bottom reconstruction) and by
recon L1 (~0.013 is good) + perplexity (well above 50, otherwise it collapsed).
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from skingen import config as C
from skingen.data import load_images
from skingen.utils import log_metrics, save_atomic, save_grid
from skingen.vqvae import VQVAE

RUN  = C.run_dir(C.VQ_RUN)
CKPT = os.path.join(RUN, "vqvae.pth")


def to_unit(x):
    """uint8 [0,255] -> float [-1,1]. Exactly what data.to_tensor computes
    (ToTensor: /255, Normalize: -0.5, /0.5), just on the GPU and per batch."""
    return x.float().div_(255).sub_(0.5).div_(0.5)

if __name__ == "__main__":
    _, imgs = load_images()
    # Everything in RAM, as uint8: 16 KB per skin (~1.5 GB at 96k). As float32 it
    # was 64 KB per skin, and building it held two copies at once - ~14 GB peak
    # at 96k, which pushed a 16 GB machine into swap (~10x slower).
    arr = np.empty((len(imgs), C.IMAGE_SIZE, C.IMAGE_SIZE, C.CHANNELS), dtype=np.uint8)
    for i, im in enumerate(imgs):
        arr[i] = np.asarray(im)
    del imgs
    data = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()   # (N, 4, 64, 64)
    del arr
    loader = DataLoader(data, batch_size=C.VQ_BATCH, shuffle=True, drop_last=True,
                        pin_memory=True)
    preview = to_unit(data[:16])
    print(f"{len(data):,} skins | {len(loader)} steps/epoch | {C.DEVICE}")

    model = VQVAE().to(C.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=C.VQ_LR, weight_decay=0.0)
    ch_w = torch.tensor([1, 1, 1, C.ALPHA_WEIGHT], device=C.DEVICE).view(1, 4, 1, 1)

    start = 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=C.DEVICE)
        if ck["epoch"] + 1 >= C.VQ_EPOCHS:
            raise SystemExit(f"{CKPT} is fully trained. New VQ-VAE = new VQ_RUN name.")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["epoch"] + 1
        print(f"Resuming at epoch {start + 1}")

    for epoch in range(start, C.VQ_EPOCHS):
        model.train()
        s_rec = s_commit = s_ppl = 0.0
        for x in (pbar := tqdm(loader, desc=f"Epoch {epoch + 1}/{C.VQ_EPOCHS}")):
            x = to_unit(x.to(C.DEVICE, non_blocking=True))
            # No AMP: the quantiser overflows in fp16 -> NaN.
            rec, commit, ppl, _ = model(x)
            l1 = (F.l1_loss(rec, x, reduction="none") * ch_w).mean()
            loss = l1 + commit
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
            s_rec += l1.item(); s_commit += commit.item(); s_ppl += ppl.item()
            pbar.set_postfix(l1=f"{l1.item():.4f}", ppl=f"{ppl.item():.0f}")

        n = len(loader)
        log_metrics(os.path.join(RUN, "metrics.csv"), epoch + 1,
                    loss=f"{(s_rec + s_commit) / n:.6f}", recon_l1=f"{s_rec / n:.6f}",
                    perplexity=f"{s_ppl / n:.1f}")
        print(f"Epoch {epoch + 1} | recon L1 {s_rec / n:.4f} | perplexity {s_ppl / n:.0f}/{C.NUM_CODES}")
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch}, CKPT)

        if (epoch + 1) % C.SAMPLE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                rec = model(preview.to(C.DEVICE))[0]
            save_grid(torch.cat([preview.to(C.DEVICE), rec]),
                      os.path.join(RUN, "samples", f"recon{epoch + 1:04d}.png"), nrow=16)
