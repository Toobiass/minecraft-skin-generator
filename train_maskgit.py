"""Stage 2: train MaskGIT with tags.   python train_maskgit.py

Needs runs/vqvae/vqvae.pth (frozen) + PNGs + tags.json.
Result: runs/maskgit/maskgit.pth - it contains the tag vocabulary, generate.py
needs nothing else.

DON'T JUDGE BY THE CE LOSS. It ends up around 2.1-2.2 for every variant, and the
best model had the worst value. Judge by the previews (samples/gen*.png, fixed
prompt set) and blind A/B tests.
"""
import os

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from skingen import config as C
from skingen.data import build_vocab, load_images, load_meta, prompt_to_vec, tag_matrix, to_tensor
from skingen.maskgit import MaskGIT, generate, lr_lambda, random_masking, subset_dropout
from skingen.utils import log_metrics, save_atomic, save_grid
from skingen.vqvae import load_frozen

RUN     = C.run_dir(C.MG_RUN)
CKPT    = os.path.join(RUN, "maskgit.pth")
VQ_CKPT = os.path.join(C.RUNS_DIR, C.VQ_RUN, "vqvae.pth")
TOKENS  = os.path.join(C.RUNS_DIR, C.VQ_RUN, "tokens.pt")


@torch.no_grad()
def get_tokens(vq, meta):
    """Tokenise all tagged skins, cached next to the VQ-VAE.

    The cache is only used if it matches the current data and the current VQ-VAE
    EXACTLY. Otherwise it is rebuilt (~1.5 min) - never silently keep training on
    an old subset.
    """
    # Fingerprint: which skins would be used (PNG exists AND tagged) + which VQ-VAE.
    wanted = sorted({f[:-4] for f in os.listdir(C.SKIN_DIR) if f.endswith(".png")} & set(meta))
    stamp = {"wanted": wanted, "vq_mtime": os.path.getmtime(VQ_CKPT)}
    if os.path.exists(TOKENS):
        d = torch.load(TOKENS)
        if d["stamp"] == stamp:
            print(f"Token cache: {TOKENS}")
            return d["tokens"], d["ids"]
        print("Token cache does not match data/VQ-VAE - tokenising again.")

    ids, imgs = load_images(set(wanted))
    chunks = []
    for i in tqdm(range(0, len(imgs), 256), desc="Tokenising"):
        x = torch.stack([to_tensor(im) for im in imgs[i:i + 256]]).to(C.DEVICE)
        chunks.append(vq.encode(x).cpu())
    tokens = torch.cat(chunks)
    torch.save({"tokens": tokens, "ids": ids, "stamp": stamp}, TOKENS)
    return tokens, ids


@torch.no_grad()
def preview(model, vq, vocab, path, per_row=8):
    rows = [p for p in C.PREVIEW_PROMPTS if not prompt_to_vec(p, vocab)[1]]
    y = torch.stack([prompt_to_vec(p, vocab)[0] for p in rows for _ in range(per_row)])
    toks = generate(model, y.to(C.DEVICE))
    save_grid(vq.decode(toks), path, nrow=per_row)
    return rows


if __name__ == "__main__":
    meta = load_meta()
    vocab, _ = build_vocab(meta)
    vq = load_frozen(VQ_CKPT)
    tokens, ids = get_tokens(vq, meta)
    tags = tag_matrix(meta, ids, vocab)
    print(f"{len(ids):,} skins | {len(vocab)} tag columns | {tags.sum(1).mean():.1f} tags/skin")

    loader = DataLoader(TensorDataset(tokens, tags), batch_size=C.MG_BATCH, shuffle=True,
                        drop_last=True, pin_memory=True)
    total = len(loader) * C.MG_EPOCHS

    model = MaskGIT(len(vocab)).to(C.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=C.MG_LR, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total))
    scaler = GradScaler(C.DEVICE, enabled=C.DEVICE == "cuda")
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters | "
          f"{len(loader)} steps/epoch")

    start = 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=C.DEVICE)
        if ck["vocab"] != vocab:
            raise SystemExit(f"{CKPT} has a different tag vocabulary. Pick a new MG_RUN name.")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start = ck["epoch"] + 1
        print(f"Resuming at epoch {start + 1}")

    for epoch in range(start, C.MG_EPOCHS):
        model.train()
        s_loss = s_tags = 0.0
        for x, y in (pbar := tqdm(loader, desc=f"Epoch {epoch + 1}/{C.MG_EPOCHS}")):
            x, y = x.to(C.DEVICE, non_blocking=True), y.to(C.DEVICE, non_blocking=True)
            inp, mask = random_masking(x)
            keep = (torch.rand(x.size(0), device=C.DEVICE) > C.P_UNCOND).float()
            y = subset_dropout(y)

            with autocast(C.DEVICE, enabled=C.DEVICE == "cuda"):
                logits = model(inp, y, keep)
            loss = F.cross_entropy(logits.float()[mask], x[mask])   # masked positions only

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            sched.step()

            s_loss += loss.item(); s_tags += y.sum(1).mean().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{sched.get_last_lr()[0]:.1e}")

        n = len(loader)
        log_metrics(os.path.join(RUN, "metrics.csv"), epoch + 1, loss=f"{s_loss / n:.6f}",
                    lr=f"{sched.get_last_lr()[0]:.2e}", tags_per_sample=f"{s_tags / n:.2f}")
        print(f"Epoch {epoch + 1} | CE {s_loss / n:.4f} | tags/sample after dropout {s_tags / n:.1f}")
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(),
                     "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": epoch, "vocab": vocab,
                     "arch": {"DIM": C.DIM, "DEPTH": C.DEPTH, "HEADS": C.HEADS}}, CKPT)

        if (epoch + 1) % C.SAMPLE_EVERY == 0:
            rows = preview(model, vq, vocab, os.path.join(RUN, "samples", f"gen{epoch + 1:04d}.png"))
            print(f"  Preview rows: {rows}")
