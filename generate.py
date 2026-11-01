"""Generate skins.

    python generate.py "pirate,beard,classic" -n 8 --seed 7
    python generate.py --list-tags
    python generate.py "pirate" --weights weights/                  # packed weights (export_weights.py)

Writes out/<prompt>_<i>.png (real, loadable 64x64 skins) + out/<prompt>_grid.png.
"""
import argparse
import os

import torch

from skingen import config as C
from skingen.data import prompt_to_vec
from skingen.maskgit import MaskGIT, generate
from skingen.utils import save_grid, to_skin
from skingen.vqvae import load_frozen

p = argparse.ArgumentParser()
p.add_argument("prompt", nargs="?", default="")
p.add_argument("-n", type=int, default=8)
p.add_argument("--seed", type=int, default=None)
p.add_argument("--guidance", type=float, default=C.GUIDANCE)
p.add_argument("--maskgit", default=os.path.join(C.RUNS_DIR, C.MG_RUN, "maskgit.pth"))
p.add_argument("--vqvae", default=os.path.join(C.RUNS_DIR, C.VQ_RUN, "vqvae.pth"))
p.add_argument("--weights", default=None,
               help="packed weights folder from export_weights.py (default: the training checkpoints)")
p.add_argument("--out", default=C.OUT_DIR)
p.add_argument("--list-tags", action="store_true")
args = p.parse_args()

if args.weights:
    from skingen.weights import load_weights
    model, vq, vocab = load_weights(args.weights)
else:
    ck = torch.load(args.maskgit, map_location=C.DEVICE)
    # Column order ALWAYS comes from the checkpoint ("categories" = older checkpoints)
    vocab = ck.get("vocab") or ck["categories"]
    for k, v in ck.get("arch", {}).items():   # architecture of the checkpoint, not of the config
        setattr(C, k, v)
    model = MaskGIT(len(vocab)).to(C.DEVICE)
    model.load_state_dict(ck["model"])
    vq = load_frozen(args.vqvae)

if args.list_tags:
    print("\n".join(vocab))
    raise SystemExit

y, unknown = prompt_to_vec(args.prompt, vocab)
if unknown:
    raise SystemExit(f"Unknown tags: {unknown}  (show all: --list-tags)")
if args.seed is not None:
    torch.manual_seed(args.seed)

toks = generate(model, y.to(C.DEVICE).repeat(args.n, 1), guidance=args.guidance)
imgs = vq.decode(toks)

os.makedirs(args.out, exist_ok=True)
name = args.prompt.replace(",", "_").replace(" ", "-") or "unconditional"
for i, im in enumerate(imgs):
    to_skin(im).save(os.path.join(args.out, f"{name}_{i}.png"))
save_grid(imgs, os.path.join(args.out, f"{name}_grid.png"), nrow=min(args.n, 8))
print(f"{args.n} skins -> {args.out}")
