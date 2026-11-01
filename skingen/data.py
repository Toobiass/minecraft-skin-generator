"""Dataset: a folder of PNGs plus one tag file.

Format (see data/README.md):

    data/skins/<id>.png        64x64 RGBA
    data/tags.json             [{"id": "<id>", "tags": ["pirate", ...], "model": "classic"}, ...]

`python -m skingen.data` only prints statistics - a good check before a long run.
"""
import hashlib
import json
import os
from collections import Counter

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from . import config as C

# [0,255] -> [-1,1], matching the tanh in the decoder. Must be identical for
# training AND tokenising, otherwise the encoder produces wrong tokens.
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 4, [0.5] * 4),
])

# Tags that don't describe anything a skin has or doesn't have.
BANNED_TAGS = {
    "minecraft", "minecraft skin", "skin", "character", "person", "figure", "pixel art",
    "avatar", "render", "unknown", "none", "hair", "eyes", "body", "clothes", "outfit",
    "head", "face", "color", "colors", "nose", "mouth", "legs", "arms", "hands",
    "eye color", "hair color", "body features", "3d", "cube", "mobs",
}
# Bare colours without an object are the proven cause of colour leaking:
# "red" tints the whole skin, "red hair" only the hair.
BARE_COLORS = {
    "black", "white", "gray", "red", "blue", "green", "yellow", "orange", "purple",
    "pink", "brown", "beige", "cyan", "turquoise", "gold", "silver", "tan", "violet",
    "dark", "light", "bright", "neon",
}


def clean_tag(t):
    t = t.strip().lower() if isinstance(t, str) else ""
    if not t or t in BANNED_TAGS or t in BARE_COLORS or t.startswith(("no ", "without ")):
        return None
    return t


def load_images(ids=None):
    """-> (ids, [PIL]) sorted. Only real 64x64, exact duplicates (md5) are dropped."""
    files = sorted(f for f in os.listdir(C.SKIN_DIR) if f.endswith(".png"))
    out_ids, imgs, seen = [], [], set()
    for fn in tqdm(files, desc="Loading skins"):
        sid = fn[:-4]
        if ids is not None and sid not in ids:
            continue
        try:
            img = Image.open(os.path.join(C.SKIN_DIR, fn)).convert("RGBA")
        except Exception:
            continue
        if img.size != (C.IMAGE_SIZE, C.IMAGE_SIZE):
            continue                      # old 64x32 format
        h = hashlib.md5(img.tobytes()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out_ids.append(sid)
        imgs.append(img)
    return out_ids, imgs


def load_meta():
    """tags.json -> {id: {"tags": [...], "model": ...}}. A list or a dict both work."""
    with open(C.TAGS_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("skins", raw)
        raw = [dict(v, id=v.get("id", k)) for k, v in raw.items()]
    meta = {}
    for e in raw:
        sid = e.get("id") or os.path.splitext(e.get("file", ""))[0]
        if not sid:
            continue
        tags = sorted({t for t in map(clean_tag, e.get("tags") or []) if t})
        meta[sid] = {"tags": tags, "model": e.get("model") or e.get("arm_model")}
    return meta


def build_vocab(meta):
    """Fixed column order: sorted tags + arm model.

    The order IS the meaning of the columns. It is stored in the checkpoint and
    read from there when generating, never recomputed.
    """
    counts = Counter(t for e in meta.values() for t in e["tags"])
    style = sorted(t for t, n in counts.items() if n >= C.MIN_TAG_COUNT)
    return style + [m for m in C.MODEL_LABELS if m not in style], counts


def tag_matrix(meta, ids, vocab):
    """Multi-hot (N, len(vocab)) in exactly the order of `ids`."""
    index = {t: i for i, t in enumerate(vocab)}
    mat = torch.zeros(len(ids), len(vocab))
    for row, sid in enumerate(ids):
        e = meta[sid]
        for t in e["tags"] + [e["model"]]:
            if t in index:
                mat[row, index[t]] = 1.0
    return mat


def prompt_to_vec(prompt, vocab):
    """'pirate,beard' -> multi-hot. Unknown tags are reported, not silently dropped."""
    y = torch.zeros(len(vocab))
    unknown = []
    for t in (p.strip().lower() for p in prompt.split(",")):
        if not t:
            continue
        if t in vocab:
            y[vocab.index(t)] = 1.0
        else:
            unknown.append(t)
    return y, unknown


if __name__ == "__main__":
    meta = load_meta()
    vocab, counts = build_vocab(meta)
    n_png = sum(f.endswith(".png") for f in os.listdir(C.SKIN_DIR))
    print(f"PNGs:          {n_png:,}  ({C.SKIN_DIR})")
    print(f"Entries:       {len(meta):,}  ({C.TAGS_JSON})")
    print(f"Tags total:    {len(counts):,}, of which >= {C.MIN_TAG_COUNT}x: {len(vocab) - len(C.MODEL_LABELS)}")
    print(f"Columns:       {len(vocab)} (incl. {C.MODEL_LABELS})")
    print(f"Tags/skin:     {sum(len(e['tags']) for e in meta.values()) / max(len(meta), 1):.1f}")
    print(f"Most common:   {', '.join(f'{t} ({n})' for t, n in counts.most_common(15))}")
