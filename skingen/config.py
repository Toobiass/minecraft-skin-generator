"""Every setting in one place.

Each number is the best value measured in the original project; the comment says
why. If you change something: one change per run, and a new RUN name.
"""
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo folder


def _load_dotenv(path):
    """Minimal .env reader (KEY=value per line, # = comment), no extra package.
    Real environment variables always win over the file."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


_load_dotenv(os.path.join(ROOT, ".env"))

# --- Paths ----------------------------------------------------------------------
# Default: everything inside the repo (data/, runs/). Own paths go into .env
# (template: .env.example) or directly into environment variables:
#   SKIN_DIR=~/data/skins TAGS_JSON=~/data/tags.json python train_maskgit.py
def _path(key, default):
    """Env value or default; relative paths are relative to the repo, not the cwd."""
    return os.path.normpath(os.path.join(ROOT, os.path.expanduser(os.environ.get(key, default))))


SKIN_DIR  = _path("SKIN_DIR", "data/skins")
TAGS_JSON = _path("TAGS_JSON", "data/tags.json")
RUNS_DIR  = _path("RUNS_DIR", "runs")
OUT_DIR   = _path("OUT_DIR", "out")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Stage 1: VQ-VAE ------------------------------------------------------------
VQ_RUN     = "vqvae"
IMAGE_SIZE = 64
CHANNELS   = 4      # RGBA - the silhouette lives in the alpha channel
CODE_DIM   = 64
NUM_CODES  = 512    # the alphabet = token vocabulary of stage 2
DOWNSAMPLE = 2      # 64 -> 32x32 = 1024 tokens. /4 came out soft.
BASE_CH    = 128

VQ_BATCH        = 128
VQ_EPOCHS       = 200
VQ_LR           = 2e-4
COMMITMENT_COST = 0.25
EMA_DECAY       = 0.99   # codebook updated by EMA instead of gradients - more stable
DEAD_CODE_THRESH = 1.0   # codes that are never picked get recycled
ALPHA_WEIGHT    = 2.0    # alpha counts double -> hard silhouette

# --- Stage 2: MaskGIT -----------------------------------------------------------
MG_RUN  = "maskgit"
DIM     = 384    # 15 M parameters, the size of the best measured model
DEPTH   = 8
HEADS   = 6      # DIM must be divisible by HEADS
MLP     = 4
DROPOUT = 0.1

MG_BATCH     = 32     # 1024 tokens per sample cost a lot of attention memory
MG_EPOCHS    = 300    # 300 beat 500 in a blind A/B test
MG_LR        = 3e-4
MG_LR_MIN    = 1e-5
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.03
GRAD_CLIP    = 1.0

# --- Tags / conditioning --------------------------------------------------------
MIN_TAG_COUNT = 100        # below this the model measurably does not learn a tag
MODEL_LABELS  = ["classic", "slim"]   # arm model, from the "model" field
P_UNCOND      = 0.15       # share of steps without tags -> needed for CFG
SUBSET_DROPOUT = True      # the key lever: short prompts only work with it

# --- Generation -----------------------------------------------------------------
GEN_STEPS   = 24
SAMPLE_TEMP = 0.8
CHOICE_TEMP = 4.5   # Gumbel noise on the confidence, annealed to zero
GUIDANCE    = 2.0   # sweet spot 1.5-2.0, from 2.5 on over-saturated

# --- Previews -------------------------------------------------------------------
SAMPLE_EVERY = 5
# Fixed set - keep it IDENTICAL across runs, otherwise previews are not comparable.
PREVIEW_PROMPTS = [
    "girl,pirate,cute,slim",
    "boy,blue hoodie,blond hair,blue eyes,classic",
    "creeper",
    "pirate",
    "",            # unconditional as a reference
]


def run_dir(name):
    d = os.path.join(RUNS_DIR, name)
    os.makedirs(os.path.join(d, "samples"), exist_ok=True)
    return d
