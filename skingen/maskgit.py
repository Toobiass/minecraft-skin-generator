"""Stage 2: MaskGIT. Bidirectional transformer over 1024 tokens + a tag vector."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C

SEQ_LEN = (C.IMAGE_SIZE // C.DOWNSAMPLE) ** 2   # 1024
MASK_ID = C.NUM_CODES                           # 512 = [MASK]
VOCAB   = C.NUM_CODES + 1


class MaskGIT(nn.Module):
    """Tags are ADDED to every position.

    Additive is stable, cross-attention diverged reproducibly.
    keep=0 zeroes the tag signal including its bias -> a true "unconditional" for CFG.
    """

    def __init__(self, num_tags):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, C.DIM)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, C.DIM) * 0.02)
        self.cat_proj = nn.Linear(num_tags, C.DIM)
        layer = nn.TransformerEncoderLayer(C.DIM, C.HEADS, C.DIM * C.MLP, dropout=C.DROPOUT,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, C.DEPTH, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(C.DIM)
        self.head = nn.Linear(C.DIM, VOCAB)

    def forward(self, x, tags, keep):
        h = self.tok(x) + self.pos + (self.cat_proj(tags) * keep[:, None]).unsqueeze(1)
        return self.head(self.norm(self.blocks(h)))


def random_masking(tokens):
    """Mask ratio per sample cos(u*pi/2) - the model has to handle EVERY fill level,
    because generation starts at 100 % masked."""
    b, l = tokens.shape
    ratio = torch.cos(torch.rand(b, device=tokens.device) * math.pi / 2)
    n_mask = (ratio * l).long().clamp(min=1)
    rand = torch.rand(b, l, device=tokens.device)
    cutoff = rand.sort(1).values.gather(1, (n_mask - 1)[:, None])
    mask = rand <= cutoff
    return torch.where(mask, MASK_ID, tokens), mask


def subset_dropout(tags):
    """Keep only a random subset of the tags per sample (share ~ U(0,1)).

    This way the model sees sparse prompts during training - otherwise `pirate`
    alone doesn't work. At least one tag stays: an empty vector with keep=1 would
    be a second, wrong "nothing" next to the real unconditional.
    """
    if not C.SUBSET_DROPOUT:
        return tags
    frac = torch.rand(tags.size(0), 1, device=tags.device)
    out = tags * (torch.rand_like(tags) < frac)
    empty = (out.sum(1) == 0) & (tags.sum(1) > 0)
    if empty.any():
        pick = torch.where(tags > 0, torch.rand_like(tags), -1.0).argmax(1)
        out[empty, pick[empty]] = 1.0
    return out


@torch.no_grad()
def generate(model, tags, steps=C.GEN_STEPS, guidance=C.GUIDANCE,
             sample_temp=C.SAMPLE_TEMP, choice_temp=C.CHOICE_TEMP):
    """All [MASK] -> each round guess everything, keep the most confident, re-mask the rest.

    CFG: logits = u + guidance * (c - u). [MASK] is never sampled.
    """
    model.eval()
    n, dev = tags.size(0), tags.device
    cur = torch.full((n, SEQ_LEN), MASK_ID, device=dev)
    on, off = torch.ones(n, device=dev), torch.zeros(n, device=dev)

    for t in range(steps):
        logits = model(cur, tags, on)[..., :C.NUM_CODES].float()
        if guidance != 1.0:
            u = model(cur, tags, off)[..., :C.NUM_CODES].float()
            logits = u + guidance * (logits - u)

        probs = F.softmax(logits / sample_temp, -1)
        sampled = torch.multinomial(probs.view(-1, C.NUM_CODES), 1).view(n, SEQ_LEN)
        conf = probs.gather(-1, sampled[..., None]).squeeze(-1)

        unknown = cur == MASK_ID
        sampled = torch.where(unknown, sampled, cur)
        conf = torch.where(unknown, conf, torch.inf)

        done = (t + 1) / steps
        gumbel = -torch.log(-torch.log(torch.rand_like(conf) + 1e-9) + 1e-9)
        conf = conf + choice_temp * (1 - done) * gumbel

        cur = sampled
        n_mask = 0 if t == steps - 1 else int(SEQ_LEN * math.cos(math.pi / 2 * done))
        if n_mask > 0:
            cur = cur.scatter(1, conf.topk(n_mask, largest=False).indices, MASK_ID)
    return cur


def lr_lambda(step, total):
    """Linear warmup, then cosine down to LR_MIN."""
    if step < C.WARMUP_STEPS:
        return step / C.WARMUP_STEPS
    prog = min((step - C.WARMUP_STEPS) / max(1, total - C.WARMUP_STEPS), 1.0)
    return max(C.MG_LR_MIN / C.MG_LR, 0.5 * (1 + math.cos(math.pi * prog)))
