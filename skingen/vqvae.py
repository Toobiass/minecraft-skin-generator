"""Stage 1: VQ-VAE. Skin (64x64 RGBA) <-> 32x32 grid of token IDs (0..511)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.silu(self.n2(self.c1(F.silu(self.n1(x))))))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(C.CHANNELS, C.BASE_CH, 3, padding=1), ResBlock(C.BASE_CH)]
        ch = C.BASE_CH
        for _ in range(int(math.log2(C.DOWNSAMPLE))):
            out = min(ch * 2, 256)
            layers += [nn.Conv2d(ch, out, 4, stride=2, padding=1), ResBlock(out)]
            ch = out
        layers += [nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, C.CODE_DIM, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Nearest upsampling instead of ConvTranspose: no checkerboard artefacts in pixel art."""

    def __init__(self):
        super().__init__()
        n_up = int(math.log2(C.DOWNSAMPLE))
        ch = min(C.BASE_CH * 2 ** n_up, 256)
        layers = [nn.Conv2d(C.CODE_DIM, ch, 3, padding=1), ResBlock(ch)]
        for _ in range(n_up):
            out = max(ch // 2, C.BASE_CH)
            layers += [nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv2d(ch, out, 3, padding=1), ResBlock(out)]
            ch = out
        layers += [nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, C.CHANNELS, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return torch.tanh(self.net(z))


class VectorQuantizerEMA(nn.Module):
    """Nearest codebook entry by L2. Codebook updated by EMA, dead codes get recycled.

    Always computes in fp32 - the distance formula squares and overflows in fp16.
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        emb = torch.randn(C.NUM_CODES, C.CODE_DIM)
        self.register_buffer("embedding", emb)
        self.register_buffer("cluster_size", torch.zeros(C.NUM_CODES))
        self.register_buffer("ema_embed", emb.clone())

    def forward(self, z):
        in_dtype = z.dtype
        z = torch.nan_to_num(z.float()).clamp(-10.0, 10.0)
        b, _, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, C.CODE_DIM)

        # ||a-b||^2 = ||a||^2 - 2ab + ||b||^2
        d = (flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.embedding.t()
             + self.embedding.pow(2).sum(1))
        idx = d.argmin(1)
        onehot = F.one_hot(idx, C.NUM_CODES).float()
        z_q = self.embedding[idx].view(b, h, w, C.CODE_DIM).permute(0, 3, 1, 2)

        if self.training:
            with torch.no_grad():
                self.cluster_size.mul_(C.EMA_DECAY).add_(onehot.sum(0), alpha=1 - C.EMA_DECAY)
                self.ema_embed.mul_(C.EMA_DECAY).add_(onehot.t() @ flat, alpha=1 - C.EMA_DECAY)
                n = self.cluster_size.sum()
                cluster = (self.cluster_size + self.eps) / (n + C.NUM_CODES * self.eps) * n
                self.embedding.copy_((self.ema_embed / cluster.clamp(min=1e-2)[:, None]).clamp(-20, 20))
                dead = self.cluster_size < C.DEAD_CODE_THRESH
                if dead.any():
                    pick = torch.randint(0, flat.size(0), (int(dead.sum()),), device=flat.device)
                    self.embedding[dead] = flat[pick]
                    self.ema_embed[dead] = flat[pick]
                    self.cluster_size[dead] = 1.0

        commit = C.COMMITMENT_COST * F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()          # straight-through
        probs = onehot.mean(0)
        perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())
        return z_q.to(in_dtype), commit, perplexity, idx.view(b, h, w)


class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.vq = VectorQuantizerEMA()
        self.dec = Decoder()

    def forward(self, x):
        z_q, commit, ppl, idx = self.vq(self.enc(x))
        return self.dec(z_q), commit, ppl, idx

    @torch.no_grad()
    def encode(self, x):
        """(B,4,64,64) -> (B, 1024) token IDs."""
        return self.vq(self.enc(x))[3].flatten(1)

    @torch.no_grad()
    def decode(self, tokens):
        """(B, 1024) token IDs -> (B,4,64,64) in [-1,1]."""
        g = C.IMAGE_SIZE // C.DOWNSAMPLE
        z_q = self.vq.embedding[tokens].view(-1, g, g, C.CODE_DIM).permute(0, 3, 1, 2)
        return self.dec(z_q)


def load_frozen(path):
    m = VQVAE().to(C.DEVICE)
    m.load_state_dict(torch.load(path, map_location=C.DEVICE)["model"])
    m.eval().requires_grad_(False)
    return m
