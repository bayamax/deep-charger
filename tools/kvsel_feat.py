"""Features of a cached entry for the learned KV selector (shared by the kvsel harness, the trainer and the rollout).
sum6: (n, 6) attention received, summed over the hooked layers - cumulative, mean per observed chunk, last chunk, EMA 0.9, EMA 0.98,
max over chunks; nobs: chunks it could be attended in; age: chunks since it was written; rel: its index / the current position;
vnorm, knorm: (n, NL) mean norms of its values / keys per layer; tok: (n,) token ids (an embedding is learned)."""
import torch, torch.nn as nn

N_NUM = 6 + 3 + 56   # before the token embedding


def numeric(sum6, nobs, age, rel, vnorm, knorm):
    x = torch.cat([torch.log1p(1000.0 * sum6.float().clamp(min=0)), torch.log1p(nobs.float())[:, None], torch.log1p(age.float())[:, None],
                   rel.float()[:, None], vnorm.float(), knorm.float()], 1)
    return x


class Selector(nn.Module):
    def __init__(self, vocab, emb=16, hid=256, mu=None, sd=None):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.net = nn.Sequential(nn.Linear(N_NUM + emb, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
        self.register_buffer("mu", mu if mu is not None else torch.zeros(N_NUM)); self.register_buffer("sd", sd if sd is not None else torch.ones(N_NUM))

    def forward(self, x, tok):
        return self.net(torch.cat([(x - self.mu) / self.sd, self.emb(tok)], 1))[:, 0]
