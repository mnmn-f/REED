"""Paper equations 7–20. Decoder weights stay frozen throughout the stream."""
import math
import torch
from torch import nn


class OnlineAdapters(nn.Module):
    def __init__(self, reference, aps, beta, classes=50, rank=32):
        super().__init__()
        dim = reference.shape[-1]
        self.register_buffer("reference", reference.float())
        self.register_buffer("aps", aps.float())
        self.register_buffer("frequency", torch.full((2, classes), 1/classes))
        self.beta = beta
        self.age = 0
        self.stable = nn.ModuleList(nn.Sequential(nn.Linear(dim,rank,bias=False), nn.ReLU(),
                                                  nn.Linear(rank,dim,bias=False)) for _ in range(2))
        self.plastic = nn.ModuleList(nn.Linear(dim,dim,bias=False) for _ in range(2))
        for stable, plastic in zip(self.stable,self.plastic):
            nn.init.zeros_(stable[-1].weight)
            nn.init.zeros_(plastic.weight)

    def step(self, decoder, raw, processed, noisy, changed, optimizer):
        if changed:
            self.age = 0
        alpha = .001 + .299*(1-.2**min(self.age/32,1))/.8
        self.age += 1
        noisy_params, clean_params, kl_terms, tokens = [], [], [], []
        for m,(x,y) in enumerate(zip(raw,processed)):
            if noisy[m]:
                u = y - (y.mean((0,1))-self.reference[m]) + self.stable[m](x).detach()
                tokens.append(u+self.plastic[m](u))
                noisy_params.extend(self.plastic[m].parameters())
            else:
                adapted = x+self.stable[m](x)
                tokens.append(adapted)
                clean_params.extend(self.stable[m].parameters())
                log_clean = decoder(adapted).log_softmax(-1)
                kl_terms.append((log_clean.exp()*(log_clean-decoder(x).detach().log_softmax(-1))).sum(-1).mean())
        logp = decoder(torch.cat(tokens,1)).log_softmax(-1)
        pred = logp.detach().exp()
        entropy = -(logp.exp()*logp).sum(-1).mean()
        ordered, idx = pred.sort(descending=True)
        rank = min(len(self.aps),math.ceil((len(self.aps)+1)*(1-alpha)))
        allowed = ordered.cumsum(-1)<=self.aps[rank-1]
        allowed[:,0] = True
        mask = torch.zeros_like(allowed).scatter_(1,idx,allowed)
        optimizer.zero_grad(set_to_none=True)
        if clean_params:
            torch.autograd.backward(.01*entropy+.1*torch.stack(kl_terms).mean(),inputs=clean_params,retain_graph=bool(noisy_params))
        if noisy_params:
            losses = []
            for m,active in enumerate(noisy):
                if active:
                    q = mask*pred/self.frequency[m].clamp_min(1e-4)
                    q = q/q.sum(-1,keepdim=True)
                    losses.append(-(q*logp).sum(-1).mean())
                    self.frequency[m].lerp_(pred.mean(0),1-self.beta)
            torch.autograd.backward(.01*entropy+.9*torch.stack(losses).mean(),inputs=noisy_params)
        optimizer.step()
        return pred
