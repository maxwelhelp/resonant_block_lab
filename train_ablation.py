#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from resonant_block_lab import ResonantBlock, ResonantBlockConfig, ParallelResonantBlock


class SyntheticSeqDataset(Dataset):
    def __init__(self, n=4096, length=128, input_dim=16, classes=8, seed=1, hard=False):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, length, input_dim, generator=g) * (0.20 if hard else 0.15)
        self.y = torch.randint(0, classes, (n,), generator=g)
        t = torch.linspace(0, 1, length)
        for i in range(n):
            c = int(self.y[i])
            ch = c % input_dim
            freq = 1 + c % 5
            amp = 0.45 if hard else 0.8
            self.x[i, :, ch] += torch.sin(2 * math.pi * freq * t) * amp
            pos = (length // 5 + (c * 7) % max(8, length // 2)) if hard else length // 4
            pos = min(pos, length - 8)
            self.x[i, pos:pos + 6, (ch + 1) % input_dim] += (0.35 if hard else 0.6) + 0.05 * c
            trend = torch.linspace(-0.35, 0.35, length) if c % 2 else torch.linspace(0.35, -0.35, length)
            self.x[i, :, (ch + 2) % input_dim] += trend
            if hard:
                # Distractor channel and random crop-like phase offset.
                self.x[i, :, (ch + 3) % input_dim] += torch.sin(2 * math.pi * (freq + 2) * t) * 0.22
                shift = int(torch.randint(-8, 9, (1,), generator=g))
                self.x[i] = torch.roll(self.x[i], shifts=shift, dims=0)

    def __len__(self):
        return self.y.numel()

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class BaselineClassifier(nn.Module):
    def __init__(self, input_dim, dim, classes, ff_scale=0.5):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, dim)
        self.ff_scale = float(ff_scale)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.head = nn.Linear(dim * 4, classes)

    @staticmethod
    def features(x):
        return torch.cat([x.mean(1), x.std(1), x.max(1).values, x.pow(2).mean(1)], dim=-1)

    def forward(self, x, return_stats=False):
        h = self.in_proj(x)
        h = h + self.ff_scale * self.ff(h)
        logits = self.head(self.features(h))
        return (logits, {}) if return_stats else logits


class ParallelClassifier(nn.Module):
    def __init__(self, input_dim, dim, classes, n_modes, micro_steps, gate_init=-4.0, ff_scale=0.5):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, dim)
        cfg = ResonantBlockConfig(dim=dim, n_modes=n_modes, micro_steps=micro_steps, aux_classes=classes, gate_init=gate_init)
        self.block = ParallelResonantBlock(cfg)
        self.ff_scale = float(ff_scale)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.head = nn.Linear(dim * 4, classes)

    @staticmethod
    def features(x):
        return torch.cat([x.mean(1), x.std(1), x.max(1).values, x.pow(2).mean(1)], dim=-1)

    def forward(self, x, return_stats=False):
        h = self.in_proj(x)
        if return_stats:
            h, stats = self.block(h, return_stats=True)
        else:
            h = self.block(h)
            stats = {}
        h = h + self.ff_scale * self.ff(h)
        logits = self.head(self.features(h))
        return (logits, stats) if return_stats else logits


class ResonantOnlyClassifier(nn.Module):
    def __init__(self, input_dim, dim, classes, n_modes, micro_steps, ff_scale=0.0):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, dim)
        cfg = ResonantBlockConfig(dim=dim, n_modes=n_modes, micro_steps=micro_steps, aux_classes=classes)
        self.block = ResonantBlock(cfg)
        self.ff_scale = float(ff_scale)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.head = nn.Linear(dim * 4, classes)

    @staticmethod
    def features(x):
        return torch.cat([x.mean(1), x.std(1), x.max(1).values, x.pow(2).mean(1)], dim=-1)

    def forward(self, x, return_stats=False):
        h = self.in_proj(x)
        if return_stats:
            h, stats = self.block(h, return_stats=True)
        else:
            h = self.block(h)
            stats = {}
        if self.ff_scale:
            h = h + self.ff_scale * self.ff(h)
        logits = self.head(self.features(h))
        return (logits, stats) if return_stats else logits


def sequence_ce(step_logits, y):
    b, m, c = step_logits.shape
    target = y[:, None].expand(b, m).reshape(b * m)
    ce = F.cross_entropy(step_logits.reshape(b * m, c), target, reduction='none').view(b, m)
    w = torch.linspace(0.2, 1.0, m, device=y.device).view(1, m)
    return (ce * w).sum() / (w.sum() * b)


def build_model(args):
    if args.variant == 'baseline':
        return BaselineClassifier(args.input_dim, args.dim, args.classes, ff_scale=args.ff_scale)
    if args.variant == 'resonant_only':
        return ResonantOnlyClassifier(args.input_dim, args.dim, args.classes, args.n_modes, args.micro_steps, ff_scale=args.ff_scale)
    return ParallelClassifier(args.input_dim, args.dim, args.classes, args.n_modes, args.micro_steps, gate_init=args.gate_init, ff_scale=args.ff_scale)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['baseline', 'parallel', 'resonant_only'], default='parallel')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--length', type=int, default=128)
    p.add_argument('--input_dim', type=int, default=16)
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--classes', type=int, default=8)
    p.add_argument('--n_modes', type=int, default=8)
    p.add_argument('--micro_steps', type=int, default=3)
    p.add_argument('--gate_init', type=float, default=-4.0)
    p.add_argument('--ff_scale', type=float, default=0.5)
    p.add_argument('--deep_weight', type=float, default=0.35)
    p.add_argument('--hard', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir', default='runs/ablation')
    p.add_argument('--limit_batches', type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    out = Path(args.out_dir) / args.variant
    out.mkdir(parents=True, exist_ok=True)
    tr = DataLoader(SyntheticSeqDataset(4096, args.length, args.input_dim, args.classes, seed=1, hard=args.hard), batch_size=args.batch, shuffle=True)
    va = DataLoader(SyntheticSeqDataset(1024, args.length, args.input_dim, args.classes, seed=999, hard=args.hard), batch_size=args.batch)
    model = build_model(args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    print('variant', args.variant, 'params', sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    rows = []
    for ep in range(1, args.epochs + 1):
        model.train(); ok=tot=0; loss_sum=0.0; t0=time.time(); last_stats={}; bi=0
        for bi, (x, y) in enumerate(tr, 1):
            x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
            logits, stats = model(x, return_stats=True)
            loss = F.cross_entropy(logits, y)
            if 'step_logits' in stats and args.deep_weight > 0:
                loss = loss + args.deep_weight * sequence_ce(stats['step_logits'], y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            ok += (logits.argmax(-1)==y).sum().item(); tot += y.numel(); loss_sum += float(loss.detach()); last_stats=stats
            if args.limit_batches and bi >= args.limit_batches: break
        model.eval(); vok=vtot=0
        with torch.no_grad():
            for x,y in va:
                x=x.to(device); y=y.to(device); logits=model(x); vok += (logits.argmax(-1)==y).sum().item(); vtot += y.numel()
        gate = None
        if hasattr(model, 'block') and hasattr(model.block, 'branch_gate'):
            gate = float(torch.sigmoid(model.block.branch_gate).detach().cpu())
        row = {'epoch': ep, 'train_acc': ok/max(1,tot), 'val_acc': vok/max(1,vtot), 'loss': loss_sum/max(1,bi), 'gate': gate, 'sec': time.time()-t0}
        if last_stats:
            row['delta_last'] = float(last_stats.get('deltas', torch.zeros(1,1,device=device))[:, -1].mean().detach().cpu())
            if 'program' in last_stats:
                row['program'] = last_stats['program']
        rows.append(row); print(json.dumps(row, ensure_ascii=False)[:1000], flush=True)
    (out/'history.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    torch.save({'model': model.state_dict(), 'args': vars(args), 'rows': rows}, out/'best.pt')
    print('saved', out, flush=True)

if __name__ == '__main__':
    main()
