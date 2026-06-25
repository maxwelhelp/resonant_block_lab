#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from resonant_block_lab import TinyParallelResonantClassifier


class SyntheticSeqDataset(Dataset):
    def __init__(self, n=4096, length=128, input_dim=16, classes=8, seed=1):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, length, input_dim, generator=g) * 0.15
        self.y = torch.randint(0, classes, (n,), generator=g)
        t = torch.linspace(0, 1, length)
        for i in range(n):
            c = int(self.y[i])
            freq = 1 + c % 5
            ch = c % input_dim
            self.x[i, :, ch] += torch.sin(2 * math.pi * freq * t) * 0.8
            self.x[i, length // 4:length // 4 + 6, (ch + 1) % input_dim] += 0.6 + 0.1 * c
            trend = torch.linspace(-0.5, 0.5, length) if c % 2 else torch.linspace(0.5, -0.5, length)
            self.x[i, :, (ch + 2) % input_dim] += trend

    def __len__(self):
        return self.y.numel()

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def sequence_ce(step_logits, y):
    b, m, c = step_logits.shape
    target = y[:, None].expand(b, m).reshape(b * m)
    ce = F.cross_entropy(step_logits.reshape(b * m, c), target, reduction='none').view(b, m)
    w = torch.linspace(0.2, 1.0, m, device=y.device).view(1, m)
    return (ce * w).sum() / (w.sum() * b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--length', type=int, default=128)
    p.add_argument('--input_dim', type=int, default=16)
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--classes', type=int, default=8)
    p.add_argument('--n_modes', type=int, default=8)
    p.add_argument('--micro_steps', type=int, default=3)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir', default='runs/synthetic_smoke')
    p.add_argument('--limit_batches', type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr = DataLoader(SyntheticSeqDataset(4096, args.length, args.input_dim, args.classes, seed=1), batch_size=args.batch, shuffle=True)
    va = DataLoader(SyntheticSeqDataset(1024, args.length, args.input_dim, args.classes, seed=999), batch_size=args.batch)
    model = TinyParallelResonantClassifier(args.input_dim, args.dim, args.classes, args.n_modes, args.micro_steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    print('params', sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

    rows = []
    for ep in range(1, args.epochs + 1):
        model.train()
        ok = tot = 0
        loss_sum = 0.0
        t0 = time.time()
        last_stats = None
        bi = 0
        for bi, (x, y) in enumerate(tr, 1):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits, stats = model(x, return_stats=True)
            loss = F.cross_entropy(logits, y)
            if 'step_logits' in stats:
                loss = loss + 0.35 * sequence_ce(stats['step_logits'], y)
            loss = loss + 0.01 * torch.relu(torch.tensor(0.03, device=device) - torch.sigmoid(model.block.branch_gate)).pow(2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ok += (logits.argmax(-1) == y).sum().item()
            tot += y.numel()
            loss_sum += float(loss.detach())
            last_stats = stats
            if args.limit_batches and bi >= args.limit_batches:
                break

        model.eval()
        vok = vtot = 0
        with torch.no_grad():
            for x, y in va:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                vok += (logits.argmax(-1) == y).sum().item()
                vtot += y.numel()
        row = {
            'epoch': ep,
            'train_acc': ok / max(1, tot),
            'val_acc': vok / max(1, vtot),
            'loss': loss_sum / max(1, bi),
            'gate': float(torch.sigmoid(model.block.branch_gate).detach().cpu()),
            'sec': time.time() - t0,
        }
        if last_stats is not None:
            row['delta_last'] = float(last_stats['deltas'][:, -1].mean().detach().cpu())
            row['attn_entropy'] = float(last_stats['attn_entropy'].detach().cpu())
            row['program'] = last_stats['program']
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False)[:1000], flush=True)

    (out / 'history.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    torch.save({'model': model.state_dict(), 'args': vars(args), 'rows': rows}, out / 'best.pt')
    print('saved', out, flush=True)


if __name__ == '__main__':
    main()
