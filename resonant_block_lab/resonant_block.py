from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn


@dataclass
class ResonantBlockConfig:
    dim: int
    n_modes: int = 8
    micro_steps: int = 3
    controller_hidden: int = 128
    dropout: float = 0.0
    gate_init: float = -4.0
    aux_classes: int = 0
    max_steps: int = 32


class CrossReader(nn.Module):
    """Pooled O(B*L*D) reader, not full O(L^2) attention."""
    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.dim = int(dim)

    def read_precomputed(self, field: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.q(field.mean(dim=1)).unsqueeze(1)
        attn = torch.softmax(torch.bmm(query, keys.transpose(1, 2)) / math.sqrt(self.dim), dim=-1)
        ctx = torch.bmm(attn, values).squeeze(1)
        ent = -(attn.squeeze(1).clamp_min(1e-8).log() * attn.squeeze(1)).sum(dim=-1).mean()
        return ctx, ent

    def forward(self, field: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.read_precomputed(field, self.k(context), self.v(context))


class DynamicOperatorBank1D(nn.Module):
    """Replayable sequence operator bank for dynamic length [B,L,D]."""
    def __init__(self, dim: int, n_modes: int):
        super().__init__()
        self.dim = int(dim)
        self.n_modes = int(n_modes)
        self.n_fixed_modes = 6
        extra = max(0, n_modes - self.n_fixed_modes)
        self.dw = nn.ModuleList([nn.Conv1d(dim, dim, 3, padding=1, groups=dim, bias=False) for _ in range(extra)])
        self.mode_scale = nn.Parameter(torch.ones(n_modes))
        self.mode_bias = nn.Parameter(torch.zeros(n_modes, 1, dim))
        for conv in self.dw:
            nn.init.zeros_(conv.weight)
            with torch.no_grad():
                conv.weight[:, 0, 1] = 1.0

    @staticmethod
    def shift_left(x):
        return torch.cat([x[:, 1:, :], x[:, -1:, :]], dim=1)

    @staticmethod
    def shift_right(x):
        return torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)

    @staticmethod
    def local_avg(x):
        return (DynamicOperatorBank1D.shift_right(x) + x + DynamicOperatorBank1D.shift_left(x)) / 3.0

    def apply_all(self, x: torch.Tensor) -> torch.Tensor:
        ops = [x]
        if len(ops) < self.n_modes:
            ops.append(self.shift_left(x))
        if len(ops) < self.n_modes:
            ops.append(self.shift_right(x))
        if len(ops) < self.n_modes:
            ops.append(self.local_avg(x))
        if len(ops) < self.n_modes:
            ops.append(x.mean(dim=1, keepdim=True).expand_as(x))
        if len(ops) < self.n_modes:
            ops.append(x - self.local_avg(x))
        i = 0
        while len(ops) < self.n_modes:
            conv = self.dw[min(i, len(self.dw) - 1)]
            ops.append(conv(x.transpose(1, 2)).transpose(1, 2))
            i += 1
        return torch.stack(ops[:self.n_modes], dim=1)

    def mix(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        ops = self.apply_all(x)
        ops = ops * self.mode_scale.view(1, -1, 1, 1) + self.mode_bias.view(1, self.n_modes, 1, self.dim)
        return torch.einsum('br,brld->bld', q, ops)

    def replay_description(self, q: torch.Tensor, topk: int = 4) -> Dict:
        names = ['identity', 'shift_left', 'shift_right', 'local_avg', 'global_mean', 'highpass']
        while len(names) < self.n_modes:
            names.append(f'learned_depthwise_{len(names)-5}')
        q_mean = q.detach().float().mean(dim=0).cpu()
        vals, idx = q_mean.topk(min(topk, self.n_modes))
        return {
            'top_modes': [(names[int(i)], float(v)) for v, i in zip(vals, idx)],
            'all_weights': {names[i]: float(q_mean[i]) for i in range(self.n_modes)},
        }


class UnifiedResonantController(nn.Module):
    def __init__(self, dim: int, n_modes: int, hidden: int = 128, max_steps: int = 32):
        super().__init__()
        self.n_modes = int(n_modes)
        self.step_emb = nn.Embedding(max_steps, 32)
        self.norm = nn.LayerNorm(dim * 3 + 32)
        self.net = nn.Sequential(
            nn.Linear(dim * 3 + 32, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_modes + 5 + dim),
        )

    def forward(self, pool: torch.Tensor, ctx: torch.Tensor, memory: torch.Tensor, step_idx: int):
        b = pool.shape[0]
        idx = torch.full((b,), min(step_idx, self.step_emb.num_embeddings - 1), device=pool.device, dtype=torch.long)
        h = self.norm(torch.cat([pool, ctx, memory, self.step_emb(idx).to(pool.dtype)], dim=-1))
        y = self.net(h)
        a = 0
        q = torch.softmax(y[:, a:a+self.n_modes], dim=-1); a += self.n_modes
        gates = torch.sigmoid(y[:, a:a+5]); a += 5
        memory_next = torch.tanh(y[:, a:])
        return {'q': q, 'gates': gates, 'memory': memory_next}


class ResonantBlock(nn.Module):
    """Drop-in resonant sequence mixer: [B,L,D] -> [B,L,D]."""
    def __init__(self, config: ResonantBlockConfig):
        super().__init__()
        self.config = config
        d = config.dim
        self.reader = CrossReader(d)
        self.bank = DynamicOperatorBank1D(d, config.n_modes)
        self.controller = UnifiedResonantController(d, config.n_modes, config.controller_hidden, config.max_steps)
        self.state_proj = nn.Linear(d, d, bias=False)
        self.input_proj = nn.Linear(d, d)
        self.init_proj = nn.Linear(d, d, bias=False)
        self.memory_proj = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)
        self.aux_projector = nn.Linear(d * 4, config.aux_classes) if config.aux_classes > 0 else None

    @staticmethod
    def features(x):
        return torch.cat([x.mean(1), x.std(1), x.max(1).values, x.pow(2).mean(1)], dim=-1)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, return_stats: bool = False):
        if context is None:
            context = x
        phi = x
        init = x
        b, l, d = x.shape
        memory = torch.zeros(b, d, device=x.device, dtype=x.dtype)
        q_hist, gate_hist, deltas, aux, ents = [], [], [], [], []
        for m in range(self.config.micro_steps):
            ctx, ent = self.reader(phi, context)
            cfg = self.controller(phi.mean(1), ctx, memory, m)
            q, gates = cfg['q'], cfg['gates']
            alpha = gates[:, 0].view(b, 1, 1)
            beta = gates[:, 1].view(b, 1, 1)
            gamma = gates[:, 2].view(b, 1, 1)
            eta = gates[:, 3].view(b, 1, 1)
            rho = gates[:, 4].view(b, 1, 1)
            mixed = self.bank.mix(phi, q)
            update = torch.tanh(
                gamma * self.state_proj(mixed)
                + beta * self.input_proj(ctx[:, None, :].expand(b, l, d))
                + eta * self.init_proj(init)
                + rho * self.memory_proj(memory[:, None, :].expand(b, l, d))
            )
            old = phi
            phi = self.norm((1.0 - alpha) * phi + alpha * self.dropout(update))
            memory = cfg['memory']
            q_hist.append(q)
            gate_hist.append(gates.mean(0))
            deltas.append((phi - old).float().pow(2).mean((1, 2)).sqrt())
            ents.append(ent)
            if self.aux_projector is not None:
                aux.append(self.aux_projector(self.features(phi)))
        if not return_stats:
            return phi
        stats = {
            'q_history': torch.stack(q_hist, dim=1),
            'gate_history': torch.stack(gate_hist, dim=0),
            'deltas': torch.stack(deltas, dim=1),
            'attn_entropy': torch.stack(ents).mean(),
            'program': self.bank.replay_description(torch.stack(q_hist, dim=1).reshape(-1, self.config.n_modes)),
        }
        if aux:
            stats['step_logits'] = torch.stack(aux, dim=1)
        return phi, stats


class ParallelResonantBlock(nn.Module):
    """Safe adapter: y = base(x) + sigmoid(gate) * (ResonantBlock(norm(base)) - norm(base))."""
    def __init__(self, config: ResonantBlockConfig, base_block: Optional[nn.Module] = None):
        super().__init__()
        self.base_block = base_block
        self.pre_norm = nn.LayerNorm(config.dim)
        self.resonant = ResonantBlock(config)
        self.branch_gate = nn.Parameter(torch.tensor(float(config.gate_init)))

    def forward(self, x, context: Optional[torch.Tensor] = None, return_stats: bool = False):
        base = self.base_block(x) if self.base_block is not None else x
        z = self.pre_norm(base)
        if return_stats:
            res, stats = self.resonant(z, context=context, return_stats=True)
        else:
            res, stats = self.resonant(z, context=context, return_stats=False), None
        gate = torch.sigmoid(self.branch_gate)
        y = base + gate * (res - z)
        if not return_stats:
            return y
        stats['parallel_gate'] = gate.detach()
        return y, stats


class TinyParallelResonantClassifier(nn.Module):
    def __init__(self, input_dim: int, dim: int, n_classes: int, n_modes: int = 8, micro_steps: int = 3):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, dim)
        cfg = ResonantBlockConfig(dim=dim, n_modes=n_modes, micro_steps=micro_steps, aux_classes=n_classes)
        self.block = ParallelResonantBlock(cfg)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.head = nn.Linear(dim * 4, n_classes)

    def forward(self, x, return_stats: bool = False):
        h = self.in_proj(x)
        if return_stats:
            h, stats = self.block(h, return_stats=True)
        else:
            h, stats = self.block(h), {}
        h = h + 0.5 * self.ff(h)
        logits = self.head(ResonantBlock.features(h))
        return (logits, stats) if return_stats else logits
