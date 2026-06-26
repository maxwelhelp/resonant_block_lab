from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Replayable sequence operator bank for dynamic length [B,L,D].

    The first modes are readable fixed operators. Optional matrix-program families
    can occupy the next free modes:
      - symbolic_additive: W = sum_i c_i O_i
      - symbolic_product: W = prod_s (I + step_s * sum_i c_si O_i)
      - low_rank_global: cheap global learned low-rank summary
      - input_conditioned: primitive generated directly from current input statistics

    This keeps the controller idea intact: the controller routes over modes, but
    some modes are themselves small matrix-program compilers.
    """
    def __init__(self, dim: int, n_modes: int, *,
                 enable_symbolic_additive: bool = False,
                 enable_symbolic_product: bool = False,
                 enable_lowrank: bool = False,
                 enable_input_primitive: bool = False,
                 enable_token_primitives: bool = False,
                 enable_hand_token_primitives: bool = False,
                 program_steps: int = 2,
                 program_rank: int = 8):
        super().__init__()
        self.dim = int(dim)
        self.n_modes = int(n_modes)
        self.n_fixed_modes = 8 if bool(enable_hand_token_primitives) else (4 if bool(enable_token_primitives) else 6)
        self.program_steps = int(program_steps)
        self.program_rank = int(program_rank)
        self.enable_symbolic_additive = bool(enable_symbolic_additive)
        self.enable_symbolic_product = bool(enable_symbolic_product)
        self.enable_lowrank = bool(enable_lowrank)
        self.enable_input_primitive = bool(enable_input_primitive)
        self.enable_token_primitives = bool(enable_token_primitives)
        self.enable_hand_token_primitives = bool(enable_hand_token_primitives)

        if self.enable_hand_token_primitives:
            names = ['identity', 'prev1', 'prev2', 'prev4', 'causal_avg3', 'prefix_mean', 'delta_prev', 'global_mean']
        elif self.enable_token_primitives:
            names = ['identity', 'learned_causal_kernel', 'learned_causal_pool', 'delta_causal']
        else:
            names = ['identity', 'shift_left', 'shift_right', 'local_avg', 'global_mean', 'highpass']
        if self.enable_symbolic_additive and len(names) < self.n_modes:
            names.append('symbolic_additive')
        if self.enable_symbolic_product and len(names) < self.n_modes:
            names.append('symbolic_product')
        if self.enable_lowrank and len(names) < self.n_modes:
            names.append('low_rank_global')
        if self.enable_input_primitive and len(names) < self.n_modes:
            names.append('input_conditioned')
        learned_count = 0
        null_count = 0
        while len(names) < self.n_modes:
            if self.enable_token_primitives and not self.enable_hand_token_primitives:
                # No hidden convolutional/super-primitive in token discovery mode.
                # Extra capacity is explicit no-op/null slots so the controller must
                # solve the task with causal kernel/pool/delta + costed memory.
                null_count += 1
                names.append(f'null_{null_count}')
            else:
                learned_count += 1
                names.append(f'learned_depthwise_{learned_count}')
        self.mode_names = names[:self.n_modes]
        self.learned_indices = [i for i,n in enumerate(self.mode_names) if n.startswith('learned_depthwise_')]
        self.dw = nn.ModuleList([nn.Conv1d(dim, dim, 3, padding=1, groups=dim, bias=False) for _ in self.learned_indices])

        self.mode_scale = nn.Parameter(torch.ones(n_modes))
        self.mode_bias = nn.Parameter(torch.zeros(n_modes, 1, dim))
        self.mode_logit_bias = nn.Parameter(torch.zeros(n_modes))

        # Matrix-program parameters. They are only used if their mode exists.
        self.symbolic_additive_logits = nn.Parameter(torch.zeros(self.n_fixed_modes))
        self.symbolic_product_logits = nn.Parameter(torch.zeros(self.program_steps, self.n_fixed_modes))
        self.symbolic_product_step = nn.Parameter(torch.full((self.program_steps,), 0.12))
        r = max(1, min(self.program_rank, dim))
        self.low_down = nn.Linear(dim, r, bias=False)
        self.low_up = nn.Linear(r, dim, bias=False)
        self.input_gate = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim), nn.Sigmoid())
        # Discovery token operators: not hand prev1/prefix. They learn offset/decay distribution.
        self.causal_offsets = [0, 1, 2, 4, 8, 16, 32]
        self.causal_kernel_logits = nn.Parameter(torch.zeros(len(self.causal_offsets)))
        self.causal_pool_logits = nn.Parameter(torch.zeros(6))
        self.causal_pool_scales = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

        for conv in self.dw:
            nn.init.zeros_(conv.weight)
            with torch.no_grad():
                conv.weight[:, 0, 1] = 1.0
        with torch.no_grad():
            # Start product programs close to identity for stability.
            self.symbolic_additive_logits[0] = 2.0
            self.symbolic_product_logits[:, 0] = 1.0

    @staticmethod
    def shift_left(x):
        return torch.cat([x[:, 1:, :], x[:, -1:, :]], dim=1)

    @staticmethod
    def shift_right(x):
        return torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)

    @staticmethod
    def local_avg(x):
        return (DynamicOperatorBank1D.shift_right(x) + x + DynamicOperatorBank1D.shift_left(x)) / 3.0

    @staticmethod
    def prev_k(x, k: int):
        if k <= 0:
            return x
        return torch.cat([x[:, :1, :].expand(-1, k, -1), x[:, :-k, :]], dim=1) if x.shape[1] > k else x[:, :1, :].expand_as(x)

    @staticmethod
    def causal_avg3(x):
        return (x + DynamicOperatorBank1D.prev_k(x, 1) + DynamicOperatorBank1D.prev_k(x, 2)) / 3.0

    @staticmethod
    def prefix_mean(x):
        c = x.cumsum(dim=1)
        den = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        return c / den

    def learned_causal_kernel(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.causal_kernel_logits, dim=-1).to(dtype=x.dtype, device=x.device)
        out = torch.zeros_like(x)
        for w, k in zip(weights, self.causal_offsets):
            out = out + w * self.prev_k(x, int(k))
        return out

    def learned_causal_pool(self, x: torch.Tensor) -> torch.Tensor:
        # Fast vectorized causal window pool. It still learns the scale, but avoids
        # Python loops over sequence positions. Complexity is O(num_scales * BLD),
        # not O(num_scales * L Python tensor ops).
        mix = torch.softmax(self.causal_pool_logits, dim=-1).to(dtype=x.dtype, device=x.device)
        b, l, d = x.shape
        cs = torch.cat([torch.zeros(b, 1, d, device=x.device, dtype=x.dtype), x.cumsum(dim=1)], dim=1)
        idx = torch.arange(l, device=x.device)
        out = torch.zeros_like(x)
        for w, scale in zip(mix, self.causal_pool_scales):
            win = max(1, int(scale))
            start = (idx + 1 - win).clamp_min(0)
            end = idx + 1
            sums = cs[:, end, :] - cs[:, start, :]
            denom = (end - start).to(dtype=x.dtype).view(1, l, 1).clamp_min(1.0)
            out = out + w * (sums / denom)
        return out

    @staticmethod
    def _features(x):
        return torch.cat([x.mean(1), x.std(1), x.max(1).values, x.pow(2).mean(1)], dim=-1)

    def base_ops(self, x: torch.Tensor):
        if self.enable_hand_token_primitives:
            p1 = self.prev_k(x, 1)
            return [x, p1, self.prev_k(x, 2), self.prev_k(x, 4), self.causal_avg3(x), self.prefix_mean(x), x - p1, x.mean(dim=1, keepdim=True).expand_as(x)]
        if self.enable_token_primitives:
            k = self.learned_causal_kernel(x)
            return [x, k, self.learned_causal_pool(x), x - k]
        return [
            x,
            self.shift_left(x),
            self.shift_right(x),
            self.local_avg(x),
            x.mean(dim=1, keepdim=True).expand_as(x),
            x - self.local_avg(x),
        ]

    def symbolic_additive(self, x: torch.Tensor) -> torch.Tensor:
        ops = self.base_ops(x)
        coeff = torch.softmax(self.symbolic_additive_logits, dim=-1)
        out = torch.zeros_like(x)
        for c, op in zip(coeff, ops):
            out = out + c * op
        return out

    def symbolic_product(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for s in range(self.program_steps):
            ops = self.base_ops(h)
            coeff = torch.softmax(self.symbolic_product_logits[s], dim=-1)
            delta = torch.zeros_like(h)
            for c, op in zip(coeff, ops):
                delta = delta + c * op
            step = 0.35 * torch.tanh(self.symbolic_product_step[s])
            h = h + step * delta
        return h

    def low_rank_global(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_token_primitives and not self.enable_hand_token_primitives:
            return self.low_up(self.low_down(self.learned_causal_pool(x)))
        g = self.low_up(self.low_down(x.mean(dim=1))).unsqueeze(1)
        return g.expand_as(x)

    def input_conditioned(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_token_primitives and not self.enable_hand_token_primitives:
            k = self.learned_causal_kernel(x)
            local = self.learned_causal_pool(x)
            high = x - k
            # Per-position causal gate. No sequence-wide mean/max/std here, because
            # that would leak future token information in LM mode.
            gate = self.input_gate(torch.cat([x, k, local, high], dim=-1))
            return gate * local + (1.0 - gate) * high
        gate = self.input_gate(self._features(x)).unsqueeze(1)
        local = self.local_avg(x)
        high = x - local
        return gate * local + (1.0 - gate) * high

    def apply_mode(self, x: torch.Tensor, mode_idx: int) -> torch.Tensor:
        name = self.mode_names[int(mode_idx)]
        if name == 'identity': return x
        if name == 'prev1': return self.prev_k(x, 1)
        if name == 'prev2': return self.prev_k(x, 2)
        if name == 'prev4': return self.prev_k(x, 4)
        if name == 'causal_avg3': return self.causal_avg3(x)
        if name == 'prefix_mean': return self.prefix_mean(x)
        if name == 'delta_prev': return x - self.prev_k(x, 1)
        if name == 'learned_causal_kernel': return self.learned_causal_kernel(x)
        if name == 'learned_causal_pool': return self.learned_causal_pool(x)
        if name == 'delta_causal': return x - self.learned_causal_kernel(x)
        if name == 'shift_left': return self.shift_left(x)
        if name == 'shift_right': return self.shift_right(x)
        if name == 'local_avg': return self.local_avg(x)
        if name == 'global_mean': return x.mean(dim=1, keepdim=True).expand_as(x)
        if name == 'highpass': return x - self.local_avg(x)
        if name == 'symbolic_additive': return self.symbolic_additive(x)
        if name == 'symbolic_product': return self.symbolic_product(x)
        if name == 'low_rank_global': return self.low_rank_global(x)
        if name == 'input_conditioned': return self.input_conditioned(x)
        if name.startswith('learned_depthwise_'):
            j = self.learned_indices.index(int(mode_idx))
            return self.dw[j](x.transpose(1, 2)).transpose(1, 2)
        if name.startswith('null_'):
            return torch.zeros_like(x)
        return x

    def apply_all(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.apply_mode(x, i) for i in range(self.n_modes)], dim=1)

    def route(self, q: torch.Tensor) -> torch.Tensor:
        q = q.clamp_min(1e-8)
        return torch.softmax(q.log() + self.mode_logit_bias.view(1, -1), dim=-1)

    def _postprocess_ops(self, ops: torch.Tensor) -> torch.Tensor:
        return ops * self.mode_scale.view(1, -1, 1, 1) + self.mode_bias.view(1, self.n_modes, 1, self.dim)

    def mix(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        q = self.route(q)
        ops = self._postprocess_ops(self.apply_all(x))
        return torch.einsum('br,brld->bld', q, ops)

    def mix_topk(self, x: torch.Tensor, q: torch.Tensor, topk: int = 3) -> torch.Tensor:
        q = self.route(q)
        k = min(int(topk), self.n_modes)
        if k <= 0 or k >= self.n_modes:
            return self.mix(x, q)
        vals, idx = q.topk(k, dim=-1)
        vals = vals / vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        b = x.shape[0]
        out = torch.zeros_like(x)
        # Lazy execution: compute only modes selected by at least one sample.
        for j in range(k):
            for mode in idx[:, j].unique().tolist():
                mask = (idx[:, j] == int(mode))
                if not bool(mask.any()):
                    continue
                op = self.apply_mode(x, int(mode))
                op = op * self.mode_scale[int(mode)] + self.mode_bias[int(mode)].view(1, 1, self.dim)
                out[mask] = out[mask] + vals[mask, j].view(-1, 1, 1) * op[mask]
        return out

    def program_formulas(self) -> Dict:
        if self.enable_hand_token_primitives:
            base = ['identity', 'prev1', 'prev2', 'prev4', 'causal_avg3', 'prefix_mean', 'delta_prev', 'global_mean']
        elif self.enable_token_primitives:
            base = ['identity', 'learned_causal_kernel', 'learned_causal_pool', 'delta_causal']
        else:
            base = ['identity', 'shift_left', 'shift_right', 'local_avg', 'global_mean', 'highpass']
        coeff = torch.softmax(self.symbolic_additive_logits.detach().float(), dim=-1).cpu()
        prod = torch.softmax(self.symbolic_product_logits.detach().float(), dim=-1).cpu()
        steps = torch.tanh(self.symbolic_product_step.detach().float()).cpu() * 0.35
        return {
            'mode_names': list(self.mode_names),
            'symbolic_additive': [(base[i], float(coeff[i])) for i in range(len(base))],
            'symbolic_product': [
                {'step': s, 'step_scale': float(steps[s]), 'ops': [(base[i], float(prod[s, i])) for i in range(len(base))]}
                for s in range(self.program_steps)
            ],
            'program_rank': self.program_rank,
            'token_primitives': self.enable_token_primitives,
            'hand_token_primitives': self.enable_hand_token_primitives,
            'causal_kernel_offsets': list(self.causal_offsets),
            'causal_kernel_weights': [float(v) for v in torch.softmax(self.causal_kernel_logits.detach().float(), dim=-1).cpu()],
            'causal_pool_scales': list(self.causal_pool_scales),
            'causal_pool_weights': [float(v) for v in torch.softmax(self.causal_pool_logits.detach().float(), dim=-1).cpu()],
        }

    def replay_description(self, q: torch.Tensor, topk: int = 4) -> Dict:
        q_mean = q.detach().float().mean(dim=0).cpu()
        vals, idx = q_mean.topk(min(topk, self.n_modes))
        return {
            'top_modes': [(self.mode_names[int(i)], float(v)) for v, i in zip(vals, idx)],
            'all_weights': {self.mode_names[i]: float(q_mean[i]) for i in range(self.n_modes)},
            'matrix_programs': self.program_formulas(),
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
