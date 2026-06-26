from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resonant_block import CrossReader, DynamicOperatorBank1D


@dataclass
class FractalResonantConfig:
    dim: int
    n_modes: int = 8
    macro_steps: int = 8
    micro_steps: int = 3
    controller_hidden: int = 192
    aux_classes: int = 0
    dropout: float = 0.0
    max_macro_steps: int = 64
    max_micro_steps: int = 16
    use_ff_refine: bool = False
    mix_topk: int = 0
    enable_matrix_program: bool = False
    enable_symbolic_product: bool = False
    enable_lowrank: bool = False
    enable_input_primitive: bool = False
    program_steps: int = 2
    program_rank: int = 8


class UnifiedFractalController(nn.Module):
    """One controller law for macro and micro levels.

    It receives level/macro/micro embeddings, so macro/micro behavior can specialize,
    but weights are shared. This avoids two disconnected controllers.
    """
    def __init__(self, dim: int, n_modes: int, hidden: int = 192, max_macro: int = 64, max_micro: int = 16):
        super().__init__()
        self.dim = int(dim)
        self.n_modes = int(n_modes)
        emb = 32
        self.level_emb = nn.Embedding(2, emb)
        self.macro_emb = nn.Embedding(max_macro, emb)
        self.micro_emb = nn.Embedding(max_micro, emb)
        self.in_norm = nn.LayerNorm(dim * 3 + emb * 3)
        self.net = nn.Sequential(
            nn.Linear(dim * 3 + emb * 3, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_modes * 2 + 6 + dim),
        )

    def forward(self, field_pool: torch.Tensor, context: torch.Tensor, memory: torch.Tensor,
                macro_idx: int, micro_idx: int, level_id: int) -> Dict[str, torch.Tensor]:
        b = field_pool.shape[0]
        lvl = self.level_emb(torch.full((b,), level_id, device=field_pool.device, dtype=torch.long))
        me = self.macro_emb(torch.full((b,), min(macro_idx, self.macro_emb.num_embeddings - 1), device=field_pool.device, dtype=torch.long))
        ue = self.micro_emb(torch.full((b,), min(micro_idx, self.micro_emb.num_embeddings - 1), device=field_pool.device, dtype=torch.long))
        h = torch.cat([field_pool, context, memory, lvl.to(field_pool.dtype), me.to(field_pool.dtype), ue.to(field_pool.dtype)], dim=-1)
        y = self.net(self.in_norm(h))
        a = 0
        macro_q = torch.softmax(y[:, a:a+self.n_modes], dim=-1); a += self.n_modes
        micro_q = torch.softmax(y[:, a:a+self.n_modes], dim=-1); a += self.n_modes
        gates = torch.sigmoid(y[:, a:a+6]); a += 6
        new_memory = torch.tanh(y[:, a:a+self.dim])
        return {
            'macro_q': macro_q,
            'micro_q': micro_q,
            'gates': gates,       # alpha,beta,gamma,eta,rho,macro_mix
            'memory': new_memory,
        }


class FractalResonantSequenceBlock(nn.Module):
    """Macro/micro resonant sequence block.

    Shape: [B,L,D] -> [B,L,D]

    This is the sequence analogue of the audio v4 field:
      macro step t:
        macro controller makes plan/memory
        micro loop executes/refines using micro controller calls
      affine projectors read every macro and micro state for direct gradients.
    """
    def __init__(self, config: FractalResonantConfig):
        super().__init__()
        self.config = config
        d = config.dim
        self.reader = CrossReader(d)
        self.bank = DynamicOperatorBank1D(
            d, config.n_modes,
            enable_symbolic_additive=config.enable_matrix_program,
            enable_symbolic_product=config.enable_symbolic_product,
            enable_lowrank=config.enable_lowrank,
            enable_input_primitive=config.enable_input_primitive,
            program_steps=config.program_steps,
            program_rank=config.program_rank,
        )
        self.controller = UnifiedFractalController(
            d, config.n_modes, config.controller_hidden, config.max_macro_steps, config.max_micro_steps
        )
        self.state_proj = nn.Linear(d, d, bias=False)
        self.input_proj = nn.Linear(d, d, bias=True)
        self.init_proj = nn.Linear(d, d, bias=False)
        self.memory_proj = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)
        self.affine_projector = nn.Linear(d * 4, config.aux_classes) if config.aux_classes > 0 else None
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d)) if config.use_ff_refine else None

    @staticmethod
    def features(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(1), x.std(1), x.max(1).values, x.pow(2).mean(1)], dim=-1)

    def _update(self, phi: torch.Tensor, init: torch.Tensor, ctx: torch.Tensor, memory: torch.Tensor,
                macro_q: torch.Tensor, micro_q: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        b, l, d = phi.shape
        alpha = gates[:, 0].view(b, 1, 1)
        beta = gates[:, 1].view(b, 1, 1)
        gamma = gates[:, 2].view(b, 1, 1)
        eta = gates[:, 3].view(b, 1, 1)
        rho = gates[:, 4].view(b, 1, 1)
        macro_mix = gates[:, 5].view(b, 1)
        q_eff = F.normalize(macro_mix * macro_q + (1.0 - macro_mix) * micro_q, p=1, dim=-1)
        mixed = self.bank.mix_topk(phi, q_eff, self.config.mix_topk) if self.config.mix_topk else self.bank.mix(phi, q_eff)
        update = torch.tanh(
            gamma * self.state_proj(mixed)
            + beta * self.input_proj(ctx[:, None, :].expand(b, l, d))
            + eta * self.init_proj(init)
            + rho * self.memory_proj(memory[:, None, :].expand(b, l, d))
        )
        out = self.norm((1.0 - alpha) * phi + alpha * self.dropout(update))
        if self.ff is not None:
            out = out + 0.25 * self.ff(out)
        return out

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, return_stats: bool = False):
        if context is None:
            context = x
        psi = x
        init = x
        b, l, d = x.shape
        ctx_keys = self.reader.k(context)
        ctx_values = self.reader.v(context)
        macro_memory_state = torch.zeros(b, d, device=x.device, dtype=x.dtype)

        macro_states, micro_states = [], []
        macro_logits, micro_logits = [], []
        macro_plan_q_hist, micro_macro_q_hist, micro_q_hist, q_eff_hist, gate_hist = [], [], [], [], []
        macro_delta_hist, micro_delta_hist, ent_hist = [], [], []

        for t in range(self.config.macro_steps):
            macro_ctx, macro_ent = self.reader.read_precomputed(psi, ctx_keys, ctx_values)
            macro_cfg = self.controller(psi.mean(1), macro_ctx, macro_memory_state, t, 0, level_id=0)
            macro_memory_state = macro_cfg['memory']
            macro_q = macro_cfg['macro_q']
            macro_plan_q_hist.append(macro_q)
            micro_memory = macro_memory_state
            old_macro = psi
            phi = psi

            for m in range(self.config.micro_steps):
                ctx, ent = self.reader.read_precomputed(phi, ctx_keys, ctx_values)
                cfg = self.controller(phi.mean(1), ctx, micro_memory, t, m, level_id=1)
                old = phi
                gates = cfg['gates']
                micro_q = cfg['micro_q']
                macro_q_for_update = macro_q
                phi = self._update(phi, init, ctx, macro_memory_state, macro_q_for_update, micro_q, gates)
                micro_memory = cfg['memory']

                if self.affine_projector is not None:
                    micro_logits.append(self.affine_projector(self.features(phi)))
                micro_states.append(phi)
                micro_macro_q_hist.append(macro_q_for_update)
                micro_q_hist.append(micro_q)
                macro_mix = gates[:, 5].view(b, 1)
                q_eff_hist.append(F.normalize(macro_mix * macro_q_for_update + (1.0 - macro_mix) * micro_q, p=1, dim=-1))
                gate_hist.append(gates.mean(0))
                micro_delta_hist.append((phi - old).float().pow(2).mean((1, 2)).sqrt())
                ent_hist.append(ent)

            psi = phi
            if self.affine_projector is not None:
                macro_logits.append(self.affine_projector(self.features(psi)))
            macro_states.append(psi)
            macro_delta_hist.append((psi - old_macro).float().pow(2).mean((1, 2)).sqrt())
            ent_hist.append(macro_ent)

        y = macro_states[-1]
        if not return_stats:
            return y

        q_eff_all = torch.stack(q_eff_hist, dim=1) if q_eff_hist else torch.empty(0, device=x.device)
        macro_q_all = torch.stack(macro_plan_q_hist, dim=1)
        micro_macro_q_all = torch.stack(micro_macro_q_hist, dim=1)
        micro_q_all = torch.stack(micro_q_hist, dim=1)
        stats = {
            'macro_history': torch.stack(macro_states, dim=1),
            'micro_history': torch.stack(micro_states, dim=1),
            # Audio-v4 compatible: macro_q_history is one entry per macro step.
            'macro_q_history': macro_q_all,
            # Macro-q actually used inside every micro update.
            'micro_macro_q_history': micro_macro_q_all,
            'micro_q_history': micro_q_all,
            'q_eff_history': q_eff_all,
            'gate_history': torch.stack(gate_hist, dim=0),
            'macro_deltas': torch.stack(macro_delta_hist, dim=1),
            'micro_deltas': torch.stack(micro_delta_hist, dim=1),
            'macro_mode_entropy': -(macro_q_all.clamp_min(1e-8).log() * macro_q_all).sum(dim=-1).mean(),
            'micro_mode_entropy': -(micro_q_all.clamp_min(1e-8).log() * micro_q_all).sum(dim=-1).mean(),
            'eff_mode_entropy': -(q_eff_all.clamp_min(1e-8).log() * q_eff_all).sum(dim=-1).mean(),
            'psi_final': y,
            'field_energy': y.pow(2).mean(),
            'attn_entropy': torch.stack(ent_hist).mean(),
            'program': self.bank.replay_description(q_eff_all.reshape(-1, self.config.n_modes)),
        }
        if macro_logits:
            stats['macro_logits'] = torch.stack(macro_logits, dim=1)
        if micro_logits:
            stats['micro_logits'] = torch.stack(micro_logits, dim=1)
        return y, stats


class SequenceAttractorHead(nn.Module):
    """Feature-space attractor head for dynamic sequence length.

    Audio v4 keeps attractors in [nodes,dim]. Sequence blocks have dynamic L,
    so the attractor lives in pooled field features [mean,std,max,energy].
    """
    def __init__(self, n_classes: int, dim: int, temperature: float = 0.35, head_mode: str = 'attractor_only'):
        super().__init__()
        self.n_classes = int(n_classes)
        self.dim = int(dim)
        self.head_mode = str(head_mode)
        self.attractors = nn.Parameter(torch.randn(n_classes, dim * 4) * 0.02)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))
        self.linear_aux = nn.Linear(dim * 4, n_classes)
        self.mix = nn.Parameter(torch.tensor(0.15))

    def energy_logits(self, feats: torch.Tensor):
        z = F.normalize(feats, dim=-1)
        a = F.normalize(self.attractors, dim=-1)
        cosine = z @ a.T
        energy = 1.0 - cosine
        temp = self.log_temperature.exp().clamp(0.05, 5.0)
        return cosine / temp, energy, energy.min(dim=1).values

    def forward(self, feats: torch.Tensor):
        attr_logits, energy, min_energy = self.energy_logits(feats)
        linear_logits = self.linear_aux(feats)
        if self.head_mode == 'linear_only':
            logits = linear_logits
        elif self.head_mode == 'hybrid':
            logits = attr_logits + self.mix.clamp(0.0, 2.0) * linear_logits
        else:
            logits = attr_logits
        return logits, attr_logits, linear_logits, energy, min_energy

    def separation_loss(self, min_dist: float = 0.08) -> torch.Tensor:
        a = F.normalize(self.attractors, dim=-1)
        energy = 1.0 - (a @ a.T)
        mask = ~torch.eye(self.n_classes, device=energy.device, dtype=torch.bool)
        return F.relu(float(min_dist) - energy[mask]).mean()


class FractalResonantSequenceClassifier(nn.Module):
    """Resonant-only classifier: no parallel fallback as a hidden crutch."""
    def __init__(self, input_dim: int, dim: int, n_classes: int,
                 n_modes: int = 8, macro_steps: int = 8, micro_steps: int = 3,
                 use_ff_refine: bool = False, head_mode: str = 'attractor_only', mix_topk: int = 0,
                 enable_matrix_program: bool = False, enable_symbolic_product: bool = False,
                 enable_lowrank: bool = False, enable_input_primitive: bool = False,
                 program_steps: int = 2, program_rank: int = 8):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, dim)
        cfg = FractalResonantConfig(
            dim=dim,
            n_modes=n_modes,
            macro_steps=macro_steps,
            micro_steps=micro_steps,
            aux_classes=n_classes,
            use_ff_refine=use_ff_refine,
            mix_topk=int(mix_topk),
            enable_matrix_program=bool(enable_matrix_program),
            enable_symbolic_product=bool(enable_symbolic_product),
            enable_lowrank=bool(enable_lowrank),
            enable_input_primitive=bool(enable_input_primitive),
            program_steps=int(program_steps),
            program_rank=int(program_rank),
        )
        self.block = FractalResonantSequenceBlock(cfg)
        self.head = SequenceAttractorHead(n_classes, dim, head_mode=head_mode)

    def forward(self, x: torch.Tensor, return_stats: bool = False):
        h = self.in_proj(x)
        if return_stats:
            h, stats = self.block(h, return_stats=True)
        else:
            h = self.block(h, return_stats=False)
            stats = {}
        feats = FractalResonantSequenceBlock.features(h)
        logits, attr_logits, linear_logits, energy, min_energy = self.head(feats)
        if return_stats:
            stats.update({
                'attr_logits': attr_logits,
                'linear_logits': linear_logits,
                'attractor_energy': energy,
                'min_energy': min_energy,
                'field_energy': h.pow(2).mean(),
                'final_features': feats,
            })
        return (logits, stats) if return_stats else logits
