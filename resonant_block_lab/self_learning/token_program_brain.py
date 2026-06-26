from __future__ import annotations
from typing import Any, Dict, List
import torch
import torch.nn.functional as F

IGNORE_INDEX = -100

def resolve_mode_names(stats: Dict[str, Any] | None = None, bank=None, fallback_n_modes: int | None = None) -> List[str]:
    if bank is not None and hasattr(bank, 'mode_names'):
        return list(bank.mode_names)
    stats = stats or {}
    program = stats.get('program', {}) if isinstance(stats, dict) else {}
    mp = program.get('matrix_programs', {}) if isinstance(program, dict) else {}
    if isinstance(mp, dict) and 'mode_names' in mp:
        return list(mp['mode_names'])
    if isinstance(stats, dict) and 'mode_names' in stats:
        return list(stats['mode_names'])
    assert fallback_n_modes is not None
    return [f'mode_{i}' for i in range(fallback_n_modes)]

class TokenProgramBrain:
    def __init__(self, mode_names: List[str] | None = None):
        self.mode_names = list(mode_names or [])
        self.total_tokens = 0
        self.correct_tokens = 0
        self.loss_sum = 0.0
        self.usage_sum = None
        self.usage_count = 0
        self.correct_usage_num = None
        self.correct_usage_den = None
        self.wrong_usage_num = None
        self.wrong_usage_den = None

    @torch.no_grad()
    def add_batch(self, stats: Dict[str, Any], logits: torch.Tensor, targets: torch.Tensor, loss_per_token: torch.Tensor | None = None):
        pred = logits.argmax(dim=-1)
        mask = targets.ne(IGNORE_INDEX)
        if not bool(mask.any()):
            return
        correct = pred.eq(targets) & mask
        n = int(mask.sum().item())
        self.total_tokens += n
        self.correct_tokens += int(correct.sum().item())
        if loss_per_token is None:
            flat = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=IGNORE_INDEX, reduction='none')
            loss_per_token = flat.reshape_as(targets)
        self.loss_sum += float(loss_per_token[mask].sum().detach().cpu())
        q = stats.get('q_eff_history', None)
        if q is None:
            q = stats.get('q_eff_hist', None)
        if q is None:
            q = stats.get('micro_q_history', None)
        if q is None:
            q = stats.get('micro_q_hist', None)
        if q is None:
            return
        if q.dim() == 3:
            q_sample = q.float().mean(dim=1)
        elif q.dim() == 2:
            q_sample = q.float()
        else:
            return
        _, r = q_sample.shape
        if not self.mode_names:
            self.mode_names = resolve_mode_names(stats, fallback_n_modes=r)
        token_counts = mask.float().sum(dim=1).clamp_min(1.0)
        corr_counts = correct.float().sum(dim=1)
        wrong_counts = token_counts - corr_counts
        if self.usage_sum is None:
            dev = q_sample.device
            self.usage_sum = torch.zeros(r, device=dev)
            self.correct_usage_num = torch.zeros(r, device=dev)
            self.correct_usage_den = torch.zeros(r, device=dev)
            self.wrong_usage_num = torch.zeros(r, device=dev)
            self.wrong_usage_den = torch.zeros(r, device=dev)
        self.usage_sum += (q_sample * token_counts[:, None]).sum(dim=0)
        self.usage_count += int(token_counts.sum().item())
        self.correct_usage_num += (q_sample * corr_counts[:, None]).sum(dim=0)
        self.correct_usage_den += corr_counts.sum().expand_as(self.correct_usage_den)
        self.wrong_usage_num += (q_sample * wrong_counts[:, None]).sum(dim=0)
        self.wrong_usage_den += wrong_counts.sum().expand_as(self.wrong_usage_den)

    def finalize(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            'token_acc': float(self.correct_tokens / max(1, self.total_tokens)),
            'token_loss_mean': float(self.loss_sum / max(1, self.total_tokens)),
            'token_count': int(self.total_tokens),
            'token_correct': int(self.correct_tokens),
        }
        if self.usage_sum is not None:
            usage = self.usage_sum / max(1, self.usage_count)
            c = self.correct_usage_num / self.correct_usage_den.clamp_min(1e-8)
            w = self.wrong_usage_num / self.wrong_usage_den.clamp_min(1e-8)
            credit = c - w
            names = self.mode_names or [f'mode_{i}' for i in range(int(usage.numel()))]
            out['token_mode_usage'] = {n: float(v.detach().cpu()) for n, v in zip(names, usage)}
            out['token_mode_credit_proxy'] = {n: float(v.detach().cpu()) for n, v in zip(names, credit)}
            out['token_helpful_modes_proxy'] = sorted(out['token_mode_credit_proxy'].items(), key=lambda kv: kv[1], reverse=True)[:8]
            out['token_harmful_modes_proxy'] = sorted(out['token_mode_credit_proxy'].items(), key=lambda kv: kv[1])[:8]
        return out
