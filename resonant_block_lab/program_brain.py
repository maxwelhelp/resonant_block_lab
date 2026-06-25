from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _tolist(x: torch.Tensor, ndigits: int = 6):
    return x.detach().float().cpu().numpy().round(ndigits).tolist()


class ProgramBrain:
    """Diagnostic-only brain for resonant programs.

    It does not change forward/training. It aggregates validation batches and writes:
      - program_summary.json
      - PROGRAM_DYNAMICS.md
      - best_program.json when val_acc improves

    Credit here is a cheap proxy, not a true counterfactual ablation:
      mode_credit_proxy[r] = mean q_eff[r] on correct examples - mean q_eff[r] on wrong examples
    Pair compatibility is the same signed proxy for co-activation pairs.
    """
    def __init__(self, mode_names: Optional[list[str]] = None):
        self.mode_names = mode_names
        self.n = 0
        self.final_correct = 0.0
        self._macro_acc = None
        self._micro_acc = None
        self._macro_q = None
        self._micro_macro_q = None
        self._micro_q = None
        self._q_eff = None
        self._gates = None
        self._macro_delta = None
        self._micro_delta = None
        self._credit_correct_num = None
        self._credit_correct_den = None
        self._credit_wrong_num = None
        self._credit_wrong_den = None
        self._pair_correct_num = None
        self._pair_correct_den = None
        self._pair_wrong_num = None
        self._pair_wrong_den = None
        self.last_program = None
        self.scalar_sums: Dict[str, float] = {}
        self.scalar_count: Dict[str, int] = {}

    def _add_mean(self, key: str, value: torch.Tensor):
        self.scalar_sums[key] = self.scalar_sums.get(key, 0.0) + float(value.detach().float().cpu())
        self.scalar_count[key] = self.scalar_count.get(key, 0) + 1

    def _sum_batch_matrix(self, current, tensor: torch.Tensor, reduce_batch: bool = True):
        t = tensor.detach().float().cpu()
        if reduce_batch:
            t = t.sum(dim=0)
        return t if current is None else current + t

    def add(self, stats: Dict[str, Any], y: torch.Tensor, logits: torch.Tensor):
        y_cpu = y.detach().cpu()
        pred = logits.detach().cpu().argmax(dim=-1)
        correct = (pred == y_cpu).float()
        b = int(y_cpu.numel())
        self.n += b
        self.final_correct += float(correct.sum())

        if 'program' in stats:
            self.last_program = stats['program']
        for k in ['macro_mode_entropy', 'micro_mode_entropy', 'eff_mode_entropy', 'attn_entropy', 'field_energy']:
            if k in stats:
                self._add_mean(k, stats[k])

        if 'macro_logits' in stats:
            m = stats['macro_logits'].detach().cpu().argmax(dim=-1)
            self._macro_acc = self._sum_batch_matrix(self._macro_acc, (m == y_cpu[:, None]).float())
        if 'micro_logits' in stats:
            u = stats['micro_logits'].detach().cpu().argmax(dim=-1)
            self._micro_acc = self._sum_batch_matrix(self._micro_acc, (u == y_cpu[:, None]).float())
        if 'macro_q_history' in stats:
            self._macro_q = self._sum_batch_matrix(self._macro_q, stats['macro_q_history'])
        if 'micro_macro_q_history' in stats:
            self._micro_macro_q = self._sum_batch_matrix(self._micro_macro_q, stats['micro_macro_q_history'])
        if 'micro_q_history' in stats:
            self._micro_q = self._sum_batch_matrix(self._micro_q, stats['micro_q_history'])
        if 'q_eff_history' in stats:
            q = stats['q_eff_history'].detach().float().cpu()  # [B,S,R]
            self._q_eff = self._sum_batch_matrix(self._q_eff, q)
            corr = correct.view(-1, 1, 1)
            wrong = 1.0 - corr
            cnum = (q * corr).sum(dim=(0, 1))
            wnum = (q * wrong).sum(dim=(0, 1))
            cden = (corr.sum() * q.shape[1]).clamp_min(1e-8).expand_as(cnum)
            wden = (wrong.sum() * q.shape[1]).clamp_min(1e-8).expand_as(wnum)
            self._credit_correct_num = cnum if self._credit_correct_num is None else self._credit_correct_num + cnum
            self._credit_correct_den = cden if self._credit_correct_den is None else self._credit_correct_den + cden
            self._credit_wrong_num = wnum if self._credit_wrong_num is None else self._credit_wrong_num + wnum
            self._credit_wrong_den = wden if self._credit_wrong_den is None else self._credit_wrong_den + wden

            q_mean = q.mean(dim=1)  # [B,R]
            pc = torch.einsum('bi,bj,b->ij', q_mean, q_mean, correct)
            pw = torch.einsum('bi,bj,b->ij', q_mean, q_mean, 1.0 - correct)
            pcden = correct.sum().clamp_min(1e-8)
            pwden = (1.0 - correct).sum().clamp_min(1e-8)
            self._pair_correct_num = pc if self._pair_correct_num is None else self._pair_correct_num + pc
            self._pair_correct_den = pcden if self._pair_correct_den is None else self._pair_correct_den + pcden
            self._pair_wrong_num = pw if self._pair_wrong_num is None else self._pair_wrong_num + pw
            self._pair_wrong_den = pwden if self._pair_wrong_den is None else self._pair_wrong_den + pwden
        if 'gate_history' in stats:
            self._gates = self._sum_batch_matrix(self._gates, stats['gate_history'], reduce_batch=False)
            self.scalar_count['gate_history'] = self.scalar_count.get('gate_history', 0) + 1
        if 'macro_deltas' in stats:
            self._macro_delta = self._sum_batch_matrix(self._macro_delta, stats['macro_deltas'])
        if 'micro_deltas' in stats:
            self._micro_delta = self._sum_batch_matrix(self._micro_delta, stats['micro_deltas'])

    def finalize(self, epoch: int, row: Dict[str, Any]) -> Dict[str, Any]:
        n = max(1, self.n)
        out: Dict[str, Any] = {
            'epoch': epoch,
            'n': self.n,
            'final_acc': self.final_correct / n,
            'row': {k: v for k, v in row.items() if k != 'program'},
            'program': self.last_program,
        }
        for k, v in self.scalar_sums.items():
            out[k] = v / max(1, self.scalar_count.get(k, 1))
        if self._macro_acc is not None:
            out['macro_acc_by_step'] = _tolist(self._macro_acc / n, 5)
        if self._micro_acc is not None:
            out['micro_acc_by_microstep'] = _tolist(self._micro_acc / n, 5)
        if self._macro_q is not None:
            out['macro_mode_schedule_mean_by_step'] = _tolist(self._macro_q / n, 5)
        if self._micro_macro_q is not None:
            out['micro_macro_mode_schedule_mean_by_microstep'] = _tolist(self._micro_macro_q / n, 5)
        if self._micro_q is not None:
            out['micro_mode_schedule_mean_by_microstep'] = _tolist(self._micro_q / n, 5)
        if self._q_eff is not None:
            out['effective_mode_schedule_mean_by_microstep'] = _tolist(self._q_eff / n, 5)
        if self._gates is not None:
            out['gate_schedule_alpha_beta_gamma_eta_rho_macroMix'] = _tolist(self._gates / max(1, self.scalar_count.get('gate_history', 1)), 5)
        if self._macro_delta is not None:
            out['macro_delta_mean_by_step'] = _tolist(self._macro_delta / n, 5)
        if self._micro_delta is not None:
            out['micro_delta_mean_by_microstep'] = _tolist(self._micro_delta / n, 5)
        if self._credit_correct_num is not None:
            cmean = self._credit_correct_num / self._credit_correct_den.clamp_min(1e-8)
            wmean = self._credit_wrong_num / self._credit_wrong_den.clamp_min(1e-8)
            credit = cmean - wmean
            out['mode_credit_proxy'] = self._named_vector(credit)
            out['mode_mean_on_correct'] = self._named_vector(cmean)
            out['mode_mean_on_wrong'] = self._named_vector(wmean)
            out['helpful_modes_proxy'] = self._top_named(credit, largest=True)
            out['harmful_modes_proxy'] = self._top_named(credit, largest=False)
        if self._pair_correct_num is not None:
            pc = self._pair_correct_num / self._pair_correct_den.clamp_min(1e-8)
            pw = self._pair_wrong_num / self._pair_wrong_den.clamp_min(1e-8)
            pair = pc - pw
            out['pair_compatibility_proxy_top'] = self._top_pairs(pair, largest=True)
            out['pair_conflict_proxy_top'] = self._top_pairs(pair, largest=False)
        return out

    def _names(self, r: int):
        if self.mode_names and len(self.mode_names) >= r:
            return self.mode_names[:r]
        base = ['identity','shift_left','shift_right','local_avg','global_mean','highpass']
        while len(base) < r:
            base.append(f'learned_depthwise_{len(base)-5}')
        return base[:r]

    def _named_vector(self, v: torch.Tensor):
        names = self._names(v.numel())
        return {names[i]: float(v[i]) for i in range(v.numel())}

    def _top_named(self, v: torch.Tensor, largest: bool, k: int = 5):
        names = self._names(v.numel())
        vals, idx = torch.topk(v, min(k, v.numel()), largest=largest)
        return [(names[int(i)], float(val)) for val, i in zip(vals, idx)]

    def _top_pairs(self, mat: torch.Tensor, largest: bool, k: int = 8):
        r = mat.shape[0]
        names = self._names(r)
        m = mat.clone()
        m.fill_diagonal_(0.0)
        vals, flat = torch.topk(m.reshape(-1), min(k, r*r), largest=largest)
        out = []
        for val, f in zip(vals, flat):
            i = int(f // r); j = int(f % r)
            if i != j:
                out.append((names[i], names[j], float(val)))
        return out[:k]


def write_program_brain_outputs(out_dir: Path, summary: Dict[str, Any], best_acc: float, row_acc: float):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'program_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    md = render_program_dynamics_md(summary, best_acc=best_acc, row_acc=row_acc)
    (out_dir / 'PROGRAM_DYNAMICS.md').write_text(md, encoding='utf-8')
    best_path = out_dir / 'best_program.json'
    prev = None
    if best_path.exists():
        try:
            prev = json.loads(best_path.read_text(encoding='utf-8'))
        except Exception:
            prev = None
    prev_acc = float(prev.get('final_acc', -1.0)) if isinstance(prev, dict) else -1.0
    if row_acc >= prev_acc:
        best_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')


def render_program_dynamics_md(s: Dict[str, Any], best_acc: float, row_acc: float) -> str:
    lines = []
    lines.append('# Program Dynamics')
    lines.append('')
    lines.append(f"epoch: {s.get('epoch')}  val_acc: {row_acc:.5f}  best_acc: {best_acc:.5f}")
    lines.append(f"n_eval: {s.get('n')}  final_acc_accum: {s.get('final_acc', 0):.5f}")
    lines.append('')
    prog = s.get('program') or {}
    if prog.get('top_modes'):
        lines.append('## Executed program top modes')
        for name, weight in prog['top_modes']:
            lines.append(f'- {name}: {weight:.5f}')
        lines.append('')
    if s.get('helpful_modes_proxy'):
        lines.append('## Mode credit proxy')
        lines.append('Helpful:')
        for name, val in s['helpful_modes_proxy']:
            lines.append(f'- {name}: {val:.5f}')
        lines.append('Harmful / suspicious:')
        for name, val in s.get('harmful_modes_proxy', []):
            lines.append(f'- {name}: {val:.5f}')
        lines.append('')
    if s.get('pair_compatibility_proxy_top'):
        lines.append('## Pair compatibility proxy')
        for a,b,val in s['pair_compatibility_proxy_top'][:6]:
            lines.append(f'- {a} + {b}: {val:.5f}')
        lines.append('')
    if s.get('pair_conflict_proxy_top'):
        lines.append('## Pair conflict proxy')
        for a,b,val in s['pair_conflict_proxy_top'][:6]:
            lines.append(f'- {a} + {b}: {val:.5f}')
        lines.append('')
    if s.get('macro_acc_by_step'):
        lines.append('## Macro accuracy by step')
        lines.append(str(s['macro_acc_by_step']))
        lines.append('')
    if s.get('micro_acc_by_microstep'):
        lines.append('## Micro accuracy by microstep')
        lines.append(str(s['micro_acc_by_microstep']))
        lines.append('')
    for k in ['eff_mode_entropy','macro_mode_entropy','micro_mode_entropy','attn_entropy','field_energy']:
        if k in s:
            lines.append(f'- {k}: {s[k]:.6f}')
    lines.append('')
    return '\n'.join(lines)
