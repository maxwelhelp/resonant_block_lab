from __future__ import annotations
from typing import Dict, Any
import torch

class ModeFeedbackMemory:
    def __init__(self, levels=('macro','micro','effective'), n_steps_by_level: Dict[str,int] | None = None, n_modes: int = 1, momentum: float = 0.98, device='cpu'):
        self.levels = tuple(levels)
        self.n_steps_by_level = dict(n_steps_by_level or {lvl: 1 for lvl in levels})
        self.n_modes = int(n_modes)
        self.momentum = float(momentum)
        self.gain_ema = {}; self.regret_ema = {}; self.count = {}; self.age = {}
        for lvl in self.levels:
            shape = (int(self.n_steps_by_level.get(lvl, 1)), self.n_modes)
            self.gain_ema[lvl] = torch.zeros(shape, device=device)
            self.regret_ema[lvl] = torch.zeros(shape, device=device)
            self.count[lvl] = torch.zeros(shape, device=device)
            self.age[lvl] = torch.zeros(shape, device=device)
    def update(self, level: str, step: int, mode: int, gain):
        if level not in self.gain_ema: return
        step=int(step); mode=int(mode)
        if step < 0 or step >= self.gain_ema[level].shape[0] or mode < 0 or mode >= self.n_modes: return
        g=torch.as_tensor(gain,device=self.gain_ema[level].device,dtype=self.gain_ema[level].dtype)
        self.age[level] += 1
        if float(g.detach().cpu()) >= 0:
            self.gain_ema[level][step,mode] = self.momentum*self.gain_ema[level][step,mode] + (1-self.momentum)*g
        else:
            self.regret_ema[level][step,mode] = self.momentum*self.regret_ema[level][step,mode] + (1-self.momentum)*(-g)
        self.count[level][step,mode] += 1
        self.age[level][step,mode] = 0

    def state_dict_json(self):
        return {
            'levels': list(self.levels),
            'n_steps_by_level': self.n_steps_by_level,
            'n_modes': self.n_modes,
            'momentum': self.momentum,
            'gain_ema': {k: v.detach().cpu().tolist() for k, v in self.gain_ema.items()},
            'regret_ema': {k: v.detach().cpu().tolist() for k, v in self.regret_ema.items()},
            'count': {k: v.detach().cpu().tolist() for k, v in self.count.items()},
            'age': {k: v.detach().cpu().tolist() for k, v in self.age.items()},
        }

    def save_json(self, path):
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps(self.state_dict_json(), indent=2), encoding='utf-8')

    def load_json(self, path):
        import json
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return False
        obj = json.loads(p.read_text(encoding='utf-8'))
        for name in ('gain_ema', 'regret_ema', 'count', 'age'):
            src = obj.get(name, {})
            dst = getattr(self, name)
            for lvl, arr in src.items():
                if lvl in dst:
                    t = torch.tensor(arr, device=dst[lvl].device, dtype=dst[lvl].dtype)
                    if tuple(t.shape) == tuple(dst[lvl].shape):
                        dst[lvl].copy_(t)
        return True

    def summary(self, mode_names=None, topk:int=8) -> Dict[str,Any]:
        names=list(mode_names or [f'mode_{i}' for i in range(self.n_modes)])
        out={'mode_feedback_count': int(sum(int(v.sum().detach().cpu()) for v in self.count.values()))}
        top={}
        for lvl in self.levels:
            rows=[]; gain=self.gain_ema[lvl]; regret=self.regret_ema[lvl]; count=self.count[lvl]
            for s in range(gain.shape[0]):
                for m in range(gain.shape[1]):
                    c=int(count[s,m].detach().cpu())
                    if c:
                        rows.append({'level':lvl,'step':s,'mode':names[m] if m<len(names) else f'mode_{m}','gain_ema':float(gain[s,m].detach().cpu()),'regret_ema':float(regret[s,m].detach().cpu()),'count':c})
            top[lvl]=sorted(rows,key=lambda r:r['gain_ema']-r['regret_ema'],reverse=True)[:topk]
        out['mode_feedback_top_by_level']=top
        return out
