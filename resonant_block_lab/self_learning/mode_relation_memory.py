from __future__ import annotations
from typing import Dict, Any, Iterable
import torch

class ModeRelationMemory:
    def __init__(self, n_modes:int, momentum:float=0.98, device='cpu'):
        self.n_modes=int(n_modes); self.momentum=float(momentum)
        self.co_pos=torch.zeros(n_modes,n_modes,device=device)
        self.co_neg=torch.zeros(n_modes,n_modes,device=device)
        self.conflict=torch.zeros(n_modes,n_modes,device=device)
        self.count=torch.zeros(n_modes,n_modes,device=device)
    def update(self, ids:Iterable[int], gain):
        ids=list({int(i) for i in ids if 0 <= int(i) < self.n_modes})
        if len(ids)<2: return
        mask=torch.zeros(self.n_modes,self.n_modes,device=self.count.device)
        for i in ids:
            for j in ids:
                if i != j: mask[i,j]=1.0
        g=torch.as_tensor(gain,device=self.count.device,dtype=self.count.dtype)
        self.count += mask
        if float(g.detach().cpu()) >= 0:
            self.co_pos=self.momentum*self.co_pos+(1-self.momentum)*mask*g
        else:
            a=-g; self.co_neg=self.momentum*self.co_neg+(1-self.momentum)*mask*a; self.conflict=self.momentum*self.conflict+(1-self.momentum)*mask*a
        self.co_pos.fill_diagonal_(0); self.co_neg.fill_diagonal_(0); self.conflict.fill_diagonal_(0); self.count.fill_diagonal_(0)

    def state_dict_json(self):
        return {
            'n_modes': self.n_modes,
            'momentum': self.momentum,
            'co_pos': self.co_pos.detach().cpu().tolist(),
            'co_neg': self.co_neg.detach().cpu().tolist(),
            'conflict': self.conflict.detach().cpu().tolist(),
            'count': self.count.detach().cpu().tolist(),
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
        for name in ('co_pos', 'co_neg', 'conflict', 'count'):
            if name in obj:
                dst = getattr(self, name)
                t = torch.tensor(obj[name], device=dst.device, dtype=dst.dtype)
                if tuple(t.shape) == tuple(dst.shape):
                    dst.copy_(t)
        return True

    def summary(self, mode_names=None) -> Dict[str,Any]:
        names=list(mode_names or [f'mode_{i}' for i in range(self.n_modes)])
        def top(mat):
            rows=[]
            for i in range(self.n_modes):
                for j in range(self.n_modes):
                    v=float(mat[i,j].detach().cpu())
                    if i!=j and abs(v)>0:
                        rows.append((names[i] if i<len(names) else f'mode_{i}', names[j] if j<len(names) else f'mode_{j}', v))
            return sorted(rows,key=lambda x:x[2],reverse=True)[:10]
        eps=1e-8
        conflict_norm=self.conflict/(self.conflict+self.co_pos+eps)
        return {
            'relation_pair_count_total': float(self.count.sum().detach().cpu()),
            'relation_observed_pairs': int((self.count>0).sum().detach().cpu()),
            'relation_co_pos_mean': float(self.co_pos.mean().detach().cpu()),
            'relation_co_neg_mean': float(self.co_neg.mean().detach().cpu()),
            'relation_conflict_mean': float(self.conflict.mean().detach().cpu()),
            'relation_conflict_norm_mean': float(conflict_norm.mean().detach().cpu()),
            'relation_co_pos_top': top(self.co_pos),
            'relation_conflict_top': top(self.conflict),
        }
