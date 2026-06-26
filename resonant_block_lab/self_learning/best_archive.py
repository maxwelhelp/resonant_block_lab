from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json, torch

class BestProgramArchive:
    def __init__(self, out_dir: str | Path, topk:int=5):
        self.out_dir=Path(out_dir); self.out_dir.mkdir(parents=True,exist_ok=True)
        self.topk_dir=self.out_dir/'topk_programs'; self.topk_dir.mkdir(exist_ok=True)
        self.best_loss=float('inf'); self.best_acc=-1e9; self.topk=int(topk)
    def maybe_update(self, epoch:int, val_loss:float, val_acc:float, model, summary:Dict[str,Any], metrics:Dict[str,Any]) -> bool:
        if float(val_loss) >= self.best_loss:
            return False
        self.best_loss=float(val_loss); self.best_acc=float(val_acc)
        torch.save({'model':model.state_dict(),'epoch':epoch,'val_loss':val_loss,'val_acc':val_acc}, self.out_dir/'best.pt')
        program={'epoch':epoch,'val_loss':val_loss,'val_acc':val_acc,**(summary or {})}
        (self.out_dir/'best_program.json').write_text(json.dumps(program,indent=2,ensure_ascii=False),encoding='utf-8')
        (self.out_dir/'best_epoch_summary.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
        lines=['# Best program',f'epoch: {epoch}',f'val_loss: {val_loss}',f'val_acc: {val_acc}']
        top=(summary or {}).get('program',{}).get('top_modes',[])
        if top:
            lines.append('## Top modes')
            lines += [f'- {n}: {v}' for n,v in top]
        (self.out_dir/'best_program.md').write_text('\n'.join(lines),encoding='utf-8')
        snap=self.topk_dir/f'epoch_{epoch:04d}_loss_{float(val_loss):.4f}.json'
        snap.write_text(json.dumps(program,indent=2,ensure_ascii=False),encoding='utf-8')
        return True
