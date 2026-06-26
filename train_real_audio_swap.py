#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, os, time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

try:
    import torchaudio
except Exception as e:
    torchaudio = None
    TORCHAUDIO_IMPORT_ERROR = e
else:
    TORCHAUDIO_IMPORT_ERROR = None

from resonant_block_lab.fractal_sequence import FractalResonantConfig, FractalResonantSequenceBlock, SequenceAttractorHead
from resonant_block_lab.program_brain import ProgramBrain, counterfactual_mode_credit, write_program_brain_outputs

LABELS = ['backward','bed','bird','cat','dog','down','eight','five','follow','forward','four','go','happy','house','learn','left','marvin','nine','no','off','on','one','right','seven','sheila','six','stop','three','tree','two','up','visual','wow','yes','zero']
LABEL_TO_ID = {x:i for i,x in enumerate(LABELS)}


def find_data_root(user_root: Optional[str]) -> str:
    candidates = []
    if user_root:
        candidates.append(user_root)
    candidates += [
        './data',
        '../latent_prism_field_lab/data_speechcommands',
        '../latent_prism_field_lab/data',
        '/home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/latent_prism_field_lab/data_speechcommands',
        '/home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/latent_prism_field_lab/data',
        '/home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/resonant_block_lab/data',
        '/home/maxwelhelp/.cache/torch/datasets',
        '/tmp/speechcommands',
    ]
    for c in candidates:
        p = Path(c).expanduser()
        if (p/'SpeechCommands'/'speech_commands_v0.02').exists() or (p/'speech_commands_v0.02').exists():
            return str(p)
    # If not found, return the first candidate. With --download torchaudio will create
    # root/SpeechCommands/speech_commands_v0.02 there. Without --download we raise a
    # readable error before training starts.
    return candidates[0]


class SpeechCommandsMel(Dataset):
    def __init__(self, root: str, subset: str, sample_rate=16000, max_len=16000, n_mels=64, download=False, max_items=0):
        if torchaudio is None:
            raise RuntimeError(f'torchaudio import failed: {TORCHAUDIO_IMPORT_ERROR}')
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        self.ds = torchaudio.datasets.SPEECHCOMMANDS(root=str(root_path), url='speech_commands_v0.02', folder_in_archive='SpeechCommands', download=download, subset=subset)
        if max_items and max_items > 0:
            self.indices = list(range(min(max_items, len(self.ds))))
        else:
            self.indices = list(range(len(self.ds)))
        self.sample_rate = sample_rate
        self.max_len = max_len
        self.n_mels = n_mels
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=400, hop_length=160, win_length=400, n_mels=n_mels, power=2.0)
        self.amplitude = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        waveform, sr, label, *_ = self.ds[self.indices[i]]
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.shape[-1] < self.max_len:
            waveform = F.pad(waveform, (0, self.max_len - waveform.shape[-1]))
        else:
            waveform = waveform[..., :self.max_len]
        mel = self.amplitude(self.mel(waveform)).squeeze(0).transpose(0,1).contiguous()  # [T, n_mels]
        mel = (mel + 40.0) / 40.0
        mel = mel.clamp(-2.0, 2.0)
        return mel.float(), torch.tensor(LABEL_TO_ID[label], dtype=torch.long)


def features(x):
    return torch.cat([x.mean(1), x.std(1), x.max(1).values, x.pow(2).mean(1)], dim=-1)


class AudioSwapClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.variant = args.variant
        self.in_norm = nn.LayerNorm(args.n_mels)
        self.in_proj = nn.Linear(args.n_mels, args.dim)
        self.pos = nn.Parameter(torch.randn(1, args.max_frames, args.dim) * 0.01)
        if args.variant in ('attn','res_before_attn','attn_then_res'):
            self.attn = nn.TransformerEncoderLayer(d_model=args.dim, nhead=args.heads, dim_feedforward=args.dim*2, dropout=args.dropout, batch_first=True, norm_first=True)
        else:
            self.attn = None
        if args.variant in ('res_instead_attn','res_before_attn','attn_then_res'):
            cfg = FractalResonantConfig(dim=args.dim, n_modes=args.n_modes, macro_steps=args.macro_steps, micro_steps=args.micro_steps, aux_classes=args.classes, controller_hidden=args.controller_hidden, use_ff_refine=args.use_ff_refine, mix_topk=args.mix_topk, enable_matrix_program=args.enable_matrix_program, enable_symbolic_product=args.enable_symbolic_product, enable_lowrank=args.enable_lowrank, enable_input_primitive=args.enable_input_primitive, program_steps=args.program_steps, program_rank=args.program_rank)
            self.res = FractalResonantSequenceBlock(cfg)
        else:
            self.res = None
        self.ffn = nn.Sequential(nn.LayerNorm(args.dim), nn.Linear(args.dim, args.dim*2), nn.GELU(), nn.Dropout(args.dropout), nn.Linear(args.dim*2, args.dim))
        self.head = SequenceAttractorHead(args.classes, args.dim, head_mode=args.head_mode)

    def forward(self, x, return_stats=False):
        h = self.in_proj(self.in_norm(x))
        h = h + self.pos[:, :h.shape[1], :]
        stats = {}
        if self.variant == 'attn':
            h = self.attn(h)
        elif self.variant == 'res_instead_attn':
            h, stats = self.res(h, return_stats=True)
            h = h + 0.5 * self.ffn(h)
        elif self.variant == 'res_before_attn':
            h, stats = self.res(h, return_stats=True)
            h = self.attn(h)
        elif self.variant == 'attn_then_res':
            h = self.attn(h)
            h, stats = self.res(h, return_stats=True)
        else:
            raise ValueError(self.variant)
        feats = features(h)
        logits, attr_logits, linear_logits, energy, min_energy = self.head(feats)
        if return_stats:
            stats.update({'attr_logits': attr_logits, 'linear_logits': linear_logits, 'attractor_energy': energy, 'min_energy': min_energy, 'field_energy': h.pow(2).mean(), 'program': stats.get('program')})
            return logits, stats
        return logits


def seq_ce(logits, y, a, b):
    if logits is None:
        return None
    B,S,C = logits.shape
    target = y[:,None].expand(B,S).reshape(B*S)
    ce = F.cross_entropy(logits.reshape(B*S,C), target, reduction='none').view(B,S)
    w = torch.linspace(a,b,S,device=y.device).view(1,S)
    return (ce*w).sum() / (w.sum()*B)


def loss_fn(model, logits, stats, y, args):
    ce = F.cross_entropy(logits, y)
    attr = F.cross_entropy(stats['attr_logits'], y) if 'attr_logits' in stats else ce*0
    macro = seq_ce(stats.get('macro_logits'), y, 0.15, 1.0) if 'macro_logits' in stats else ce*0
    micro = seq_ce(stats.get('micro_logits'), y, 0.05, 0.65) if 'micro_logits' in stats else ce*0
    if 'attractor_energy' in stats:
        e = stats['attractor_energy']
        good = e.gather(1, y.view(-1,1)).squeeze(1)
        bad = e.masked_fill(F.one_hot(y, num_classes=e.shape[1]).bool(), float('inf')).min(1).values
        margin = F.relu(good - bad + args.margin).mean()
        good_e = good.mean().detach(); bad_e = bad.mean().detach()
    else:
        margin = ce*0; good_e = ce.detach()*0; bad_e = ce.detach()*0
    sep = model.head.separation_loss()
    field = (stats.get('field_energy', torch.tensor(1.0, device=y.device)) - 1.0).pow(2)
    if 'gate_history' in stats:
        alpha = stats['gate_history'][:,0]
        skip = F.relu(args.alpha_min-alpha).pow(2).mean()
        md = stats['micro_deltas']
        smooth = md[:,1:].sub(md[:,:-1]).abs().mean() if md.shape[1] > 1 else md.mean()*0
    else:
        skip = ce*0; smooth = ce*0
    total = ce + args.w_attr*attr + args.w_macro*macro + args.w_micro*micro + args.w_margin*margin + args.w_sep*sep + args.w_energy*field + args.w_skip*skip + args.w_smooth*smooth
    return total, {'ce':ce,'attr_ce':attr,'macro_ce':macro,'micro_ce':micro,'margin':margin,'sep':sep,'field_energy':field,'correct_energy':good_e,'wrong_energy':bad_e}


def avg(rows,k):
    vals=[r[k] for r in rows if k in r]
    return sum(vals)/max(1,len(vals))


def acc_last(stats,y,key):
    return float((stats[key][:,-1].argmax(-1)==y).float().mean().detach().cpu()) if key in stats else 0.0


def run_epoch(model, loader, opt, args, device, train=True):
    model.train(train)
    ok=tot=0; logs=[]; brain=ProgramBrain() if not train else None; credit_x=None; credit_y=None
    for bi,(x,y) in enumerate(loader,1):
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        with torch.set_grad_enabled(train):
            logits,stats=model(x,return_stats=True)
            loss,ls=loss_fn(model,logits,stats,y,args)
            if train:
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); opt.step()
        ok += (logits.argmax(-1)==y).sum().item(); tot += y.numel()
        if brain is not None:
            brain.add(stats,y,logits)
            if credit_x is None:
                credit_x=x.detach(); credit_y=y.detach()
        rec={k:float(v.detach().cpu()) for k,v in ls.items()}
        rec.update({'macc':acc_last(stats,y,'macro_logits'),'uacc':acc_last(stats,y,'micro_logits')})
        if 'gate_history' in stats:
            rec.update({'alpha':float(stats['gate_history'][:,0].mean().detach().cpu()), 'eff_H':float(stats['eff_mode_entropy'].detach().cpu())})
        logs.append(rec)
        if train and args.log_every and bi % args.log_every == 0:
            print(f"{args.variant} batch {bi:4d}/{len(loader)} loss={float(loss.detach()):.3f} ce={rec['ce']:.3f} macro={rec['macro_ce']:.3f} micro={rec['micro_ce']:.3f} macc={rec['macc']:.3f} alpha={rec.get('alpha',0):.3f}", flush=True)
    return ok/max(1,tot), logs, brain, credit_x, credit_y


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=['attn','res_instead_attn','res_before_attn','attn_then_res'],default='res_instead_attn')
    p.add_argument('--data_dir',default=''); p.add_argument('--download',action='store_true')
    p.add_argument('--epochs',type=int,default=15); p.add_argument('--batch',type=int,default=128); p.add_argument('--num_workers',type=int,default=4)
    p.add_argument('--max_train',type=int,default=0); p.add_argument('--max_val',type=int,default=0)
    p.add_argument('--n_mels',type=int,default=64); p.add_argument('--max_frames',type=int,default=101)
    p.add_argument('--dim',type=int,default=96); p.add_argument('--classes',type=int,default=35); p.add_argument('--heads',type=int,default=4)
    p.add_argument('--n_modes',type=int,default=8); p.add_argument('--macro_steps',type=int,default=6); p.add_argument('--micro_steps',type=int,default=2); p.add_argument('--mix_topk',type=int,default=0)
    p.add_argument('--enable_matrix_program',action='store_true'); p.add_argument('--enable_symbolic_product',action='store_true')
    p.add_argument('--enable_lowrank',action='store_true'); p.add_argument('--enable_input_primitive',action='store_true')
    p.add_argument('--program_steps',type=int,default=2); p.add_argument('--program_rank',type=int,default=8)
    p.add_argument('--controller_hidden',type=int,default=192); p.add_argument('--dropout',type=float,default=0.05); p.add_argument('--use_ff_refine',action='store_true')
    p.add_argument('--head_mode',default='attractor_only',choices=['attractor_only','hybrid','linear_only'])
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--weight_decay',type=float,default=1e-4); p.add_argument('--grad_clip',type=float,default=1.0)
    p.add_argument('--w_attr',type=float,default=0.25); p.add_argument('--w_macro',type=float,default=0.25); p.add_argument('--w_micro',type=float,default=0.20)
    p.add_argument('--w_margin',type=float,default=0.04); p.add_argument('--w_sep',type=float,default=0.015); p.add_argument('--w_energy',type=float,default=0.005); p.add_argument('--w_skip',type=float,default=0.01); p.add_argument('--w_smooth',type=float,default=0.01)
    p.add_argument('--alpha_min',type=float,default=0.15); p.add_argument('--margin',type=float,default=0.08)
    p.add_argument('--credit_modes',type=int,default=8); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir',default='runs/real_audio_swap'); p.add_argument('--log_every',type=int,default=50)
    args=p.parse_args()
    device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu')
    root=find_data_root(args.data_dir or None)
    expected = Path(root) / 'SpeechCommands' / 'speech_commands_v0.02'
    if not expected.exists() and not args.download:
        raise RuntimeError(
            f'SpeechCommands not found at {expected}. Pass --data_dir ROOT where ROOT/SpeechCommands/speech_commands_v0.02 exists, or add --download to create it.'
        )
    out=Path(args.out_dir)/args.variant; out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps({**vars(args),'resolved_data_root':root},indent=2,ensure_ascii=False))
    train_ds=SpeechCommandsMel(root,'training',n_mels=args.n_mels,download=args.download,max_items=args.max_train)
    val_ds=SpeechCommandsMel(root,'validation',n_mels=args.n_mels,download=args.download,max_items=args.max_val)
    tr=DataLoader(train_ds,batch_size=args.batch,shuffle=True,num_workers=args.num_workers,pin_memory=device.type=='cuda',persistent_workers=args.num_workers>0)
    va=DataLoader(val_ds,batch_size=args.batch,shuffle=False,num_workers=args.num_workers,pin_memory=device.type=='cuda',persistent_workers=args.num_workers>0)
    model=AudioSwapClassifier(args).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    print('variant',args.variant,'root',root,'train',len(train_ds),'val',len(val_ds),'params',sum(p.numel() for p in model.parameters() if p.requires_grad),flush=True)
    rows=[]; best=0.0
    for ep in range(1,args.epochs+1):
        t=time.time(); tr_acc,tr_logs,_,_,_=run_epoch(model,tr,opt,args,device,True); va_acc,va_logs,brain,credit_x,credit_y=run_epoch(model,va,opt,args,device,False)
        row={'epoch':ep,'variant':args.variant,'train_acc':tr_acc,'val_acc':va_acc,'sec':time.time()-t,
             'train_ce':avg(tr_logs,'ce'),'train_macro_ce':avg(tr_logs,'macro_ce'),'train_micro_ce':avg(tr_logs,'micro_ce'),'train_macc':avg(tr_logs,'macc'),'train_uacc':avg(tr_logs,'uacc'),'train_alpha':avg(tr_logs,'alpha'),
             'val_ce':avg(va_logs,'ce'),'val_macro_ce':avg(va_logs,'macro_ce'),'val_micro_ce':avg(va_logs,'micro_ce'),'val_macc':avg(va_logs,'macc'),'val_uacc':avg(va_logs,'uacc'),'val_alpha':avg(va_logs,'alpha')}
        summary=None
        if brain is not None:
            summary=brain.finalize(ep,row)
            if credit_x is not None and args.credit_modes:
                def _eval_loss(logits,stats,yy):
                    return loss_fn(model,logits,stats,yy,args)[0]
                summary['true_counterfactual_credit']=counterfactual_mode_credit(model,credit_x,credit_y,_eval_loss,args.credit_modes)
            write_program_brain_outputs(out,summary,best_acc=best,row_acc=va_acc)
        rows.append(row); print(json.dumps(row,ensure_ascii=False)[:1600],flush=True)
        if va_acc>best:
            best=va_acc; torch.save({'model':model.state_dict(),'args':vars(args),'row':row},out/'best.pt'); print('best',best,flush=True)
        keys=[k for k in rows[0].keys()]
        with (out/'history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); [w.writerow({k:r.get(k) for k in keys}) for r in rows]
        (out/'history.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
    print('saved',out,'best',best,flush=True)

if __name__=='__main__':
    main()
