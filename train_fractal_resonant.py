#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from resonant_block_lab.fractal_sequence import FractalResonantSequenceClassifier

class SeqProgramDataset(Dataset):
    def __init__(self, n=12000, length=128, input_dim=16, classes=8, seed=1, noise=0.24):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, length, input_dim, generator=g) * noise
        self.y = torch.randint(0, classes, (n,), generator=g)
        t = torch.linspace(0, 1, length)
        for i in range(n):
            c = int(self.y[i]); ch = c % input_dim; freq = 1 + c % 5
            self.x[i, :, ch] += torch.sin(2 * math.pi * freq * t + 0.4 * (c % 3)) * 0.45
            pos = min(length // 8 + (c * 11) % max(8, length - 20), length - 9)
            self.x[i, pos:pos+7, (ch+1) % input_dim] += 0.38 + 0.04 * c
            trend = torch.linspace(-0.38, 0.38, length) if c % 2 else torch.linspace(0.38, -0.38, length)
            self.x[i, :, (ch+2) % input_dim] += trend
            self.x[i, :, (ch+3) % input_dim] += torch.sin(2 * math.pi * (freq + 2) * t) * 0.22
            self.x[i] = torch.roll(self.x[i], shifts=int(torch.randint(-10, 11, (1,), generator=g)), dims=0)
    def __len__(self): return self.y.numel()
    def __getitem__(self, i): return self.x[i], self.y[i]

def seq_ce(logits, y, a, b):
    B,S,C = logits.shape
    target = y[:,None].expand(B,S).reshape(B*S)
    ce = F.cross_entropy(logits.reshape(B*S,C), target, reduction='none').view(B,S)
    w = torch.linspace(a,b,S,device=y.device).view(1,S)
    return (ce*w).sum()/(w.sum()*B)

def loss_fn(logits, stats, y, args):
    ce = F.cross_entropy(logits,y)
    macro = seq_ce(stats['macro_logits'],y,0.15,1.0)
    micro = seq_ce(stats['micro_logits'],y,0.05,0.65)
    alpha = stats['gate_history'][:,0]
    skip = F.relu(args.alpha_min-alpha).pow(2).mean()
    md = stats['micro_deltas']
    smooth = md[:,1:].sub(md[:,:-1]).abs().mean() if md.shape[1] > 1 else md.mean()*0
    total = ce + args.w_macro*macro + args.w_micro*micro + args.w_skip*skip + args.w_smooth*smooth
    return total, {'ce':ce,'macro_ce':macro,'micro_ce':micro,'skip':skip,'smooth':smooth}

def acc_last(stats,y,key):
    return float((stats[key][:,-1].argmax(-1)==y).float().mean().detach().cpu())

def avg(rows,k):
    vals=[r[k] for r in rows if k in r]
    return sum(vals)/max(1,len(vals))

def run_epoch(model,loader,opt,args,device,train=True):
    model.train(train); ok=tot=0; logs=[]; last=None
    for bi,(x,y) in enumerate(loader,1):
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        with torch.set_grad_enabled(train):
            logits,stats=model(x,return_stats=True)
            loss,ls=loss_fn(logits,stats,y,args)
            if train:
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        ok+=(logits.argmax(-1)==y).sum().item(); tot+=y.numel(); last=stats
        rec={k:float(v.detach().cpu()) for k,v in ls.items()}
        rec.update({'macc':acc_last(stats,y,'macro_logits'),'uacc':acc_last(stats,y,'micro_logits'),
                    'alpha':float(stats['gate_history'][:,0].mean().detach().cpu()),
                    'mdlast':float(stats['macro_deltas'][:,-1].mean().detach().cpu()),
                    'udlast':float(stats['micro_deltas'][:,-1].mean().detach().cpu())})
        logs.append(rec)
        if train and args.log_every and bi%args.log_every==0:
            print(f"batch {bi:4d}/{len(loader)} loss={float(loss.detach()):.3f} ce={rec['ce']:.3f} macro={rec['macro_ce']:.3f} micro={rec['micro_ce']:.3f} macc={rec['macc']:.3f} uacc={rec['uacc']:.3f} alpha={rec['alpha']:.3f}",flush=True)
    return ok/max(1,tot), logs, last

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=15); p.add_argument('--batch',type=int,default=128)
    p.add_argument('--length',type=int,default=128); p.add_argument('--input_dim',type=int,default=16)
    p.add_argument('--dim',type=int,default=64); p.add_argument('--classes',type=int,default=8)
    p.add_argument('--n_modes',type=int,default=8); p.add_argument('--macro_steps',type=int,default=8); p.add_argument('--micro_steps',type=int,default=3)
    p.add_argument('--train_n',type=int,default=12000); p.add_argument('--val_n',type=int,default=2000); p.add_argument('--noise',type=float,default=0.24)
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--w_macro',type=float,default=0.35); p.add_argument('--w_micro',type=float,default=0.30)
    p.add_argument('--w_skip',type=float,default=0.02); p.add_argument('--w_smooth',type=float,default=0.02); p.add_argument('--alpha_min',type=float,default=0.20)
    p.add_argument('--use_ff_refine',action='store_true'); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir',default='runs/fractal_resonant_full'); p.add_argument('--log_every',type=int,default=25)
    args=p.parse_args(); device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu')
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); (out/'config.json').write_text(json.dumps(vars(args),indent=2,ensure_ascii=False))
    tr=DataLoader(SeqProgramDataset(args.train_n,args.length,args.input_dim,args.classes,1,args.noise),batch_size=args.batch,shuffle=True,pin_memory=device.type=='cuda')
    va=DataLoader(SeqProgramDataset(args.val_n,args.length,args.input_dim,args.classes,999,args.noise),batch_size=args.batch,shuffle=False,pin_memory=device.type=='cuda')
    model=FractalResonantSequenceClassifier(args.input_dim,args.dim,args.classes,args.n_modes,args.macro_steps,args.micro_steps,args.use_ff_refine).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    print('params',sum(p.numel() for p in model.parameters() if p.requires_grad),flush=True)
    rows=[]; best=0.0
    for ep in range(1,args.epochs+1):
        t=time.time(); tr_acc,tr_logs,_=run_epoch(model,tr,opt,args,device,True); va_acc,va_logs,last=run_epoch(model,va,opt,args,device,False)
        row={'epoch':ep,'train_acc':tr_acc,'val_acc':va_acc,'sec':time.time()-t,
             'train_ce':avg(tr_logs,'ce'),'train_macro_ce':avg(tr_logs,'macro_ce'),'train_micro_ce':avg(tr_logs,'micro_ce'),
             'train_macc':avg(tr_logs,'macc'),'train_uacc':avg(tr_logs,'uacc'),'train_alpha':avg(tr_logs,'alpha'),
             'val_ce':avg(va_logs,'ce'),'val_macro_ce':avg(va_logs,'macro_ce'),'val_micro_ce':avg(va_logs,'micro_ce'),
             'val_macc':avg(va_logs,'macc'),'val_uacc':avg(va_logs,'uacc'),'val_alpha':avg(va_logs,'alpha')}
        if last is not None: row['program']=last['program']
        rows.append(row); print(json.dumps(row,ensure_ascii=False)[:1600],flush=True)
        if va_acc>best: best=va_acc; torch.save({'model':model.state_dict(),'args':vars(args),'row':row},out/'best.pt'); print(f'best={best:.4f}',flush=True)
        keys=[k for k in rows[0] if k!='program']
        with (out/'history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); [w.writerow({k:r.get(k) for k in keys}) for r in rows]
        (out/'history.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
    print('saved',out,'best',best,flush=True)
if __name__=='__main__': main()
