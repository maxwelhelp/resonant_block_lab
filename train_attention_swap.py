#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from resonant_block_lab.fractal_sequence import FractalResonantConfig, FractalResonantSequenceBlock, SequenceAttractorHead
from resonant_block_lab.program_brain import ProgramBrain, ProgramInterventionPolicy, counterfactual_mode_credit, write_program_brain_outputs

class SeqProgramDataset(Dataset):
    def __init__(self, n=12000, length=128, input_dim=16, classes=8, seed=1, noise=0.24):
        g=torch.Generator().manual_seed(seed)
        self.x=torch.randn(n,length,input_dim,generator=g)*noise
        self.y=torch.randint(0,classes,(n,),generator=g)
        t=torch.linspace(0,1,length)
        for i in range(n):
            c=int(self.y[i]); ch=c%input_dim; freq=1+c%5
            self.x[i,:,ch]+=torch.sin(2*math.pi*freq*t+0.4*(c%3))*0.45
            pos=min(length//8+(c*11)%max(8,length-20),length-9)
            self.x[i,pos:pos+7,(ch+1)%input_dim]+=0.38+0.04*c
            trend=torch.linspace(-0.38,0.38,length) if c%2 else torch.linspace(0.38,-0.38,length)
            self.x[i,:,(ch+2)%input_dim]+=trend
            self.x[i,:,(ch+3)%input_dim]+=torch.sin(2*math.pi*(freq+2)*t)*0.22
            self.x[i]=torch.roll(self.x[i],shifts=int(torch.randint(-10,11,(1,),generator=g)),dims=0)
    def __len__(self): return self.y.numel()
    def __getitem__(self,i): return self.x[i],self.y[i]

def features(x):
    return torch.cat([x.mean(1),x.std(1),x.max(1).values,x.pow(2).mean(1)],dim=-1)

class FFNBlock(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.net=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim*2),nn.GELU(),nn.Linear(dim*2,dim))
    def forward(self,x): return x+0.5*self.net(x)

class SwapClassifier(nn.Module):
    def __init__(self,args):
        super().__init__(); self.variant=args.variant; D=args.dim
        self.in_proj=nn.Linear(args.input_dim,D)
        self.pos=nn.Parameter(torch.randn(1,args.length,D)*0.01)
        if self.variant in ('attn','res_before_attn','attn_then_res'):
            self.attn=nn.TransformerEncoderLayer(d_model=D,nhead=args.heads,dim_feedforward=D*2,dropout=0.0,batch_first=True,norm_first=True)
        else:
            self.attn=None
        if self.variant in ('res_before_attn','res_instead_attn','attn_then_res'):
            cfg=FractalResonantConfig(dim=D,n_modes=args.n_modes,macro_steps=args.macro_steps,micro_steps=args.micro_steps,aux_classes=args.classes,controller_hidden=args.controller_hidden,use_ff_refine=args.use_ff_refine)
            self.res=FractalResonantSequenceBlock(cfg)
        else:
            self.res=None
        self.ffn=FFNBlock(D) if self.variant=='res_instead_attn' else None
        self.head=SequenceAttractorHead(args.classes,D,head_mode=args.head_mode)
    def forward(self,x,return_stats=False):
        h=self.in_proj(x)+self.pos[:,:x.shape[1],:]
        stats={}
        if self.variant=='attn':
            h=self.attn(h)
        elif self.variant=='res_before_attn':
            h,stats=self.res(h,return_stats=True)
            h=self.attn(h)
        elif self.variant=='res_instead_attn':
            h,stats=self.res(h,return_stats=True)
            h=self.ffn(h)
        elif self.variant=='attn_then_res':
            h=self.attn(h)
            h,stats=self.res(h,return_stats=True)
        else:
            raise ValueError(self.variant)
        feats=features(h)
        logits,attr_logits,linear_logits,energy,min_energy=self.head(feats)
        if return_stats:
            stats.update({'attr_logits':attr_logits,'linear_logits':linear_logits,'attractor_energy':energy,'min_energy':min_energy,'field_energy':h.pow(2).mean(),'final_features':feats})
            return logits,stats
        return logits

def seq_ce(logits,y,a,b):
    B,S,C=logits.shape; target=y[:,None].expand(B,S).reshape(B*S)
    ce=F.cross_entropy(logits.reshape(B*S,C),target,reduction='none').view(B,S)
    w=torch.linspace(a,b,S,device=y.device).view(1,S)
    return (ce*w).sum()/(w.sum()*B)

def loss_fn(model,logits,stats,y,args):
    ce=F.cross_entropy(logits,y)
    attr=F.cross_entropy(stats['attr_logits'],y) if 'attr_logits' in stats else ce*0
    macro=seq_ce(stats['macro_logits'],y,0.15,1.0) if 'macro_logits' in stats else ce*0
    micro=seq_ce(stats['micro_logits'],y,0.05,0.65) if 'micro_logits' in stats else ce*0
    if 'attractor_energy' in stats:
        e=stats['attractor_energy']; good=e.gather(1,y.view(-1,1)).squeeze(1)
        bad=e.masked_fill(F.one_hot(y,num_classes=e.shape[1]).bool(),float('inf')).min(1).values
        margin=F.relu(good-bad+args.margin).mean(); good_e=good.mean().detach(); bad_e=bad.mean().detach()
    else:
        margin=ce*0; good_e=ce.detach()*0; bad_e=ce.detach()*0
    sep=model.head.separation_loss(); field=(stats.get('field_energy',torch.tensor(1.0,device=y.device))-1.0).pow(2)
    if 'gate_history' in stats:
        alpha=stats['gate_history'][:,0]; skip=F.relu(args.alpha_min-alpha).pow(2).mean()
        md=stats['micro_deltas']; smooth=md[:,1:].sub(md[:,:-1]).abs().mean() if md.shape[1]>1 else md.mean()*0
    else:
        skip=ce*0; smooth=ce*0
    total=ce+args.w_attr*attr+args.w_macro*macro+args.w_micro*micro+args.w_margin*margin+args.w_sep*sep+args.w_energy*field+args.w_skip*skip+args.w_smooth*smooth
    return total,{'ce':ce,'attr_ce':attr,'macro_ce':macro,'micro_ce':micro,'margin':margin,'sep':sep,'field_energy':field,'correct_energy':good_e,'wrong_energy':bad_e}

def safe_acc(stats,y,key):
    return float((stats[key][:,-1].argmax(-1)==y).float().mean().detach().cpu()) if key in stats else 0.0

def avg(rows,k):
    vals=[r[k] for r in rows if k in r]; return sum(vals)/max(1,len(vals))

def run_epoch(model,loader,opt,args,device,train=True):
    model.train(train); ok=tot=0; logs=[]; last=None; last_y=None; brain=ProgramBrain() if not train else None; credit_x=None; credit_y=None; brain=ProgramBrain() if not train else None
    for bi,(x,y) in enumerate(loader,1):
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        with torch.set_grad_enabled(train):
            logits,stats=model(x,return_stats=True); loss,ls=loss_fn(model,logits,stats,y,args)
            if train:
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        ok+=(logits.argmax(-1)==y).sum().item(); tot+=y.numel(); last=stats; last_y=y.detach().cpu()
        if brain is not None:
            brain.add(stats, y, logits)
            if credit_x is None:
                credit_x=x.detach()
                credit_y=y.detach()
        rec={k:float(v.detach().cpu()) for k,v in ls.items()}
        rec.update({'macc':safe_acc(stats,y,'macro_logits'),'uacc':safe_acc(stats,y,'micro_logits')})
        if 'gate_history' in stats:
            rec.update({'alpha':float(stats['gate_history'][:,0].mean().detach().cpu()),'eff_H':float(stats['eff_mode_entropy'].detach().cpu())})
        logs.append(rec)
        if train and args.log_every and bi%args.log_every==0:
            print(f"{args.variant} batch {bi:4d}/{len(loader)} loss={float(loss.detach()):.3f} ce={rec['ce']:.3f} macro={rec['macro_ce']:.3f} micro={rec['micro_ce']:.3f} macc={rec['macc']:.3f} alpha={rec.get('alpha',0):.3f} H={rec.get('eff_H',0):.3f}",flush=True)
    return ok/max(1,tot),logs,last,last_y,brain,credit_x,credit_y

def save_diag(out,stats,y):
    if stats is None or y is None: return
    diag={'has_resonator':'macro_logits' in stats}
    if 'macro_logits' in stats:
        yf=y.view(-1,1); mp=stats['macro_logits'].detach().cpu().argmax(-1); up=stats['micro_logits'].detach().cpu().argmax(-1)
        diag.update({
            'macro_acc_by_step':(mp==yf).float().mean(0).numpy().round(5).tolist(),
            'micro_acc_by_microstep_flat':(up==yf).float().mean(0).numpy().round(5).tolist(),
            'macro_mode_schedule_mean_by_step':stats['macro_q_history'].detach().float().mean(0).cpu().numpy().round(5).tolist(),
            'micro_macro_mode_schedule_mean_by_microstep':stats['micro_macro_q_history'].detach().float().mean(0).cpu().numpy().round(5).tolist(),
            'micro_mode_schedule_mean_by_microstep':stats['micro_q_history'].detach().float().mean(0).cpu().numpy().round(5).tolist(),
            'effective_mode_schedule_mean_by_microstep':stats['q_eff_history'].detach().float().mean(0).cpu().numpy().round(5).tolist(),
            'gate_schedule_alpha_beta_gamma_eta_rho_macroMix':stats['gate_history'].detach().float().cpu().numpy().round(5).tolist(),
            'macro_delta_mean_by_step':stats['macro_deltas'].detach().float().mean(0).cpu().numpy().round(5).tolist(),
            'micro_delta_mean_by_microstep':stats['micro_deltas'].detach().float().mean(0).cpu().numpy().round(5).tolist(),
            'program':stats['program']})
    (out/'diagnostics.json').write_text(json.dumps(diag,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=['attn','res_before_attn','res_instead_attn','attn_then_res'],default='res_instead_attn')
    p.add_argument('--epochs',type=int,default=8); p.add_argument('--batch',type=int,default=128); p.add_argument('--length',type=int,default=128)
    p.add_argument('--input_dim',type=int,default=16); p.add_argument('--dim',type=int,default=64); p.add_argument('--classes',type=int,default=8)
    p.add_argument('--heads',type=int,default=4); p.add_argument('--n_modes',type=int,default=8); p.add_argument('--macro_steps',type=int,default=8); p.add_argument('--micro_steps',type=int,default=3)
    p.add_argument('--controller_hidden',type=int,default=192); p.add_argument('--train_n',type=int,default=12000); p.add_argument('--val_n',type=int,default=2000); p.add_argument('--noise',type=float,default=0.24)
    p.add_argument('--head_mode',default='attractor_only',choices=['attractor_only','hybrid','linear_only']); p.add_argument('--use_ff_refine',action='store_true')
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--w_attr',type=float,default=0.25); p.add_argument('--w_macro',type=float,default=0.35); p.add_argument('--w_micro',type=float,default=0.30)
    p.add_argument('--w_margin',type=float,default=0.05); p.add_argument('--w_sep',type=float,default=0.02); p.add_argument('--w_energy',type=float,default=0.01); p.add_argument('--w_skip',type=float,default=0.02); p.add_argument('--w_smooth',type=float,default=0.02)
    p.add_argument('--alpha_min',type=float,default=0.20); p.add_argument('--margin',type=float,default=0.08); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir',default='runs/attention_swap'); p.add_argument('--log_every',type=int,default=25)
    p.add_argument('--credit_modes',type=int,default=8); p.add_argument('--intervention_mode',default='diagnostic',choices=['off','diagnostic','active','rollback'])
    p.add_argument('--intervention_patience',type=int,default=2); p.add_argument('--burst_scale',type=float,default=0.035); p.add_argument('--dampen_scale',type=float,default=0.03)
    args=p.parse_args(); device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu')
    out=Path(args.out_dir)/args.variant; out.mkdir(parents=True,exist_ok=True); (out/'config.json').write_text(json.dumps(vars(args),indent=2,ensure_ascii=False))
    tr=DataLoader(SeqProgramDataset(args.train_n,args.length,args.input_dim,args.classes,1,args.noise),batch_size=args.batch,shuffle=True,pin_memory=device.type=='cuda')
    va=DataLoader(SeqProgramDataset(args.val_n,args.length,args.input_dim,args.classes,999,args.noise),batch_size=args.batch,shuffle=False,pin_memory=device.type=='cuda')
    model=SwapClassifier(args).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    print('variant',args.variant,'params',sum(p.numel() for p in model.parameters() if p.requires_grad),flush=True)
    rows=[]; best=0.0
    policy=ProgramInterventionPolicy(args.intervention_patience,args.burst_scale,args.dampen_scale)
    for ep in range(1,args.epochs+1):
        t=time.time(); tr_acc,tr_logs,_,_,_=run_epoch(model,tr,opt,args,device,True); va_acc,va_logs,last,last_y,brain=run_epoch(model,va,opt,args,device,False)
        row={'epoch':ep,'variant':args.variant,'train_acc':tr_acc,'val_acc':va_acc,'sec':time.time()-t,'train_ce':avg(tr_logs,'ce'),'train_macro_ce':avg(tr_logs,'macro_ce'),'train_micro_ce':avg(tr_logs,'micro_ce'),'train_macc':avg(tr_logs,'macc'),'train_uacc':avg(tr_logs,'uacc'),'train_alpha':avg(tr_logs,'alpha'),'val_ce':avg(va_logs,'ce'),'val_macro_ce':avg(va_logs,'macro_ce'),'val_micro_ce':avg(va_logs,'micro_ce'),'val_macc':avg(va_logs,'macc'),'val_uacc':avg(va_logs,'uacc'),'val_alpha':avg(va_logs,'alpha')}
        if last and 'program' in last: row['program']=last['program']
        rows.append(row); save_diag(out,last,last_y)
        if brain is not None:
            summary=brain.finalize(ep,row)
            if credit_x is not None and args.credit_modes != 0:
                def _eval_loss(logits,stats,yy):
                    return loss_fn(model,logits,stats,yy,args)[0]
                summary['true_counterfactual_credit']=counterfactual_mode_credit(model,credit_x,credit_y,_eval_loss,args.credit_modes)
            write_program_brain_outputs(out,summary,best_acc=best,row_acc=va_acc)
        print(json.dumps(row,ensure_ascii=False)[:1600],flush=True)
        if va_acc>best: best=va_acc; torch.save({'model':model.state_dict(),'args':vars(args),'row':row},out/'best.pt'); print('best',best,flush=True)
        if summary is not None:
            summary['policy_action']=policy.apply(model,summary,va_acc,args.intervention_mode)
            write_program_brain_outputs(out,summary,best_acc=best,row_acc=va_acc)
        keys=[k for k in rows[0] if k!='program']
        with (out/'history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); [w.writerow({k:r.get(k) for k in keys}) for r in rows]
        (out/'history.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    print('saved',out,'best',best,flush=True)
if __name__=='__main__': main()
