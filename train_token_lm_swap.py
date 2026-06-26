#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, random, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from resonant_block_lab.fractal_sequence import FractalResonantConfig, FractalResonantSequenceBlock
from resonant_block_lab.program_brain import ProgramBrain, counterfactual_mode_credit, write_program_brain_outputs
from resonant_block_lab.self_learning.token_program_brain import TokenProgramBrain
from resonant_block_lab.self_learning.mode_registry import summarize_categories
from resonant_block_lab.self_learning.best_archive import BestProgramArchive
from resonant_block_lab.self_learning.mode_feedback_memory import ModeFeedbackMemory
from resonant_block_lab.self_learning.mode_relation_memory import ModeRelationMemory
from resonant_block_lab.self_learning.program_search import TargetedProgramSearch

IGNORE_INDEX = -100


def collect_text(root: str, max_bytes: int = 20_000_000) -> bytes:
    p = Path(root).expanduser()
    exts = {'.py','.md','.txt','.json','.yaml','.yml','.sh','.toml','.cfg','.ini','.csv'}
    chunks = []
    total = 0
    if p.is_file():
        return p.read_bytes()[:max_bytes]
    for f in sorted(p.rglob('*')):
        if not f.is_file() or f.suffix.lower() not in exts:
            continue
        try:
            b = f.read_bytes()
        except Exception:
            continue
        # keep mostly textual files
        if b'\x00' in b[:4096]:
            continue
        chunks.append(b + b'\n')
        total += len(b) + 1
        if total >= max_bytes:
            break
    if not chunks:
        fallback = ("resonant field token benchmark\n" * 10000).encode('utf-8')
        return fallback
    return b''.join(chunks)[:max_bytes]


class ByteLMDataset(Dataset):
    def __init__(self, data: bytes, seq_len: int, n_samples: int, offset: int = 0):
        ids = torch.tensor(list(data), dtype=torch.long)
        if ids.numel() < seq_len + 2:
            ids = ids.repeat((seq_len + 2) // max(1, ids.numel()) + 1)
        self.ids = ids
        self.seq_len = int(seq_len)
        self.n_samples = int(n_samples)
        self.offset = int(offset)
        self.max_start = max(1, int(ids.numel()) - self.seq_len - 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        # deterministic strided sampling so train/val are reproducible
        start = (self.offset + i * 9973) % self.max_start
        chunk = self.ids[start:start+self.seq_len+1]
        return chunk[:-1], chunk[1:]


class LongRangeRecallDataset(Dataset):
    """Synthetic long-range token memory task.

    Input contains WRITE,value pairs far before QUERY positions. Target is only
    defined at QUERY positions. For delay > max causal-kernel offset, a local
    primitive cannot directly copy the value; the model must use pooled/slot memory
    or discover a longer strategy.
    """
    WRITE = 1
    QUERY = 2
    PAD = 0

    def __init__(self, seq_len: int, n_samples: int, vocab: int = 128, delay: int = 64, pairs: int = 3, offset: int = 0):
        assert vocab > 16
        self.seq_len = int(seq_len)
        self.n_samples = int(n_samples)
        self.vocab = int(vocab)
        self.delay = int(delay)
        self.pairs = int(pairs)
        self.offset = int(offset)
        if self.delay + 4 >= self.seq_len:
            raise ValueError(f"memory delay {self.delay} too large for seq_len {self.seq_len}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        g = torch.Generator()
        g.manual_seed(1234567 + self.offset + int(i) * 1009)
        x = torch.randint(10, self.vocab, (self.seq_len,), generator=g, dtype=torch.long)
        y = torch.full((self.seq_len,), IGNORE_INDEX, dtype=torch.long)
        max_write = self.seq_len - self.delay - 2
        # Spread pairs across the usable prefix and add tiny deterministic jitter.
        for j in range(self.pairs):
            base = 2 + (j * max(1, max_write - 3)) // max(1, self.pairs)
            jitter = int(torch.randint(0, max(1, min(4, max_write - base)), (1,), generator=g)) if base < max_write else 0
            wpos = min(max_write, base + jitter)
            qpos = wpos + self.delay
            value = int(torch.randint(10, self.vocab, (1,), generator=g))
            x[wpos] = self.WRITE
            x[wpos + 1] = value
            x[qpos] = self.QUERY
            y[qpos] = value
        return x, y


class CausalConvMixer(nn.Module):
    def __init__(self, dim: int, kernel: int = 5):
        super().__init__()
        self.kernel = int(kernel)
        self.dw = nn.Conv1d(dim, dim, kernel, groups=dim, bias=False)
        self.pw = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim))
    def forward(self, x):
        # causal depthwise conv: pad left only
        h = F.pad(x.transpose(1,2), (self.kernel-1, 0))
        h = self.dw(h).transpose(1,2)
        return x + 0.5 * self.pw(h)


class TokenLMSwap(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.variant = args.variant
        self.vocab = args.vocab
        self.embed = nn.Embedding(args.vocab, args.dim)
        self.pos = nn.Parameter(torch.randn(1, args.seq_len, args.dim) * 0.01)
        if args.variant == 'conv':
            self.block = CausalConvMixer(args.dim)
            self.res = None
        elif args.variant == 'res_lm':
            cfg = FractalResonantConfig(
                dim=args.dim,
                n_modes=args.n_modes,
                macro_steps=args.macro_steps,
                micro_steps=args.micro_steps,
                aux_classes=0,
                controller_hidden=args.controller_hidden,
                use_ff_refine=args.use_ff_refine,
                mix_topk=args.mix_topk,
                enable_matrix_program=args.enable_matrix_program,
                enable_symbolic_product=args.enable_symbolic_product,
                enable_lowrank=args.enable_lowrank,
                enable_input_primitive=args.enable_input_primitive,
                enable_token_primitives=args.enable_token_primitives,
                enable_hand_token_primitives=args.enable_hand_token_primitives,
                enable_slot_memory=args.enable_slot_memory,
                memory_slots=args.memory_slots,
                program_steps=args.program_steps,
                program_rank=args.program_rank,
            )
            self.res = FractalResonantSequenceBlock(cfg)
            self.block = None
        else:
            raise ValueError(args.variant)
        self.norm = nn.LayerNorm(args.dim)
        self.lm_head = nn.Linear(args.dim, args.vocab, bias=False)
        if args.tie_weights:
            self.lm_head.weight = self.embed.weight

    def forward(self, x, return_stats=False, intervention=None):
        h = self.embed(x) + self.pos[:, :x.shape[1], :]
        stats = {}
        if self.variant == 'conv':
            h = self.block(h)
        else:
            h, stats = self.res(h, return_stats=True, intervention=intervention)
        logits = self.lm_head(self.norm(h))
        if return_stats:
            return logits, stats
        return logits


def seq_loss(logits, y):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=IGNORE_INDEX)


def token_acc(logits, y):
    pred = logits.argmax(-1)
    mask = (y != IGNORE_INDEX)
    if not bool(mask.any()):
        return 0.0
    return float((pred[mask] == y[mask]).float().mean().detach().cpu())


def run_epoch(model, loader, opt, args, device, train=True, train_intervention=None):
    model.train(train)
    total_loss = 0.0; total_acc = 0.0; n = 0
    brain = ProgramBrain() if (not train and model.variant == 'res_lm') else None
    token_brain = TokenProgramBrain(getattr(getattr(model, 'res', None), 'bank', None).mode_names if (not train and model.variant == 'res_lm' and getattr(getattr(model, 'res', None), 'bank', None) is not None) else None) if (not train and model.variant == 'res_lm') else None
    credit_x = credit_y = None
    for bi,(x,y) in enumerate(loader,1):
        x=x.to(device); y=y.to(device)
        with torch.set_grad_enabled(train):
            logits, stats = model(x, return_stats=True, intervention=(train_intervention if train else None))
            loss = seq_loss(logits, y)
            # Optional deep supervision on every macro/micro state as token predictor.
            aux = 0.0
            if args.aux_token_loss and model.variant == 'res_lm':
                # Token-level deep supervision: every macro/micro field state must
                # be able to predict the next byte at every position. This is the
                # LM analogue of macro/micro supervision from audio classification.
                for key, w in [('macro_history', 0.7), ('micro_history', 0.3)]:
                    if key in stats:
                        hs = stats[key]  # [B,S,L,D]
                        b, st, l, d = hs.shape
                        aux_logits = model.lm_head(model.norm(hs.reshape(b * st, l, d)))
                        yy = y[:, None, :].expand(b, st, l).reshape(b * st, l)
                        aux = aux + w * F.cross_entropy(aux_logits.reshape(-1, aux_logits.shape[-1]), yy.reshape(-1))
                loss = loss + args.w_aux * aux
            if model.variant == 'res_lm' and args.enable_slot_memory:
                mem_pen = (
                    args.w_mem_read * stats.get('memory_read_cost', loss.new_tensor(0.0))
                    + args.w_mem_write * stats.get('memory_write_cost', loss.new_tensor(0.0))
                    + args.w_mem_garbage * stats.get('memory_garbage_cost', loss.new_tensor(0.0))
                    + args.w_mem_energy * stats.get('memory_slot_energy', loss.new_tensor(0.0))
                )
                loss = loss + mem_pen
            if model.variant == 'res_lm' and args.input_conditioned_cost > 0 and 'q_eff_history' in stats:
                try:
                    names = list(getattr(getattr(model, 'res', None).bank, 'mode_names', []))
                    if 'input_conditioned' in names:
                        idx = names.index('input_conditioned')
                        loss = loss + args.input_conditioned_cost * stats['q_eff_history'][..., idx].mean()
                except Exception:
                    pass
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
        bs = int((y != IGNORE_INDEX).sum().item()) if (y == IGNORE_INDEX).any() else x.numel()
        bs = max(1, bs)
        total_loss += float(loss.detach().cpu()) * bs
        total_acc += token_acc(logits, y) * bs
        n += bs
        if brain is not None:
            # Legacy ProgramBrain kept for existing reports; TokenProgramBrain is the correct token-level diagnostic.
            brain.add(stats, y[:,0].clamp_min(0), logits.mean(1))
            if token_brain is not None:
                token_brain.add_batch(stats, logits, y)
            if credit_x is None:
                credit_x=x.detach(); credit_y=y.detach()
        if train and args.log_every and bi % args.log_every == 0:
            print(f'batch {bi:4d}/{len(loader)} loss={float(loss.detach()):.3f} acc={token_acc(logits,y):.3f} memR={float(stats.get("memory_read_cost", loss.new_tensor(0.0)).detach().cpu()):.3f} memW={float(stats.get("memory_write_cost", loss.new_tensor(0.0)).detach().cpu()):.3f} garbage={float(stats.get("memory_garbage_cost", loss.new_tensor(0.0)).detach().cpu()):.3f}', flush=True)
    return total_loss/max(1,n), total_acc/max(1,n), brain, token_brain, credit_x, credit_y



@torch.no_grad()
def run_step_local_credit(model, x, y, args, feedback_memory=None, relation_memory=None):
    if model.variant != 'res_lm':
        return {}
    logits, stats = model(x, return_stats=True)
    full_loss = seq_loss(logits, y)
    q = stats.get('q_eff_history', None)
    if q is None or q.numel() == 0:
        return {'step_local_credit_count': 0}
    q_mean = q.float().mean(dim=0)  # [steps, modes]
    steps, modes = q_mean.shape
    candidates = []
    for st in range(steps):
        vals, idx = q_mean[st].topk(min(3, modes))
        for v, m in zip(vals.tolist(), idx.tolist()):
            candidates.append((float(v), int(st), int(m)))
    candidates.sort(reverse=True)
    rows = []
    names = list(getattr(getattr(model, 'res', None).bank, 'mode_names', [])) if getattr(model, 'res', None) is not None else [f'mode_{i}' for i in range(modes)]
    for _, st, m in candidates[:max(1, int(args.credit_budget))]:
        inter = {'disable_modes': [{'level': 'effective', 'step': st, 'mode': m}]}
        ab_logits, _ = model(x, return_stats=True, intervention=inter)
        ab_loss = seq_loss(ab_logits, y)
        gain = float((ab_loss - full_loss).detach().cpu())
        row = {'level': 'effective', 'step': st, 'mode_id': m, 'mode': names[m] if m < len(names) else f'mode_{m}', 'full_loss': float(full_loss.detach().cpu()), 'ablated_loss': float(ab_loss.detach().cpu()), 'gain': gain}
        rows.append(row)
        if feedback_memory is not None:
            feedback_memory.update('effective', st, m, gain)
        if relation_memory is not None:
            top_modes = q_mean[st].topk(min(3, modes)).indices.tolist()
            relation_memory.update(top_modes, gain)
    pos = sum(1 for r in rows if r['gain'] > 0)
    neg = sum(1 for r in rows if r['gain'] < 0)
    return {'step_local_credit_count': len(rows), 'step_local_credit_positive': pos, 'step_local_credit_negative': neg, 'step_local_credit_rows': rows}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['res_lm'], default='res_lm')
    ap.add_argument('--task', choices=['byte_lm','memory_recall'], default='byte_lm')
    ap.add_argument('--memory_delay', type=int, default=64)
    ap.add_argument('--memory_pairs', type=int, default=3)
    ap.add_argument('--corpus_dir', default='/home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/resonant_block_lab')
    ap.add_argument('--max_bytes', type=int, default=20000000)
    ap.add_argument('--seq_len', type=int, default=128)
    ap.add_argument('--vocab', type=int, default=256)
    ap.add_argument('--train_samples', type=int, default=20000)
    ap.add_argument('--val_samples', type=int, default=2000)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--n_modes', type=int, default=12)
    ap.add_argument('--macro_steps', type=int, default=6)
    ap.add_argument('--micro_steps', type=int, default=2)
    ap.add_argument('--controller_hidden', type=int, default=192)
    ap.add_argument('--mix_topk', type=int, default=0)
    ap.add_argument('--enable_matrix_program', action='store_true')
    ap.add_argument('--enable_symbolic_product', action='store_true')
    ap.add_argument('--enable_lowrank', action='store_true')
    ap.add_argument('--enable_input_primitive', action='store_true')
    ap.add_argument('--enable_token_primitives', action='store_true')
    ap.add_argument('--enable_hand_token_primitives', action='store_true')
    ap.add_argument('--enable_slot_memory', action='store_true')
    ap.add_argument('--memory_slots', type=int, default=8)
    ap.add_argument('--w_mem_read', type=float, default=0.005)
    ap.add_argument('--w_mem_write', type=float, default=0.010)
    ap.add_argument('--w_mem_garbage', type=float, default=0.060)
    ap.add_argument('--w_mem_energy', type=float, default=0.001)
    ap.add_argument('--program_steps', type=int, default=2)
    ap.add_argument('--program_rank', type=int, default=8)
    ap.add_argument('--aux_token_loss', action='store_true')
    ap.add_argument('--w_aux', type=float, default=0.1)
    ap.add_argument('--use_ff_refine', action='store_true')
    ap.add_argument('--tie_weights', action='store_true')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out_dir', default='runs/token_lm_swap')
    ap.add_argument('--credit_modes', type=int, default=0)
    ap.add_argument('--enable_mode_credit', action='store_true')
    ap.add_argument('--credit_budget', type=int, default=4)
    ap.add_argument('--credit_batch_size', type=int, default=32)
    ap.add_argument('--enable_mode_relation', action='store_true')
    ap.add_argument('--enable_program_search', action='store_true')
    ap.add_argument('--enable_program_search_active', action='store_true')
    ap.add_argument('--enable_relation_rerank', action='store_true')
    ap.add_argument('--load_self_memory', action='store_true')
    ap.add_argument('--self_memory_dir', default='')
    ap.add_argument('--enable_auto_rollback', action='store_true')
    ap.add_argument('--auto_rollback_patience', type=int, default=3)
    ap.add_argument('--program_search_patience', type=int, default=8)
    ap.add_argument('--program_burst_duration', type=int, default=5)
    ap.add_argument('--program_temperature_mult', type=float, default=1.4)
    ap.add_argument('--program_collapse_threshold', type=float, default=0.70)
    ap.add_argument('--input_conditioned_cost', type=float, default=0.0)
    ap.add_argument('--enable_program_dynamics', action='store_true', default=True)
    ap.add_argument('--enable_best_archive', action='store_true', default=True)
    ap.add_argument('--enable_token_program_brain', action='store_true', default=True)
    ap.add_argument('--enable_mode_categories', action='store_true', default=True)
    ap.add_argument('--log_every', type=int, default=50)
    args=ap.parse_args()
    device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu')
    if args.task == 'byte_lm':
        data=collect_text(args.corpus_dir, args.max_bytes)
        split=max(args.seq_len+2, int(len(data)*0.9))
        train=ByteLMDataset(data[:split], args.seq_len, args.train_samples, 0)
        val=ByteLMDataset(data[split:] if len(data)-split > args.seq_len+2 else data, args.seq_len, args.val_samples, 123)
        corpus_bytes = len(data)
    else:
        data=b''
        train=LongRangeRecallDataset(args.seq_len, args.train_samples, args.vocab, args.memory_delay, args.memory_pairs, 0)
        val=LongRangeRecallDataset(args.seq_len, args.val_samples, args.vocab, args.memory_delay, args.memory_pairs, 10000000)
        corpus_bytes = 0
    tr=DataLoader(train,batch_size=args.batch,shuffle=True,num_workers=2,pin_memory=device.type=='cuda')
    va=DataLoader(val,batch_size=args.batch,shuffle=False,num_workers=2,pin_memory=device.type=='cuda')
    model=TokenLMSwap(args).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    bank = getattr(getattr(model, 'res', None), 'bank', None)
    mode_names = list(getattr(bank, 'mode_names', [])) if bank is not None else []
    n_modes_for_memory = len(mode_names) if mode_names else args.n_modes
    n_eff_steps = args.macro_steps * args.micro_steps
    feedback_memory = ModeFeedbackMemory(n_steps_by_level={'effective': n_eff_steps, 'macro': args.macro_steps, 'micro': n_eff_steps}, n_modes=n_modes_for_memory, device=device) if args.enable_mode_credit else None
    relation_memory = ModeRelationMemory(n_modes_for_memory, device=device) if args.enable_mode_relation else None
    program_search = TargetedProgramSearch(args.program_search_patience, args.program_burst_duration, args.program_collapse_threshold, args.program_temperature_mult) if args.enable_program_search else None
    out=Path(args.out_dir)/args.variant; out.mkdir(parents=True,exist_ok=True)
    memory_dir = Path(args.self_memory_dir) if args.self_memory_dir else out
    if args.load_self_memory:
        if feedback_memory is not None:
            print('load_feedback_memory', feedback_memory.load_json(memory_dir/'mode_feedback_memory.json'), flush=True)
        if relation_memory is not None:
            print('load_relation_memory', relation_memory.load_json(memory_dir/'mode_relation_memory.json'), flush=True)
    (out/'config.json').write_text(json.dumps({**vars(args), 'corpus_bytes': corpus_bytes}, indent=2))
    archive = BestProgramArchive(out) if args.enable_best_archive else None
    rows=[]; best=999.0; bad_epochs=0; active_train_intervention=None; rollback_events=[]
    print('variant',args.variant,'task',args.task,'bytes',corpus_bytes,'params',sum(p.numel() for p in model.parameters() if p.requires_grad),flush=True)
    for ep in range(1,args.epochs+1):
        t=time.time()
        tr_loss,tr_acc,_,_,_,_=run_epoch(model,tr,opt,args,device,True, active_train_intervention)
        va_loss,va_acc,brain,token_brain,credit_x,credit_y=run_epoch(model,va,opt,args,device,False)
        row={'epoch':ep,'train_loss':tr_loss,'train_acc':tr_acc,'val_loss':va_loss,'val_acc':va_acc,'ppl':math.exp(min(20,va_loss)),'sec':time.time()-t}
        # Memory costs are already included in loss during training. Compact values live in PROGRAM_DYNAMICS via stats.
        rows.append(row)
        print(json.dumps(row),flush=True)
        if brain is not None:
            summary=brain.finalize(ep, {'val_acc': va_acc, 'final_acc': va_acc})
            if token_brain is not None:
                summary['token_program_brain'] = token_brain.finalize()
            bank = getattr(getattr(model, 'res', None), 'bank', None)
            if bank is not None and hasattr(bank, 'mode_names'):
                weights = None
                try:
                    weights = [float(v) for v in summary.get('program', {}).get('all_weights', {}).values()]
                except Exception:
                    weights = None
                summary.update(summarize_categories(list(bank.mode_names), weights if weights and len(weights)==len(bank.mode_names) else None))
            if credit_x is not None and args.credit_modes > 0:
                def _eval_loss(logits, stats, yy): return seq_loss(logits, yy)
                summary['true_counterfactual_credit']=counterfactual_mode_credit(model, credit_x, credit_y, _eval_loss, min(args.credit_modes, args.n_modes))
            if credit_x is not None and args.enable_mode_credit:
                cx = credit_x[:args.credit_batch_size]
                cy = credit_y[:args.credit_batch_size]
                summary['step_local_credit'] = run_step_local_credit(model, cx, cy, args, feedback_memory, relation_memory)
                if feedback_memory is not None:
                    summary.update(feedback_memory.summary(mode_names))
                if relation_memory is not None:
                    summary.update(relation_memory.summary(mode_names))
            
            if program_search is not None:
                summary['program_search'] = program_search.update(va_acc, summary)
                if args.enable_program_search_active:
                    relation_summary = summary if args.enable_relation_rerank else None
                    active_train_intervention = program_search.build_intervention(summary, mode_names, summary, relation_summary)
                    summary['active_train_intervention_next'] = active_train_intervention
                else:
                    active_train_intervention = None
            write_program_brain_outputs(out, summary, best_acc=-best, row_acc=va_acc)
        if va_loss < best:
            best=va_loss; bad_epochs=0; torch.save({'model':model.state_dict(),'args':vars(args),'row':row},out/'best.pt')
            try:
                if archive is not None and brain is not None:
                    archive.maybe_update(ep, va_loss, va_acc, model, summary if 'summary' in locals() else {}, row)
            except Exception as e:
                print('archive_warn', repr(e), flush=True)
            print('best_loss',best,flush=True)
        else:
            bad_epochs += 1
            if args.enable_auto_rollback and bad_epochs >= args.auto_rollback_patience and (out/'best.pt').exists():
                ckpt=torch.load(out/'best.pt', map_location=device)
                model.load_state_dict(ckpt['model'])
                opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
                rollback_events.append({'epoch':ep,'to_best_loss':best,'reason':'patience'})
                print('auto_rollback_to_best', best, 'at_epoch', ep, flush=True)
                bad_epochs=0
        if feedback_memory is not None:
            feedback_memory.save_json(out/'mode_feedback_memory.json')
        if relation_memory is not None:
            relation_memory.save_json(out/'mode_relation_memory.json')
        if rollback_events:
            (out/'rollback_events.json').write_text(json.dumps(rollback_events, indent=2), encoding='utf-8')
        with (out/'history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('saved',out,'best_loss',best,flush=True)

if __name__=='__main__':
    main()
