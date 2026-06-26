from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Any
import math

@dataclass
class ModeInfo:
    name: str
    category: str
    cost: float = 1.0
    causal: bool = True
    uses_memory: bool = False
    learned: bool = False
    description: str = ""

def mode_category(name: str) -> str:
    if name == 'identity' or name.startswith('null_'):
        return 'preserve'
    if name in ('learned_causal_kernel', 'delta_causal'):
        return 'causal_local'
    if name == 'learned_causal_pool':
        return 'causal_pool'
    if name == 'input_conditioned':
        return 'adaptive'
    if name in ('low_rank_global', 'global_mean'):
        return 'global'
    if name in ('symbolic_additive', 'symbolic_product'):
        return 'symbolic_matrix'
    if 'memory' in name or name in ('memory_read', 'memory_write', 'slot_memory'):
        return 'memory'
    if name.startswith('learned_'):
        return 'learned_other'
    return 'other'

def mode_description(name: str) -> str:
    desc = {
        'identity': 'preserve current field',
        'learned_causal_kernel': 'learned causal offset mixture over past positions',
        'learned_causal_pool': 'learned causal pooling scale over past context',
        'delta_causal': 'current field minus learned causal read',
        'input_conditioned': 'adaptive causal gate from current field state',
        'low_rank_global': 'low-rank global/context compression',
        'symbolic_additive': 'symbolic additive matrix program over current basis',
        'symbolic_product': 'symbolic product matrix program over current basis',
    }
    if name.startswith('null_'):
        return 'explicit no-op capacity slot, not a learned operator'
    return desc.get(name, '')

def build_mode_registry(mode_names: Iterable[str]) -> Dict[str, ModeInfo]:
    out: Dict[str, ModeInfo] = {}
    for n in mode_names:
        cat = mode_category(n)
        out[n] = ModeInfo(
            name=n,
            category=cat,
            cost=0.0 if n.startswith('null_') else (1.5 if cat == 'memory' else 1.0),
            causal=(n != 'shift_left'),
            uses_memory=(cat == 'memory'),
            learned=(n.startswith('learned_') or n in ('input_conditioned', 'low_rank_global')),
            description=mode_description(n),
        )
    return out

def summarize_categories(mode_names: List[str], weights: List[float] | None = None) -> Dict[str, Any]:
    reg = build_mode_registry(mode_names)
    if weights is None:
        weights = [1.0 / max(1, len(mode_names))] * len(mode_names)
    cat: Dict[str, float] = {}
    for n, w in zip(mode_names, weights):
        cat[reg[n].category] = cat.get(reg[n].category, 0.0) + float(w)
    total = sum(cat.values()) or 1.0
    cat = {k: v / total for k, v in cat.items()}
    top_share = max(cat.values()) if cat else 0.0
    active = sum(1 for v in cat.values() if v > 1e-3)
    entropy = -sum(v * math.log(max(v, 1e-12)) for v in cat.values())
    return {
        'mode_registry': {k: asdict(v) for k, v in reg.items()},
        'category_usage': cat,
        'category_entropy': entropy,
        'category_top_share': top_share,
        'category_active_count': active,
    }
