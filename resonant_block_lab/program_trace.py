from __future__ import annotations

from typing import Any, Dict, List, Sequence


def default_mode_names(n: int) -> List[str]:
    base = ['identity', 'shift_left', 'shift_right', 'local_avg', 'global_mean', 'highpass']
    while len(base) < n:
        base.append(f'learned_depthwise_{len(base)-5}')
    return base[:n]


OP_DESCRIPTIONS = {
    'identity': 'keep current field / residual memory path',
    'shift_left': 'move local evidence left along sequence axis',
    'shift_right': 'move local evidence right along sequence axis',
    'local_avg': 'smooth neighboring tokens / stabilize local field',
    'global_mean': 'broadcast global summary to all positions',
    'highpass': 'emphasize local contrast / edges / sudden changes',
    'symbolic_additive': 'matrix program W = sum_i c_i O_i over readable operator basis',
    'symbolic_product': 'ordered matrix program W = product_s (I + sum_i c_si O_i)',
    'low_rank_global': 'learned low-rank global matrix/basis summary',
    'input_conditioned': 'operator generated from current input/field statistics',
    'prev1': 'causal previous-token shift, no future leakage',
    'prev2': 'causal two-token memory shift',
    'prev4': 'causal four-token memory shift',
    'causal_avg3': 'causal local average over current and previous tokens',
    'prefix_mean': 'causal prefix summary / cheap global past context',
    'delta_prev': 'token transition / difference from previous state',
    'learned_causal_kernel': 'learned causal offset mixture; can discover prev1/prev2/long offsets',
    'learned_causal_pool': 'learned causal EMA/pooling over past context; not fixed prefix_mean',
    'delta_causal': 'difference between current state and learned causal kernel read',
    'memory_read': 'external costed slot-memory read, controlled by memory gates',
    'null_': 'explicit no-op capacity slot, not a learned operator',
}


def _top_modes(vec: Sequence[float], names: Sequence[str], k: int = 4):
    pairs = [(names[i], float(v)) for i, v in enumerate(vec)]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:min(k, len(pairs))]


def _fmt_modes(pairs) -> str:
    return ' + '.join(f'{name}:{weight:.3f}' for name, weight in pairs)


def _role_from_modes(pairs) -> str:
    names = [p[0] for p in pairs[:3]]
    if 'global_mean' in names and ('shift_left' in names or 'shift_right' in names):
        return 'global context + local alignment'
    if 'shift_left' in names and 'shift_right' in names:
        return 'bidirectional local alignment / shift compensation'
    if 'local_avg' in names and ('shift_left' in names or 'shift_right' in names):
        return 'local smoothing after evidence transport'
    if 'highpass' in names:
        return 'contrast / boundary extraction'
    if 'global_mean' in names:
        return 'global summary broadcast'
    if 'identity' in names:
        return 'information preservation / weak update'
    return 'mixed resonant update'


def build_interpretable_program(summary: Dict[str, Any], mode_names: List[str] | None = None, topk: int = 4) -> Dict[str, Any]:
    prog = summary.get('program') or {}
    mp = prog.get('matrix_programs') or {}
    if mode_names is None and isinstance(mp, dict) and mp.get('mode_names'):
        mode_names = list(mp.get('mode_names'))
    eff = summary.get('effective_mode_schedule_mean_by_microstep') or []
    micro_q = summary.get('micro_mode_schedule_mean_by_microstep') or []
    macro_micro = summary.get('micro_macro_mode_schedule_mean_by_microstep') or []
    macro_q = summary.get('macro_mode_schedule_mean_by_step') or []
    gates = summary.get('gate_schedule_alpha_beta_gamma_eta_rho_macroMix') or []
    macro_acc = summary.get('macro_acc_by_step') or []
    micro_acc = summary.get('micro_acc_by_microstep') or summary.get('micro_acc_by_microstep_flat') or []
    macro_delta = summary.get('macro_delta_mean_by_step') or []
    micro_delta = summary.get('micro_delta_mean_by_microstep') or []

    n_modes = 0
    for arr in [eff, micro_q, macro_micro, macro_q]:
        if arr:
            n_modes = len(arr[0])
            break
    names = mode_names or default_mode_names(n_modes)

    t_count = len(macro_q) or len(macro_acc) or 0
    total_micro = len(eff) or len(micro_q) or len(macro_micro) or len(micro_acc) or 0
    if t_count <= 0 and total_micro > 0:
        t_count = 1
    micro_per_macro = max(1, total_micro // max(1, t_count)) if total_micro else 0

    macros = []
    for t in range(t_count):
        mq = macro_q[t] if t < len(macro_q) else []
        macro_top = _top_modes(mq, names, topk) if mq else []
        steps = []
        for m in range(micro_per_macro):
            idx = t * micro_per_macro + m
            ev = eff[idx] if idx < len(eff) else []
            micro = micro_q[idx] if idx < len(micro_q) else []
            mm = macro_micro[idx] if idx < len(macro_micro) else []
            top = _top_modes(ev, names, topk) if ev else []
            gate = gates[idx] if idx < len(gates) else []
            steps.append({
                'micro_index': m + 1,
                'flat_index': idx,
                'role': _role_from_modes(top),
                'executed_top_modes': top,
                'micro_controller_top_modes': _top_modes(micro, names, topk) if micro else [],
                'macro_plan_seen_by_micro_top_modes': _top_modes(mm, names, topk) if mm else [],
                'gates_alpha_beta_gamma_eta_rho_macroMix': gate,
                'micro_acc': micro_acc[idx] if idx < len(micro_acc) else None,
                'micro_delta': micro_delta[idx] if idx < len(micro_delta) else None,
            })
        macros.append({
            'macro_index': t + 1,
            'role': _role_from_modes(macro_top or (steps[-1]['executed_top_modes'] if steps else [])),
            'macro_plan_top_modes': macro_top,
            'macro_acc': macro_acc[t] if t < len(macro_acc) else None,
            'macro_delta': macro_delta[t] if t < len(macro_delta) else None,
            'micro_steps': steps,
        })

    return {
        'mode_names': names,
        'operator_legend': {n: OP_DESCRIPTIONS.get(n, 'learned depthwise local filter') for n in names},
        'matrix_programs': mp,
        'macro_steps': macros,
    }


def render_program_trace_md(trace: Dict[str, Any]) -> str:
    lines = []
    lines.append('# Interpretable Resonant Program Trace')
    lines.append('')
    lines.append('## Operator legend')
    for name, desc in trace.get('operator_legend', {}).items():
        lines.append(f'- `{name}`: {desc}')
    lines.append('')
    if trace.get('matrix_programs'):
        lines.append('## Matrix-program formulas')
        mp = trace.get('matrix_programs') or {}
        if mp.get('symbolic_additive'):
            lines.append('symbolic_additive: ' + ' + '.join(f'{name}:{float(w):.3f}' for name, w in mp.get('symbolic_additive', [])[:8]))
        if mp.get('symbolic_product'):
            for step in mp.get('symbolic_product', [])[:8]:
                ops = ' + '.join(f'{name}:{float(w):.3f}' for name, w in step.get('ops', [])[:8])
                lines.append(f"product step {step.get('step')} scale={float(step.get('step_scale',0.0)):.3f}: {ops}")
        if mp.get('program_rank') is not None:
            lines.append(f"low_rank program_rank={mp.get('program_rank')}")
        lines.append('')
    lines.append('## Layer program')
    for macro in trace.get('macro_steps', []):
        mi = macro['macro_index']
        lines.append(f"### Macro step {mi}: {macro.get('role','')}")
        if macro.get('macro_plan_top_modes'):
            lines.append(f"macro plan: {_fmt_modes(macro['macro_plan_top_modes'])}")
        if macro.get('macro_acc') is not None:
            lines.append(f"macro_acc={macro['macro_acc']:.5f}  macro_delta={macro.get('macro_delta')}")
        for micro in macro.get('micro_steps', []):
            lines.append(f"- micro {micro['micro_index']}: {micro.get('role','')}")
            if micro.get('executed_top_modes'):
                lines.append(f"  executed: {_fmt_modes(micro['executed_top_modes'])}")
            if micro.get('macro_plan_seen_by_micro_top_modes'):
                lines.append(f"  macro_plan_seen: {_fmt_modes(micro['macro_plan_seen_by_micro_top_modes'])}")
            if micro.get('micro_controller_top_modes'):
                lines.append(f"  micro_choice: {_fmt_modes(micro['micro_controller_top_modes'])}")
            if micro.get('gates_alpha_beta_gamma_eta_rho_macroMix'):
                g = micro['gates_alpha_beta_gamma_eta_rho_macroMix']
                names = ['alpha','beta','gamma','eta','rho','macroMix']
                lines.append('  gates: ' + ', '.join(f'{names[i]}={float(v):.3f}' for i, v in enumerate(g[:len(names)])))
            if micro.get('micro_acc') is not None:
                lines.append(f"  micro_acc={micro['micro_acc']:.5f}  micro_delta={micro.get('micro_delta')}")
        lines.append('')
    return '\n'.join(lines)
