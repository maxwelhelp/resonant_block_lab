from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class ProgramSearchState:
    best_metric: float = -1e9
    steps_since_improvement: int = 0
    in_burst: bool = False
    burst_steps_remaining: int = 0
    burst_reason: str = ''
    crystallization_score: float = 0.0

class TargetedProgramSearch:
    def __init__(self, patience:int=8, burst_duration:int=5, collapse_threshold:float=0.70, temperature_mult:float=1.4):
        self.state=ProgramSearchState()
        self.patience=int(patience); self.burst_duration=int(burst_duration)
        self.collapse_threshold=float(collapse_threshold); self.temperature_mult=float(temperature_mult)
    def build_intervention(self, summary: Dict[str, Any] | None, mode_names, feedback_summary: Dict[str, Any] | None = None, relation_summary: Dict[str, Any] | None = None, max_targets: int = 4) -> Dict[str, Any]:
        """Build a safe next-epoch intervention from diagnostics.

        This does not mutate the model. It only asks the already-supported forward
        intervention hook to slightly bias useful modes at specific effective steps.
        """
        st = self.state
        if not st.in_burst:
            return {}
        mode_to_id = {n: i for i, n in enumerate(mode_names or [])}
        bias = {}
        targets = []
        fb = feedback_summary or {}
        eff_rows = ((fb.get('mode_feedback_top_by_level') or {}).get('effective') or [])
        for r in eff_rows[:max_targets]:
            name = r.get('mode')
            if name in mode_to_id and float(r.get('gain_ema', 0.0)) >= float(r.get('regret_ema', 0.0)):
                step = int(r.get('step', 0)); mid = mode_to_id[name]
                bias[f'effective:{step}:{mid}'] = float(bias.get(f'effective:{step}:{mid}', 0.0)) + 0.15
                targets.append({'level':'effective','step':step,'mode':name,'reason':'feedback_gain'})
        if relation_summary:
            for a,b,v in (relation_summary.get('relation_co_pos_top') or [])[:max_targets]:
                for name in (a,b):
                    if name in mode_to_id:
                        # Global gentle bias across all known effective steps.
                        for r in eff_rows[:max_targets] or [{'step':0}]:
                            step = int(r.get('step', 0)); mid = mode_to_id[name]
                            bias[f'effective:{step}:{mid}'] = float(bias.get(f'effective:{step}:{mid}', 0.0)) + 0.05
                            targets.append({'level':'effective','step':step,'mode':name,'reason':'relation_support'})
        if not bias:
            # Cold-start fallback: burst must still do something, but softly.
            prog = (summary or {}).get('program', {})
            all_w = prog.get('all_weights') or {}
            ordered = sorted(all_w.items(), key=lambda kv: float(kv[1]), reverse=True)
            steps = [int(r.get('step', 0)) for r in eff_rows[:max_targets]] or [0]
            for name, _ in ordered[:max_targets]:
                if name in mode_to_id:
                    for step in steps[:max_targets]:
                        mid = mode_to_id[name]
                        bias[f'effective:{step}:{mid}'] = float(bias.get(f'effective:{step}:{mid}', 0.0)) + 0.03
                        targets.append({'level':'effective','step':step,'mode':name,'reason':'cold_start_usage'})
        return {'mode_logit_bias': bias, 'search_targets': targets[:max_targets*2], 'source': 'targeted_program_search'}

    def update(self, metric:float, summary:Dict[str,Any] | None=None) -> Dict[str,Any]:
        summary=summary or {}; st=self.state
        improved=float(metric)>st.best_metric
        if improved:
            st.best_metric=float(metric); st.steps_since_improvement=0
            st.in_burst=False; st.burst_steps_remaining=0; st.burst_reason=''
            st.crystallization_score=min(1.0, st.crystallization_score+0.05)
        else:
            st.steps_since_improvement += 1
        top_share=0.0
        try:
            top_share=max((summary.get('category_usage') or {}).values())
        except Exception:
            top_share=0.0
        if not st.in_burst and (st.steps_since_improvement>=self.patience or top_share>self.collapse_threshold):
            st.in_burst=True; st.burst_steps_remaining=self.burst_duration
            st.burst_reason='collapse' if top_share>self.collapse_threshold else 'plateau'
            st.crystallization_score=max(0.0, st.crystallization_score-0.1)
        elif st.in_burst:
            st.burst_steps_remaining -= 1
            if st.burst_steps_remaining<=0:
                st.in_burst=False; st.burst_reason=''
        out=asdict(st)
        out.update({
            'temperature_multiplier': self.temperature_mult if st.in_burst else 1.0,
            'mode_noise': 0.03 if st.in_burst else 0.0,
            'targeted_steps': [],
            'targeted_modes': [],
            'observed_category_top_share': top_share,
        })
        return out
