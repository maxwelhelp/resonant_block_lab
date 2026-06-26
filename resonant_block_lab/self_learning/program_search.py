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
