import torch
from resonant_block_lab.fractal_sequence import FractalResonantSequenceBlock, FractalResonantConfig
cfg=FractalResonantConfig(dim=16,n_modes=10,macro_steps=2,micro_steps=2,enable_token_primitives=True,enable_matrix_program=True,enable_lowrank=True,enable_input_primitive=True)
m=FractalResonantSequenceBlock(cfg).eval()
x=torch.randn(2,8,16)
with torch.no_grad():
    _,s=m(x,return_stats=True,intervention={'disable_modes':[{'level':'effective','step':2,'mode':1}]})
assert float(s['q_eff_history'][:,2,1].abs().max()) < 1e-6
print('probe_step_local_ablation OK')
