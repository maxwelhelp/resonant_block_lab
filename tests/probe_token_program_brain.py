import torch
from resonant_block_lab.self_learning.token_program_brain import TokenProgramBrain
logits=torch.randn(2,4,7)
targets=torch.tensor([[1,2,-100,3],[2,3,4,-100]])
stats={'q_eff_history':torch.softmax(torch.randn(2,3,5),-1)}
b=TokenProgramBrain([f'm{i}' for i in range(5)])
b.add_batch(stats,logits,targets)
out=b.finalize()
assert out['token_count']==6
assert 'token_mode_usage' in out
print('probe_token_program_brain OK')
