import torch
import torch.nn as nn
from resonant_block_lab import ResonantBlockConfig, ParallelResonantBlock

D = 128
base_layer = nn.TransformerEncoderLayer(d_model=D, nhead=4, batch_first=True)
config = ResonantBlockConfig(dim=D, n_modes=8, micro_steps=3, gate_init=-5.0)
layer = ParallelResonantBlock(config, base_block=base_layer)

x = torch.randn(2, 64, D)
y, stats = layer(x, return_stats=True)
print('x', x.shape, 'y', y.shape)
print('parallel_gate', float(stats['parallel_gate']))
print('replayable_program', stats['program'])
