import tempfile, torch
from torch import nn
from resonant_block_lab.self_learning.best_archive import BestProgramArchive
with tempfile.TemporaryDirectory() as d:
    a=BestProgramArchive(d)
    assert a.maybe_update(1,1.0,0.5,nn.Linear(2,2),{'program':{'top_modes':[('x',1.0)]}}, {'x':1})
print('probe_best_archive OK')
