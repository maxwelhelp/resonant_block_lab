import tempfile
from pathlib import Path
from resonant_block_lab.self_learning.mode_feedback_memory import ModeFeedbackMemory
from resonant_block_lab.self_learning.mode_relation_memory import ModeRelationMemory
with tempfile.TemporaryDirectory() as d:
    d=Path(d)
    fb=ModeFeedbackMemory(n_steps_by_level={'effective':2}, n_modes=3)
    fb.update('effective',1,2,0.5)
    fb.save_json(d/'fb.json')
    fb2=ModeFeedbackMemory(n_steps_by_level={'effective':2}, n_modes=3)
    assert fb2.load_json(d/'fb.json')
    assert fb2.summary()['mode_feedback_count']==1
    rel=ModeRelationMemory(3)
    rel.update([0,1],0.25)
    rel.save_json(d/'rel.json')
    rel2=ModeRelationMemory(3)
    assert rel2.load_json(d/'rel.json')
    assert rel2.summary()['relation_pair_count_total'] > 0
print('probe_feedback_relation_persistence OK')
