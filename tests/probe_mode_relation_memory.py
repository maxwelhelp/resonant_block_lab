from resonant_block_lab.self_learning.mode_relation_memory import ModeRelationMemory
r=ModeRelationMemory(4)
r.update([0,1,2], 0.5)
r.update([1,3], -0.25)
out=r.summary(['a','b','c','d'])
assert out['relation_observed_pairs'] > 0
assert out['relation_pair_count_total'] > 0
print('probe_mode_relation_memory OK')
