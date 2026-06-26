from resonant_block_lab.self_learning.program_search import TargetedProgramSearch
ps=TargetedProgramSearch(patience=1, collapse_threshold=0.1)
summary={'category_usage':{'adaptive':0.9}, 'program':{'all_weights':{'identity':0.5,'delta_causal':0.3}}}
out=ps.update(0.1, summary)
assert out['in_burst']
inter=ps.build_intervention(summary, ['identity','delta_causal'], {'mode_feedback_top_by_level':{'effective':[]}}, None)
assert inter.get('mode_logit_bias')
print('probe_program_search_active OK')
