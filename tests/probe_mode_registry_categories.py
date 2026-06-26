from resonant_block_lab.self_learning.mode_registry import summarize_categories
out=summarize_categories(['identity','learned_causal_kernel','input_conditioned','low_rank_global','null_1'],[.1,.2,.5,.1,.1])
assert out['category_usage']['adaptive'] > .4
assert out['category_top_share'] > .4
print('probe_mode_registry_categories OK')
