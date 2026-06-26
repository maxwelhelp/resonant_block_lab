from resonant_block_lab.self_learning.mode_feedback_memory import ModeFeedbackMemory
m=ModeFeedbackMemory(n_steps_by_level={'effective':3}, n_modes=4)
m.update('effective',1,2,0.5)
m.update('effective',1,3,-0.25)
out=m.summary(['a','b','c','d'])
assert out['mode_feedback_count']==2
print('probe_mode_feedback_memory OK')
