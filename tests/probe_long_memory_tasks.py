from train_token_lm_swap import LongRangeRecallDataset, IGNORE_INDEX
x,y=LongRangeRecallDataset(seq_len=64,n_samples=1,vocab=32,delay=16,pairs=2)[0]
assert x.shape==y.shape
assert (y != IGNORE_INDEX).sum().item()==2
print('probe_long_memory_tasks OK')
