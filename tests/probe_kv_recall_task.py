from train_token_lm_swap import AssociativeKVRecallDataset, IGNORE_INDEX

ds = AssociativeKVRecallDataset(seq_len=256, n_samples=3, vocab=128, n_kv=24, n_queries=6, offset=0, noise=True)
x, y = ds[0]
assert x.shape == y.shape
assert (y != IGNORE_INDEX).sum().item() == 6
# Every supervised target is a value token, not a key/marker token.
vals = y[y != IGNORE_INDEX]
assert int(vals.min()) >= ds.val_lo and int(vals.max()) < ds.val_hi
# Query positions are marked immediately before supervised key positions.
idxs = (y != IGNORE_INDEX).nonzero().flatten().tolist()
for pos in idxs:
    assert int(x[pos-1]) == ds.QUERY
    assert ds.key_lo <= int(x[pos]) < ds.key_hi
print('probe_kv_recall_task OK', 'targets', len(idxs), 'key_range', (ds.key_lo, ds.key_hi), 'val_range', (ds.val_lo, ds.val_hi))
