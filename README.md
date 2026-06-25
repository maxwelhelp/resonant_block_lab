# Resonant Block Lab

A small PyTorch lab for **safe drop-in resonant sequence blocks**.

The core idea is to add a resonant branch in parallel to an existing attention or FFN block:

```text
x ───────────────► base block ───────────────► base_out
                       │
                       ▼
                ResonantBlock(base_out)
                       │
                       ▼
y = base_out + gate · resonant_delta
```

The gate is initialized near zero, so the original network does not collapse at the start of training.

## Why this exists

Attention performs global pairwise mixing with `O(L^2)` token interactions. `ResonantBlock` uses a controller that selects a mixture of replayable sequence operators:

- identity
- shift left/right
- local average
- global mean broadcast
- high-pass residual
- learned depthwise modes

This gives a different inductive bias: wave-like iterative refinement instead of pairwise attention.

## Install / smoke

```bash
cd resonant_block_lab
bash scripts/smoke.sh
```

Or run the toy classifier:

```bash
python train_synthetic.py --epochs 5 --device cuda
```

## Drop-in usage

```python
import torch.nn as nn
from resonant_block_lab import ResonantBlockConfig, ParallelResonantBlock

base_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
cfg = ResonantBlockConfig(dim=128, n_modes=8, micro_steps=3, gate_init=-5.0)
layer = ParallelResonantBlock(cfg, base_block=base_layer)

y = layer(x)  # x/y: [B, L, D]
```

## Replayable program

Every resonant micro-step selects a soft program:

```text
program_m = q0·identity + q1·shift_left + q2·shift_right + q3·local_avg + ...
```

The block can return this program with `return_stats=True`. It is executable because it is exactly the mixture of operators used in forward.

## Why not MPS/MPO first?

For short or medium sequence length, dynamic local operators are usually faster than tensor-network contraction on a GPU. MPO/MPS modes are planned for larger lengths (`L >= 128/256`) or for compact replayable matrix programs.

## Roadmap

- [x] Safe parallel adapter with near-zero gate
- [x] Dynamic sequence length support
- [x] Micro-step auxiliary projections for toy tasks
- [x] Replayable operator-bank program summaries
- [ ] Low-rank operator bank
- [ ] MPO/MPS operator modes
- [ ] Attention baseline benchmark
- [ ] Adaptive early exit by convergence delta
