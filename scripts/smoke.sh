#!/usr/bin/env bash
set -euo pipefail
python -m py_compile resonant_block_lab/resonant_block.py train_synthetic.py examples/dropin_transformer_example.py
PYTHONPATH=. python examples/dropin_transformer_example.py
PYTHONPATH=. python train_synthetic.py --epochs 1 --limit_batches 2 --batch 64 --length 64 --dim 32 --n_modes 6 --micro_steps 2 --device cuda
